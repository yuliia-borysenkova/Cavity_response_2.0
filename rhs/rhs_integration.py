import os
import numpy as np
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from scipy import integrate
from scipy.constants import c as c_cnst
from rhs.utils import CubicSplineInterp
import vegas

def NL_grid_chunks(hpl_dd, hcr_dd, t_data, tmin, tmax, Nt, tshift,
                   density_boost=1.0, probe_factor=10, thr_frac=0.005):
    """
    Create non-uniform time grid with higher density in signal-rich regions.
    
    Args:
        hpl_dd, hcr_dd: Gravitational wave strain data
        t_data: Time array for strain data
        tmin, tmax: Time integration bounds
        Nt: Total number of time points
        tshift: Time shift (L/c)
        density_boost: Factor to increase grid density in signal region
        probe_factor: Resolution multiplier for initial signal detection
        thr_frac: Fraction of max amplitude to define signal region
        
    Returns:
        Tuple of (pre, mid, post) time arrays for regions before, during, after signal
    """
    # Compute strain amplitude and create interpolation function
    amp = np.sqrt(hpl_dd**2 + hcr_dd**2)
    amp_fn = CubicSplineInterp(t_data, amp, amp[0], amp[-1])
    
    # Create fine probe grid to detect signal regions
    t_probe = np.linspace(tmin, tmax, probe_factor * Nt)
    # Identify times where amplitude exceeds threshold
    mask = (amp_fn(t_probe) > thr_frac * np.max(amp)) & (amp_fn(t_probe - tshift) > thr_frac * np.max(amp))
    
    # Find signal boundaries with time shift buffer
    idx = np.where(mask)[0]
    t_start = max(t_probe[idx[0]] - tshift, tmin) if np.any(mask) else tmin
    t_end = min(t_probe[idx[-1]] + tshift, tmax) if np.any(mask) else tmax
    
    # Calculate weighted widths for each region
    w_pre, w_mid, w_post = t_start - tmin, density_boost * (t_end - t_start), tmax - t_end
    w_sum = w_pre + w_mid + w_post or 1
    
    # Allocate points proportionally, with minimum counts
    N_pre = max(2, int(Nt * w_pre / w_sum))
    N_mid = max(4, int(Nt * w_mid / w_sum))
    N_post = max(2, Nt - N_pre - N_mid)
    
    # Return three time arrays: before signal, during signal, after signal
    return (np.linspace(tmin, t_start, N_pre, endpoint=False),
            np.linspace(t_start, t_end, N_mid, endpoint=False),
            np.linspace(t_end, tmax, N_post))


def rhs_integral(t, x_par_arr, E_plus, E_cross, hplus_dd, hcross_dd, method="quad"):

    def integrand(x_par):
        return E_plus(x_par) * hplus_dd(t - x_par/c_cnst) + E_cross(x_par) * hcross_dd(t - x_par/c_cnst)

    if method == "vegas":

        integ = vegas.Integrator([x_par_arr.min(), x_par_arr.max()])
        integ(integrand, nitn=3, neval=5_000)   # warmup
        res = integ(integrand, nitn=7, neval=15_000)

        return res.mean
        
    elif method == "quad":
        return integrate.quad(integrand, x_par_arr.min(), x_par_arr.max(), limit=200)[0]
        
    else:
        print("Unknown integration method.")
        return None

def integrate_chunk(ts, method, rhs_func, nproc):
    if len(ts) == 0:
        return []

    with Pool(nproc) as pool:
        return list(tqdm(pool.imap(rhs_func, ts), total=len(ts), desc=f"RHS ({method})"))

def compute_rhs_time_series(t_data, x_par_arr, E_plus, E_cross, hplus_dd, hcross_dd, Nt, L, method="quad", nproc=1, NLGrid=False, density_boost=5.0):

    tshift = L / c_cnst
    tmin = t_data[0] - tshift
    tmax = t_data[-1] + tshift

    E_plus_func = CubicSplineInterp(x_par_arr, E_plus, 0.0, 0.0)
    E_cross_func = CubicSplineInterp(x_par_arr, E_cross, 0.0, 0.0)
    hplus_dd_func = CubicSplineInterp(t_data, hplus_dd, 0.0, 0.0)
    hcross_dd_func = CubicSplineInterp(t_data, hplus_dd, 0.0, 0.0)
    
    rhs_func_user = partial(
        rhs_integral, x_par_arr=x_par_arr,
        E_plus=E_plus_func, E_cross=E_cross_func,
        hplus_dd=hplus_dd_func, hcross_dd=hcross_dd_func,
        method=method,
    )

    rhs_func_vegas = partial(
        rhs_integral, x_par_arr=x_par_arr,
        E_plus=E_plus_func, E_cross=E_cross_func,
        hplus_dd=hplus_dd_func, hcross_dd=hcross_dd_func,
        method="vegas",
    )

    print(f"[INFO] Computing RHS...")

    if NLGrid:
        ts_pre, ts_mid, ts_post = NL_grid_chunks(
            hplus_dd, hcross_dd,
            t_data, tmin, tmax, Nt, tshift,
            density_boost,
        )

        RHS_pre  = integrate_chunk(ts_pre,  method, rhs_func_user,  nproc)
        RHS_mid  = integrate_chunk(ts_mid,  "vegas", rhs_func_vegas, nproc)
        RHS_post = integrate_chunk(ts_post, method, rhs_func_user,  nproc)

        ts  = np.concatenate([ts_pre, ts_mid, ts_post])
        RHS = np.array(RHS_pre + RHS_mid + RHS_post)

    else:
        ts = np.linspace(tmin, tmax, Nt)
        RHS = integrate_chunk(ts, method, rhs_func_user, nproc)
        RHS = np.array(RHS)

    print(f"[INFO] RHS integration complete.")

    return ts, RHS
