import numpy as np
from scipy.constants import c as c_cnst
from scipy.signal import fftconvolve


def compute_rhs_time_series(t_data, x_par_arr, E_plus, E_cross, hplus_dd, hcross_dd, Nt, L, label="RHS"):
    tshift = L / c_cnst
    tmin = t_data[0] - tshift
    tmax = t_data[-1] + tshift

    ts = np.linspace(tmin, tmax, Nt)
    dt = ts[1] - ts[0]

    hplus  = np.interp(ts, t_data, hplus_dd,  left=0.0, right=0.0)
    hcross = np.interp(ts, t_data, hcross_dd, left=0.0, right=0.0)

    # Kernel: F(tau) = c * E(c * tau), defined on tau = x/c in [-L/c, L/c]
    tau = x_par_arr / c_cnst
    F_plus  = c_cnst * E_plus
    F_cross = c_cnst * E_cross

    # Build a uniform tau grid covering the FULL range (negative and positive)
    tau_grid = np.arange(tau.min(), tau.max() + dt, dt)
    F_plus_tau  = np.interp(tau_grid, tau, F_plus,  left=0.0, right=0.0)
    F_cross_tau = np.interp(tau_grid, tau, F_cross, left=0.0, right=0.0)

    # FFT convolution — use mode="full" and manually extract the correct window
    conv_plus  = fftconvolve(hplus,  F_plus_tau,  mode="full") * dt
    conv_cross = fftconvolve(hcross, F_cross_tau, mode="full") * dt

    # The kernel is centered at tau=0, so find the index offset
    # tau_grid[0] is the lag of the first kernel sample relative to tau=0
    kernel_offset = int(round(-tau_grid[0] / dt))  # samples from start to tau=0
    start = kernel_offset
    end   = start + len(ts)
    RHS = conv_plus[start:end] + conv_cross[start:end]

    # Interpolate result back onto the original t_data grid
    RHS_out = np.interp(t_data, ts, RHS)

    print(f"[INFO] {label} computed using FFT convolution.")
    return t_data, RHS_out
