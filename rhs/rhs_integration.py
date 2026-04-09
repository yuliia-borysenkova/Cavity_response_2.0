import numpy as np
from scipy.constants import c as c_cnst
from scipy.signal import fftconvolve


def compute_rhs_time_series(t_data, x_par_arr, E_plus, E_cross, hplus_dd, hcross_dd, Nt, L_char, label="RHS"):

    # ---- retarded time bounds ----
    tshift = L_char / c_cnst
    tmin = t_data[0] - tshift
    tmax = t_data[-1] + tshift

    # ---- uniform time grid ----
    ts = np.linspace(tmin, tmax, Nt)
    dt = ts[1] - ts[0]
    
    # ---- interpolate waveform onto uniform grid ----
    hplus = np.interp(ts, t_data, hplus_dd, left=0.0, right=0.0)
    hcross = np.interp(ts, t_data, hcross_dd, left=0.0, right=0.0)

    # ---- convert spatial coordinate to retarded time ----
    tau = x_par_arr / c_cnst

    # IMPORTANT: E is defined in x-space, so we only multiply by c
    # to account for dx = c dτ
    F_plus = c_cnst * E_plus
    F_cross = c_cnst * E_cross

    # ---- kernel spacing in τ ----
    dtau = tau[1] - tau[0]

    # ---- build uniform τ grid matching the FFT time resolution ----
    tau_grid = np.arange(0, tau.max() + dt, dt)

    # ---- interpolate kernel onto τ grid ----
    F_plus_tau = np.interp(tau_grid, tau, F_plus, left=0.0, right=0.0)
    F_cross_tau = np.interp(tau_grid, tau, F_cross, left=0.0, right=0.0)

    # ---- FFT convolution ----
    RHS_plus = fftconvolve(hplus, F_plus_tau, mode="same") * dt
    RHS_cross = fftconvolve(hcross, F_cross_tau, mode="same") * dt

    RHS = RHS_plus + RHS_cross

    print(f"[INFO] {label} computed using FFT convolution.")

    return ts, RHS
