import os, json, math
import numpy as np
from geometry import CylindricalCavity, SphericalCavity, RectangularCavity
from modes import CylindricalMode, SphericalMode, RectangularMode
from scipy import interpolate
from scipy.constants import c as c_cnst
from scipy.constants import epsilon_0
from scipy.signal.windows import tukey

#Files loading/saving utilities

def load_rhs(rhs_path):
    
    if not os.path.isfile(rhs_path):
        raise FileNotFoundError(f"[ERROR] {rhs_path} file not found")
        
    data = np.load(rhs_path)
    ts = data["ts"]
    RHS = np.real(data["RHS"])
    pre_RHS = np.real(data["pre_RHS"])

    return ts, RHS, pre_RHS

def save_amplitude(save_dir, result):
    path = os.path.join(save_dir, "amplitude_package.npz")
    np.savez(path, **result)
    print("[INFO] Saving ODE solution to ", path)
    return path

def load_from_config(run_dir):
    cfg_path = os.path.join(run_dir, "run_config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"run_config.json not found in {run_dir}")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    omega = cfg['omega']
    norm = cfg['norm']
    return omega, norm

def update_config_with_Q(config_file, args):

    Q = args.Q if hasattr(args, "Q") else args["Q"]

    with open(config_file) as f:
        combined = json.load(f)

    combined["cavity_info"]["parameters"]["Q"] = Q

    with open(config_file, "w") as f:
        json.dump(combined, f, indent=4)


#Onset smoothing: apply a smooth ramp to the beginning of the RHS signal to mitigate numerical artifacts from sharp onsets. 

def cosine_onset_window(n, i0, width):
    w = np.ones(n)
    w[:i0] = 0.0
    i1 = min(i0 + width, n)
    if i1 > i0:
        x = np.linspace(0, np.pi, i1 - i0)
        w[i0:i1] = 0.5 * (1 - np.cos(x))
    return w

def find_zero_crossings(RHS, n):
    """
    Return indices of the first n zero crossings of RHS.
    A crossing occurs where the signal changes sign.
    """
    signs = np.sign(RHS)
    # ignore leading zeros/noise
    crossings = np.where(np.diff(signs) != 0)[0]
    return crossings[:n]

def apply_onset_smoothing(ts, RHS, n_periods=5):
    """
    Smooth the onset over the first n_periods oscillations,
    measured by zero crossings.
    
    n_periods=5 means the ramp spans ~5 half-cycles from signal start.
    """
    crossings = find_zero_crossings(RHS, n=n_periods + 1)

    if len(crossings) < 2:
        # fallback: no clear oscillation found, no smoothing
        print("Warning: not enough zero crossings found, skipping smoothing.")
        return ts, RHS, interpolate.interp1d(ts, RHS, kind="linear",
                   bounds_error=False, fill_value=(RHS[0], RHS[-1]))

    i0 = crossings[0]   # where oscillation begins
    i1 = crossings[-1]  # end of the ramp region
    width = i1 - i0

    window = cosine_onset_window(len(ts), i0=i0, width=width)
    new_RHS = RHS * window

    new_RHS_fn = interpolate.interp1d(ts, new_RHS, kind="linear",
                     bounds_error=False, fill_value=(RHS[0], RHS[-1]))

    return ts, new_RHS, new_RHS_fn

#Signal processing utilities

def extend_rhs(time, RHS, factor):
    if factor <= 0:
        raise ValueError("factor must be > 0")

    if len(time) != len(RHS):
        raise ValueError("time and RHS must have the same length")

    dt = abs(time[-2] - time[-1])
    n_original = len(time)

    # Ensure integer length regardless of fractional factor
    n_new = int(math.ceil(n_original * factor))

    new_time = time[0] + dt * np.arange(n_new)
    new_RHS = np.zeros(n_new, dtype=RHS.dtype)
    new_RHS[:n_original] = RHS

    new_RHS_fn = interpolate.interp1d(
        new_time, new_RHS, kind="linear",
        bounds_error=False, fill_value=0.0
    )

    return new_time, new_RHS, new_RHS_fn

# ODE solving utilities

def analytical_free_decay(result, ts_ext, omega, Q):
    """
    Extend ODE solution analytically over the zero-forcing tail.
    Returns result dict with c, cD, t stitched over full ts_ext.
    """
    ts  = result['ts']
    t0  = ts[-1]
    c0  = result['c'][-1]
    cD0 = result['cD'][-1]

    alpha   = omega / (2.0 * Q)
    omega_d = omega * np.sqrt(1.0 - 1.0 / (4.0 * Q**2))

    A_n = c0
    B_n = (cD0 + alpha * c0) / omega_d

    tau = ts_ext[len(ts):] - t0

    c_tail  = np.exp(-alpha * tau) * (
        A_n * np.cos(omega_d * tau) + B_n * np.sin(omega_d * tau)
    )
    cD_tail = np.exp(-alpha * tau) * (
        (-alpha * A_n + omega_d * B_n) * np.cos(omega_d * tau) +
        (-alpha * B_n - omega_d * A_n) * np.sin(omega_d * tau)
    )

    return {
        **result,
        "ts":       ts_ext,
        "c":       np.concatenate([result['c'],  c_tail]),
        "cD":      np.concatenate([result['cD'], cD_tail]),
        "A_n":     A_n,       # <-- exposed for Fourier
        "B_n":     B_n,
        "alpha":   alpha,
        "omega_d": omega_d,
        "t0":      t0,
    }

#Furier transform utilities
def compute_full_fourier(result, ts_ext, n_driven, tukey_alpha=0.1):

    dt    = ts_ext[1] - ts_ext[0]
    n     = len(ts_ext)
    freqs = np.fft.fftfreq(n, d=dt)
    omegas = freqs * 2 * np.pi

    # driven part: zero-padded FFT
    # Tukey window applied only to driven segment, right edge only
    c_padded = np.zeros(n, dtype=result['c'].dtype)
    window   = tukey(n_driven, alpha=tukey_alpha)
    window[:n_driven//2] = 1.0                      # flatten left half — signal starts smoothly
    c_padded[:n_driven]  = result['c'][:n_driven] * window
    c_hat_num            = np.fft.fft(c_padded) * dt * np.exp(-1j*omegas*ts_ext[0])

    # free-decay tail: analytical (no windowing needed)
    s         = result['alpha'] + 1j * omegas
    c_hat_ana = np.exp(-1j * omegas * result['t0']) * \
                (result['A_n'] * s + result['B_n'] * result['omega_d']) / \
                (s**2 + result['omega_d']**2)

    return freqs, c_hat_num, c_hat_ana, c_hat_num + c_hat_ana

# ODE solution post-processing utilities

def compute_b(c, cD, pre_RHS, Q, omega):
    b = 1/omega * (cD + c_cnst * pre_RHS) + c/Q
    return b

def compute_U(c, b):
    U = 1/2 * epsilon_0 * (c**2 + b**2)
    return U