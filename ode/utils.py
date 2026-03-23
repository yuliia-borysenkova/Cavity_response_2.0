import os, json, math
import numpy as np
from geometry import CylindricalCavity, SphericalCavity, RectangularCavity
from modes import CylindricalMode, SphericalMode, RectangularMode
from scipy import interpolate
from scipy.constants import c as c_cnst
from scipy.constants import epsilon_0


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
    """Load run_config.json from the specified directory."""
    cfg_path = os.path.join(run_dir, "run_config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"run_config.json not found in {run_dir}")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    omega = cfg['omega']
    norm = cfg['norm']
    return omega, norm
    
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

def estimate_onset_index(RHS, ts, threshold_ratio=1e-3):
    """
    Estimate the onset index where RHS signal becomes significant.
    """
    # Method: Find where amplitude exceeds threshold
    max_val = np.max(np.abs(RHS))
    threshold = threshold_ratio * max_val
    i_amp = np.argmax(np.abs(RHS) > threshold)

    return i_amp

def estimate_width(ts, fraction=0.03):
    """
    Estimate the smoothing width as a fraction of the total time span.
    """
    n = len(ts)
    return int(fraction * n)

def cosine_onset_window(n, i0, width):
    w = np.ones(n)
    w[:i0] = 0.0
    i1 = min(i0 + width, n)
    if i1 > i0:
        x = np.linspace(0, np.pi, i1 - i0)
        w[i0:i1] = 0.5 * (1 - np.cos(x))
    return w

def apply_onset_smoothing(ts, RHS, i0=None, width=None, fraction=0.03, threshold_ratio=1e-3):
    
    if i0 is None:
        i0 = estimate_onset_index(RHS, ts, threshold_ratio=threshold_ratio)
    if width is None:
        width = estimate_width(ts, fraction=fraction)
    
    i0 = min(i0, len(ts) - 1)
    window = cosine_onset_window(len(ts), i0=i0, width=width)
    new_RHS = RHS * window

    new_RHS_fn = interpolate.interp1d(
        ts, new_RHS, kind="linear",
        bounds_error=False, fill_value=(RHS[0], RHS[-1]))

    return ts, new_RHS, new_RHS_fn

def compute_b(c, cD, pre_RHS, Q, omega):

    b = 1/omega * (cD + c_cnst * pre_RHS) + c/Q

    return b

def compute_U(c, b):

    U = 1/2 * epsilon_0 * (c**2 + b**2)

    return U
    
def clip_at_zero_crossing(t, c):
    # indices where sign changes
    crossings = np.where(np.diff(np.sign(c)) != 0)[0]

    if len(crossings) < 2:
        raise ValueError("Less than two zero crossings found")

    i = crossings[1] + 1  # second crossing

    return t[i:], c[i:]

def taper_signal(RHS, n_taper=100):
    RHS = np.asarray(RHS)
    N = len(RHS)

    if n_taper >= N:
        raise ValueError("n_taper must be smaller than signal length")

    mask = np.ones(N)

    # cosine taper from 1 → 0
    x = np.linspace(0, np.pi/2, n_taper)
    mask[-n_taper:] = np.cos(x)**2

    return RHS * mask


