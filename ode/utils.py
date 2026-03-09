import os, json, math
import numpy as np
from geometry import CylindricalCavity, SphericalCavity, RectangularCavity
from modes import CylindricalMode, SphericalMode, RectangularMode
from scipy import interpolate
from scipy.integrate import cumulative_simpson


def load_rhs(rhs_path):
    
    if not os.path.isfile(rhs_path):
        raise FileNotFoundError(f"[ERROR] {rhs_path} file not found in {save_dir}")
        
    data = np.load(rhs_path)
    ts = data["ts"]
    RHS = np.real(data["RHS"])

    RHS_fn = interpolate.interp1d(
        ts, RHS, kind="linear",
        bounds_error=False, fill_value=0.0
    )

    return ts, RHS, RHS_fn

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

def start_at_zero_crossing(time, RHS):

    # Find indices where sign changes
    crossings = []
    #crossings = np.where(np.sign(RHS[:-1]) != np.sign(RHS[1:]))[0]

    if len(crossings) == 0:
        return time, RHS  # no crossing found

    start_idx = crossings[1] + 1
    return time [start_idx:], RHS[start_idx:]

def compute_b(time, c, omega):

    if time.ndim != 1 or c.ndim != 1:
        raise ValueError("time and c must be 1D arrays")

    if len(time) != len(c):
        raise ValueError("time and c must have the same length")

    if len(time) < 2:
        raise ValueError("time array must contain at least two points")

    # Ensure uniform time spacing
    dt = time[1] - time[0]
    if not np.allclose(np.diff(time), dt):
        raise ValueError("time array must be uniformly spaced")

    # Cumulative trapezoidal integration
    integral = np.zeros_like(c, dtype=float)
    # integral[1:] = np.cumsum(0.5 * (c[1:] + c[:-1]) * dt)
    integral = cumulative_simpson(c, x=time, initial=0)

    b = -omega * integral

    # # ---- Find first local extrema ----
    # db = np.diff(b)
    # sign = np.sign(db)
    # sign_change = np.diff(sign)

    # # local maxima where sign goes + -> -
    # maxima = np.where(sign_change < 0)[0] + 1

    # # local minima where sign goes - -> +
    # minima = np.where(sign_change > 0)[0] + 1

    # if len(maxima) == 0 or len(minima) == 0:
    #     raise ValueError("Could not detect oscillation extrema")

    # first_max = maxima[1]
    # first_min = minima[minima > first_max]

    # if len(first_min) == 0:
    #     # If minimum comes first instead
    #     first_min = minima[1]
    #     first_max = maxima[maxima > first_min][0]
    # else:
    #     first_min = first_min[0]

    # shift = 0.5 * (b[first_max] + b[first_min])

    # print(shift, b.mean())

    return b - b.mean()

