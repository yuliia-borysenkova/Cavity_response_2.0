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


# ---------------------------------------------------------------------------
# Onset detection
# ---------------------------------------------------------------------------

def estimate_onset_index(RHS, threshold_ratio=1e-3):
    """
    Find the sample index where the RHS signal first becomes significant.

    Uses a dual-criterion approach:
      1. Amplitude: |RHS| > threshold_ratio * max(|RHS|)
      2. Sustained: the amplitude stays above threshold for at least
         `min_sustain` consecutive samples, so isolated numerical spikes
         near t=0 don't fool the detector.

    Returns 0 if no onset is found (signal may already be active at t=0).
    """
    abs_rhs = np.abs(RHS)
    peak = abs_rhs.max()
    if peak == 0.0:
        return 0

    threshold = threshold_ratio * peak
    above = abs_rhs > threshold

    # Require the signal to stay above threshold for at least min_sustain
    # consecutive samples. Scale with array length so it's resolution-agnostic.
    min_sustain = max(3, len(RHS) // 500)

    for i in range(len(above) - min_sustain):
        if np.all(above[i : i + min_sustain]):
            return i

    # Fallback: first crossing
    crossings = np.where(above)[0]
    return int(crossings[0]) if len(crossings) else 0


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------

def cosine_onset_window(n, i0, width):
    """
    Build a window that is:
      0          for indices < i0
      raised-cosine ramp  for i0 <= index < i0+width
      1          for index >= i0+width

    The raised-cosine (Hann) ramp guarantees C1 continuity at both edges,
    which is critical for stiff ODE solvers.
    """
    if width <= 0:
        # No ramp: hard step at i0
        w = np.zeros(n)
        w[i0:] = 1.0
        return w

    w = np.ones(n)
    # Zero region before onset
    if i0 > 0:
        w[:i0] = 0.0
    # Smooth ramp
    i1 = min(i0 + width, n)
    ramp_len = i1 - i0
    if ramp_len > 0:
        phase = np.linspace(0.0, np.pi, ramp_len, endpoint=False)
        w[i0:i1] = 0.5 * (1.0 - np.cos(phase))
    return w


# ---------------------------------------------------------------------------
# Width heuristic
# ---------------------------------------------------------------------------

def _estimate_ramp_width(ts, RHS, i0, omega_cavity=None):
    """
    Choose a ramp width (in samples) that is:
      - At least one full oscillation period of the cavity (if omega_cavity
        is known), so the ramp is smooth relative to the resonant mode.
      - At least 1 % of the signal length for robustness on short signals.
      - No more than 20 % of the signal length so we don't swamp real physics.
      - Never wider than the gap between t[0] and the onset (i0 samples),
        because the ramp must finish before signal onset.

    Parameters
    ----------
    ts : array
        Uniform time grid.
    RHS : array
        Right-hand side signal (used only for length).
    i0 : int
        Detected onset index.
    omega_cavity : float or None
        Angular frequency of the cavity mode [rad/s]. If given, the ramp
        spans at least one full period.

    Returns
    -------
    int
        Ramp width in samples.
    """
    n = len(ts)
    dt = ts[1] - ts[0]

    # Floor: 1 % of signal length
    w_min = max(10, n // 100)

    # Ceiling: 20 % of signal length
    w_max = n // 5

    # Physics-based: cover at least N_periods full cavity periods
    N_periods = 3
    if omega_cavity is not None and omega_cavity > 0:
        T_cav = 2.0 * np.pi / omega_cavity
        w_phys = int(np.ceil(N_periods * T_cav / dt))
        w_phys = max(w_min, min(w_phys, w_max))
    else:
        # Fall back: 5 % of signal length
        w_phys = max(w_min, n // 20)

    # Hard constraint: ramp must end at or before onset
    w_max_i0 = max(1, i0)  # if i0==0 onset is at start; width=1 is trivial

    width = min(w_phys, w_max_i0, w_max)
    return max(1, width)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_onset_smoothing(
    ts,
    RHS,
    i0=None,
    width=None,
    threshold_ratio=1e-3,
    omega_cavity=None,
    verbose=True,
):
    """
    Apply a smooth onset window to RHS so that the driven ODE starts from
    rest without a discontinuous kick.

    The function:
      1. Detects where the signal truly "starts" (onset index i0).
      2. Prepends a raised-cosine ramp of `width` samples that grows from 0
         to 1, reaching 1 exactly at the onset.
      3. Returns the windowed signal and a callable interpolant.

    Parameters
    ----------
    ts : ndarray
        Uniform time array (length N).
    RHS : ndarray
        Driving term array (length N).
    i0 : int or None
        Override detected onset index. Auto-detected if None.
    width : int or None
        Override ramp width in samples. Auto-estimated if None.
    threshold_ratio : float
        Fraction of peak amplitude used for onset detection.
    omega_cavity : float or None
        Cavity angular frequency [rad/s]. Used to set a physics-informed
        minimum ramp width (>= 1 cavity period).
    verbose : bool
        Print diagnostic information.

    Returns
    -------
    ts : ndarray
        Same time array (unchanged).
    new_RHS : ndarray
        Windowed RHS array.
    new_RHS_fn : callable
        Linear interpolant of the windowed RHS, with fill_value=0 outside.
    """
    RHS = np.asarray(RHS, dtype=float)
    n = len(ts)

    # --- Step 1: detect onset ---
    if i0 is None:
        i0 = estimate_onset_index(RHS, threshold_ratio=threshold_ratio)

    if i0 == 0:
        # Signal starts immediately — we cannot zero-pad before it.
        # In this case the best we can do is ramp from the first sample.
        # Warn user: ideally extend the time grid backwards.
        if verbose:
            print(
                "[WARN] onset_smoothing: onset detected at i0=0 — signal starts "
                "at t[0]. The ramp will be applied from the beginning, but the "
                "ODE initial kick cannot be fully suppressed. Consider extending "
                "the time grid to include pre-signal silence (extend_rhs factor > 1)."
            )

    # --- Step 2: choose ramp width ---
    if width is None:
        width = _estimate_ramp_width(ts, RHS, i0, omega_cavity=omega_cavity)

    # --- Step 3: build window and apply ---
    window = cosine_onset_window(n, i0=i0, width=width)
    new_RHS = RHS * window

    if verbose:
        dt = ts[1] - ts[0]
        t_onset = ts[i0] if i0 < n else ts[-1]
        t_ramp_end = ts[min(i0 + width - 1, n - 1)]
        print(
            f"[INFO] onset_smoothing: i0={i0}, width={width} samples, "
            f"t_onset={t_onset*1e9:.3f} ns, "
            f"ramp ends at {t_ramp_end*1e9:.3f} ns"
        )

    # --- Step 4: build interpolant ---
    new_RHS_fn = interpolate.interp1d(
        ts, new_RHS, kind="linear",
        bounds_error=False, fill_value=0.0
    )

    return ts, new_RHS, new_RHS_fn


def compute_b(c, cD, pre_RHS, Q, omega):
    b = 1/omega * (cD + c_cnst * pre_RHS) + c/Q
    return b

def compute_U(c, b):
    U = 1/2 * epsilon_0 * (c**2 + b**2)
    return U
    
def clip_at_zero_crossing(t, c):
    crossings = np.where(np.diff(np.sign(c)) != 0)[0]
    if len(crossings) < 2:
        raise ValueError("Less than two zero crossings found")
    i = crossings[1] + 1
    return t[i:], c[i:]

def taper_signal(RHS, fraction=0.05):
    """
    Apply a cosine taper to the tail of the signal.

    `fraction` (default 5 %) sets the taper length as a fraction of the
    total signal length, making it resolution-agnostic unlike the old
    hard-coded n_taper=100.
    """
    RHS = np.asarray(RHS)
    N = len(RHS)
    n_taper = max(2, int(fraction * N))

    if n_taper >= N:
        raise ValueError("n_taper must be smaller than signal length")

    mask = np.ones(N)
    x = np.linspace(0, np.pi / 2, n_taper)
    mask[-n_taper:] = np.cos(x) ** 2
    return RHS * mask

def update_config_with_Q(config_file, args):
    """
    Load an existing combined config and add Q to cavity_info.derived.

    Parameters
    ----------
    config_file : path to the combined JSON config (created by build_config_file)
    args        : namespace / dict that must contain at least `Q`
    """

    # ── 1. Resolve Q ──────────────────────────────────────────────────────────
    Q = args.Q if hasattr(args, "Q") else args["Q"]

    # ── 2. Load, patch, save ──────────────────────────────────────────────────
    with open(config_file) as f:
        combined = json.load(f)

    combined["cavity_info"]["derived"]["Q"] = Q

    with open(config_file, "w") as f:
        json.dump(combined, f, indent=4)