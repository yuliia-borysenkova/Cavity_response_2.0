import json
from pathlib import Path
import numpy as np
from scipy.constants import G, c
from astropy import constants as const

def gw_frequency_PBH(t, m_absolute, q, t_coal=0.0):
    """Calculate gravitational wave frequency for PBH merger."""
    t = np.asarray(t, dtype=float)
    tau = t_coal - t
    
    # Only compute for tau > 0
    valid = tau > 0
    if not np.any(valid):
        return np.full_like(t, np.nan, dtype=float)
    
    M_chirp = (q / (1 + q) ** 2) ** (3 / 5) * m_absolute * const.M_sun.value
    freq = np.full_like(t, np.nan, dtype=float)
    freq[valid] = (1 / np.pi) * (5 / 256 / tau[valid]) ** (3 / 8) * (G * M_chirp / c ** 3) ** (-5 / 8)
    return freq

def load_gw_config(data_dir,data_file_name):
    """Load gravitational wave configuration from JSON."""
    with open(f"{data_dir}/{data_file_name}_config.json", "r", encoding="utf-8") as f:
        return json.load(f)

def load_run_dir(results_dir, cavity_type, mode_name, mode_ind, theta, phi):
    """Construct run directory path."""
    return Path(results_dir) / f"{cavity_type}_{mode_name}_{mode_ind}_theta={theta}_phi={phi}"

def load_cavity_frequency_from_run_config(results_dir, cavity_type, mode_name, mode_ind, theta, phi):
    """Load cavity frequency from run configuration."""
    run_dir = load_run_dir(results_dir, cavity_type, mode_name, mode_ind, theta, phi)
    with open(run_dir / "run_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg["omega"] / (2 * np.pi)  # Convert rad/s to Hz

def find_chirp_match_time(ts, f_cavity, data_dir, data_file_name, t_coal=0.0, rtol=1e-2):
    """Find time when GW frequency matches cavity frequency."""
    cfg = load_gw_config(data_dir, data_file_name)
    f_gw = gw_frequency_PBH(ts, m_absolute=cfg["m_absolute"], q=cfg["q"], t_coal=t_coal)
    
    valid = np.isfinite(f_gw)
    if not np.any(valid):
        return None, None
    
    idx = np.where(valid)[0][np.argmin(np.abs(f_gw[valid] - f_cavity))]
    
    return (ts[idx], f_gw[idx]) if np.isclose(f_gw[idx], f_cavity, rtol=rtol) else (None, None)