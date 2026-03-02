import os, json
import numpy as np
from geometry import CylindricalCavity, SphericalCavity, RectangularCavity
from modes import CylindricalMode, SphericalMode, RectangularMode
from scipy import interpolate

def build_mode_from_config(cfg):
    args = cfg["args"]
    cavity_type = args["geometry"]

    if cavity_type == "rectangular":
        cavity = RectangularCavity(a=args["a"], b=args["b"], c=args["c"])
        mode_class = RectangularMode
    elif cavity_type == "cylindrical":
        cavity = CylindricalCavity(R=args["R"], L=args["L"])
        mode_class = CylindricalMode
    elif cavity_type ==  "spherical":
        cavity = SphericalCavity(R=args["R"])
        mode_class = SphericalMode

    indices = tuple(int(x) for x in args["mode_ind"].split(","))

    if args["mode_par"] is not None:
        mode_name = args["mode_fam"] + args["mode_par"]
    else:
        mode_name = args["mode_fam"]
    
    mode = mode_class(indices=indices, mode_name=mode_name, cavity=cavity)
    omega = mode.omega()

    return omega

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

def load_run_config(run_dir):
    """Load run_config.json from the specified directory."""
    cfg_path = os.path.join(run_dir, "run_config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"run_config.json not found in {run_dir}")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    return cfg

