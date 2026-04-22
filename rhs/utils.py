# utils.py
import json
import os
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.constants import c as c_cnst

def compute_k_pol(theta, phi):

    k = -np.array([np.sin(theta)*np.cos(phi), 
                  np.sin(theta)*np.sin(phi), 
                  np.cos(theta)
                 ])

    e1 = np.array([-np.sin(phi), 
                   np.cos(phi), 
                   0.0
                 ])

    e2 = np.array([np.cos(theta)*np.cos(phi), 
                   np.cos(theta)*np.sin(phi), 
                   -np.sin(theta)
                 ])

    return k, e1, e2

def decompose_B(B, k, e1, e2):
    """
    Compute B_plus and B_cross polarizations perpendicular to k
    """
    Bperp = B - np.dot(B, k) * k
    B_plus  = np.dot(Bperp, e2) * e1 + np.dot(Bperp, e1) * e2
    B_cross = -np.dot(Bperp, e1) * e1 + np.dot(Bperp, e2) * e2
    
    return B_plus, B_cross

def make_jeff(B, cavity, hplus, hcross, k, e1, e2):

    B_plus, B_cross = decompose_B(B, k, e1, e2)
    center = cavity.center()

    def tau(Y, t):
        return t - np.vdot(k, cavity.native_to_cart(Y) - center) / c_cnst

    def jeff_from_B(B, h):
        def jeff(Y, t):
            return cavity.cart_vec_to_native(h(tau(Y, t)) * B, Y)
        return jeff

    jeff_plus  = jeff_from_B(B_plus,  hplus)
    jeff_cross = jeff_from_B(B_cross, hcross)

    def jeff_full(Y, t):
        return jeff_plus(Y, t) + jeff_cross(Y, t)

    return {
        "plus":  jeff_plus,
        "cross": jeff_cross,
        "mix":   jeff_full,
    }

def save_plot(save_dir, filename):
    """Save current matplotlib figure to file"""
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename}", dpi=200)
    plt.close()

def parse_vector(s):
    """Parse a comma-separated string into a normalized numpy vector"""
    vec = np.array([float(x) for x in s.split(",")])
    return vec / np.linalg.norm(vec)

def load_slice_integrals(data, results_dir, cavity_type, mode_name, mode_ind, theta, phi, Ns):
    
    data_name = os.path.splitext(os.path.basename(data))[0]

    dir1_name = f"{cavity_type}_{mode_name}_{mode_ind}_theta={theta}_phi={phi}_Ns={Ns}"
    dir2_name = f"DATA_{data_name}"
    
    run_dir = os.path.join(results_dir, dir1_name)
    os.makedirs(run_dir, exist_ok=True)
    
    save_dir = os.path.join(results_dir, dir1_name, dir2_name)
    os.makedirs(save_dir, exist_ok=True)

    filename_plus = os.path.join(run_dir, f"slice_integrals_plus.npz")
    file_plus = np.load(filename_plus)
    area_integral_plus, x_par_vals = file_plus["area"], file_plus["x_par"]

    filename_cross = os.path.join(run_dir, f"slice_integrals_cross.npz")
    file_cross = np.load(filename_cross)
    area_integral_cross, _ = file_cross["area"], file_cross["x_par"]

    slice_integrals = np.array([x_par_vals, area_integral_plus, area_integral_cross])

    return save_dir, slice_integrals

class CubicSplineInterp:
    def __init__(self, ts, y, left_val, right_val):
        self._spline = interpolate.interp1d(
            ts, y,
            kind="cubic", #options: "linear", "nearest", "slinear", "quadratic", "cubic"
            bounds_error=False,
            fill_value=(left_val, right_val)
        )
    def __call__(self, tau):
        return self._spline(tau)
    
def load_characteristic_length_from_run_config(results_dir, geometry, mode_name, mode_ind, theta, phi, Ns):
    dir_name = f"{geometry}_{mode_name}_{mode_ind}_theta={theta}_phi={phi}_Ns={Ns}"
    run_dir = os.path.join(results_dir, dir_name)
    config_path = os.path.join(run_dir, "run_config.json")
    print("here")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            run_info = json.load(f)
            L = 0
            if geometry == "rectangular":
                a = run_info["args"].get("a", None)
                b = run_info["args"].get("b", None)
                c = run_info["args"].get("c", None)
                if a is not None or b is not None or c is not None:
                    L = max(a or 0, b or 0, c or 0)
            elif geometry == "cylindrical":
                L = run_info["args"].get("L", None)
            elif geometry == "spherical":
                R = run_info["args"].get("R", None)
                if R is not None:
                    L = 2 * R  # Use diameter as characteristic length for spherical cavity
            return L   
    else:
        print(f"[WARNING] Run config file not found at {config_path}. Cannot load characteristic length.")
        return None
    
def build_config_file(gw_config_file, slice_integrals_config_file, args):

    #load data from both config files
    with open(gw_config_file) as f:
        gw_data = json.load(f)
    with open(slice_integrals_config_file) as f:
        slice_data = json.load(f)

    #build combinded structure




    return;