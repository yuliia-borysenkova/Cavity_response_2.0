#!/usr/bin/env python
import os, json, socket, argparse
from math import radians
from datetime import datetime, timezone
from geometry import CylindricalCavity, SphericalCavity, RectangularCavity
from modes import RectangularMode, CylindricalMode, SphericalMode
from rhs.slice_integration import SliceIntegration
from rhs.utils import parse_vector
import numpy as np

# ---------------- Argument parsing ----------------
def parse_args():
    parser = argparse.ArgumentParser(description="Cavity mode simulation CLI")
    
    # Geometry selection
    parser.add_argument("--geometry", choices=["rectangular", "cylindrical", "spherical"], default="cylindrical")
    
    # Mode selection
    parser.add_argument("--mode-fam", choices=["TE", "TM"])
    parser.add_argument("--mode-par", choices=["a", "b", None])
    parser.add_argument("--mode-ind", default="0,1,0")
    
    # Simulation parameters
    parser.add_argument("--Bz", type=float, default=14.0)
    
    parser.add_argument("--theta", type=float, default=45.0, help="Polar angle of gravitational wave approach in degrees")
    parser.add_argument("--phi", type=float, default=0.0, help="Azimuthal angle of gravitational wave approach in degrees")
    
    parser.add_argument("--Ns", type=int, default=100)
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--method", choices=["vegas", "nquad"], default="nquad")
    
    # Rectangular cavity
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--b", type=float, default=0.1)
    parser.add_argument("--c", type=float, default=0.1)
    
    # Cylindrical/Spherical cavity
    parser.add_argument("--R", type=float, default=0.04)
    parser.add_argument("--L", type=float, default=0.24)

    # Results directory
    parser.add_argument("--results-dir", type=str, default="results")
    
    return parser.parse_args()


# ---------------- Main CLI ----------------
def main():
    args = parse_args()

    theta_rad = radians(args.theta)
    phi_rad = radians(args.phi)
    
    if args.mode_par is not None:
        mode_name = args.mode_fam + args.mode_par
    else:
        mode_name = args.mode_fam

    # --- Prepare results directory ---
    dir_name = f"{args.geometry}_{mode_name}_{args.mode_ind}_theta={args.theta}_phi={args.phi}"
    save_dir = os.path.join(args.results_dir, dir_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save run config
    run_info = {
        "args": vars(args),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname()
    }
    with open(os.path.join(save_dir, "run_config.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    # --- Parse mode indices and GW vector ---
    mode_ind = [int(x) for x in args.mode_ind.split(",")]
    B = np.array([0, 0, args.Bz])

    # --- Create cavity and mode ---
    if args.geometry == "rectangular":
        cavity = RectangularCavity(a=args.a, b=args.b, c=args.c)
        mode_class = RectangularMode
    elif args.geometry == "cylindrical":
        cavity = CylindricalCavity(R=args.R, L=args.L)
        mode_class = CylindricalMode
    elif args.geometry ==  "spherical":
        cavity = SphericalCavity(R=args.R)
        mode_class = SphericalMode

    mode = mode_class(indices=mode_ind, mode_name=mode_name, cavity=cavity)
    mode.normalize()

    # --- Run slice simulation ---
    sim = SliceIntegration(
        cavity=cavity,
        mode=mode,
        theta=theta_rad, phi=phi_rad,
        B=B, Ns=args.Ns,
        nproc=args.nproc, method=args.method,
        save_dir=save_dir
    )
    sim.run()


if __name__ == "__main__":
    main()
