#!/usr/bin/env python
import os, json, socket, argparse
from math import radians
from datetime import datetime, timezone
from geometry import CylindricalCavity, SphericalCavity, RectangularCavity
from modes import RectangularMode, CylindricalMode, SphericalMode
from rhs.slice_integration import SliceIntegration
import numpy as np

# ---------------- Argument parsing ----------------
def parse_args():
    parser = argparse.ArgumentParser(description="Cavity mode simulation CLI")
    
    # Geometry selection
    parser.add_argument("--geometry", choices=["rectangular", "cylindrical", "spherical"], default="cylindrical", help="Cavity geometry type")
    
    # Mode selection
    parser.add_argument("--mode-fam", choices=["TE", "TM"], default="TM", help="Mode family (TE or TM)")
    parser.add_argument("--mode-par", choices=["a", "b", None], default="b", help="Mode parity (even or odd)")
    parser.add_argument("--mode-ind", default="0,1,0", help="Mode indices as comma-separated values")
    
    # Simulation parameters
    parser.add_argument("--Bz", type=float, default=14.0, help="Magnetic field strength in Tesla")
    
    parser.add_argument("--theta", type=float, default=45.0, help="Polar angle of gravitational wave approach in degrees")
    parser.add_argument("--phi", type=float, default=0.0, help="Azimuthal angle of gravitational wave approach in degrees")
    
    parser.add_argument("--Ns", type=int, default=100, help="Number of samples for integration")
    parser.add_argument("--nproc", type=int, default=1, help="Number of processors for parallel execution")
    parser.add_argument("--method", choices=["vegas", "nquad"], default="nquad", help="Integration method")
    
    # Rectangular cavity
    parser.add_argument("--a", type=float, default=0.1, help="Rectangular cavity x-dimension length (meters)")
    parser.add_argument("--b", type=float, default=0.1, help="Rectangular cavity y-dimension length (meters)")
    parser.add_argument("--c", type=float, default=0.1, help="Rectangular cavity z-dimension length (meters)")
    
    # Cylindrical/Spherical cavity
    parser.add_argument("--R", type=float, default=0.05, help="Cavity radius (meters)")
    parser.add_argument("--L", type=float, default=0.05, help="Cylindrical cavity length (meters)")

    # Results directory
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save simulation results")
    
    return parser.parse_args()


# ---------------- Main CLI ----------------
def main():
    args = parse_args()

    theta_rad = np.radians(args.theta)
    phi_rad = np.radians(args.phi)
    
    if args.mode_par is not None:
        mode_name = args.mode_fam + args.mode_par
    else:
        mode_name = args.mode_fam

    # --- Prepare results directory ---
    dir_name = f"{args.geometry}_{mode_name}_{args.mode_ind}_theta={args.theta}_phi={args.phi}_Ns={args.Ns}"
    save_dir = os.path.join(args.results_dir, dir_name)
    os.makedirs(save_dir, exist_ok=True)

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

        # Save run config
    run_info = {
        "args": vars(args),
        "omega": mode.omega(),
        "norm": mode.norm,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname()
    }
    with open(os.path.join(save_dir, "run_config.json"), "w") as f:
        json.dump(run_info, f, indent=2)

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
