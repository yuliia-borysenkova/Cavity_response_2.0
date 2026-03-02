import argparse
import numpy as np
import os

from geometry import CylindricalCavity, SphericalCavity, RectangularCavity
from modes import CylindricalMode, SphericalMode, RectangularMode

from coupling.coupling import CouplingStrength
from coupling.utils import mean_calc


# ---------------- Argument parsing ----------------
def parse_args():
    parser = argparse.ArgumentParser(description="Compute cavity-GW coupling strength η(θ,φ) for various geometries and modes.")
    
    # Geometry selection
    parser.add_argument("--geometry", choices=["rectangular", "cylindrical", "spherical"], default="cylindrical")
    
    # Mode selection
    parser.add_argument("--mode-fam", choices=["TM", "TE"], default="TM", help="Mode family")
    parser.add_argument("--mode-ind", default="0,1,0", help=("Mode indices, provide as comma-separated values 'a,b,c'.\n"
                                                                        "Cylinder: [n,p,q] with n>=0,p>=1,q>=0.\n"
                                                                        "Sphere: [m,n,p] with m ∈ [-n, ..., n], n>=1,p>=1.\n"
                                                                        "Rectangle: [m, n, p]"))
    
    # Cylindrical/Spherical cavity
    parser.add_argument("--R", default=None, type=float, help="Cavity radius [m]")
    parser.add_argument("--L", default=None, type=float, help="Cavity length [m]")

    # Rectangular cavity
    parser.add_argument("--a", default=None, type=float, help="Cavity length [m]")
    parser.add_argument("--b", default=None, type=float, help="Cavity width [m]")
    parser.add_argument("--c", default=None, type=float, help="Cavity height [m]")
    
    # Simulation parameters
    parser.add_argument("--pol", choices=["plus", "cross"], default="cross", help="GW polarization: 'plus' or 'cross'")
    parser.add_argument("--N-theta", type=int, default=10, help="Number of theta angles to sample")
    parser.add_argument("--N-phi", type=int, default=0, help="Number of phi angles to sample")
    parser.add_argument("--nproc", type=int, default=1)
    
    # Results directory
    parser.add_argument("--save-dir", type=str, default="results/coupling")
    
    return parser.parse_args()

# ---------------- Main CLI ----------------
def main():
    args = parse_args()

    mode_ind = [int(x) for x in args.mode_ind.split(",")]

    # --- Create cavity and mode ---
    if args.geometry == "cylindrical":
        cavity = CylindricalCavity(R=args.R, L=args.L)
        mode_class = CylindricalMode
        mode_name_arr = [args.mode_fam + "a", args.mode_fam + "b"]
        
    elif args.geometry ==  "spherical":
        cavity = SphericalCavity(R=args.R)
        mode_class = SphericalMode
        mode_name_arr = [args.mode_fam + "a", args.mode_fam + "b"]
        
    elif args.geometry ==  "rectangular":
        cavity = RectangularCavity(a=args.a, b=args.b, c=args.c)
        mode_class = RectangularMode
        mode_name_arr = [args.mode_fam]

    result = []
    theta_vals = np.linspace(0.0, np.pi, args.N_theta)
    phi_vals = np.linspace(0.0, 2.0 * np.pi, args.N_phi) if args.N_phi > 0 else np.zeros(1)

    for mode_name in mode_name_arr:
        mode = mode_class(indices=mode_ind, mode_name=mode_name, cavity=cavity)
        mode.normalize()

        res = CouplingStrength(cavity=cavity, mode=mode, theta_vals=theta_vals, phi_vals=phi_vals, pol=args.pol, nproc=args.nproc)
        result.append(res.compute_coupling(t=0.0))

    eta_a = result[0]
    eta_b = result[1] if len(result) > 1 else 0.0
    eta = np.sqrt(np.abs(eta_a)**2 + np.abs(eta_b)**2)
    
    C = eta**2
    
    mean_eta = mean_calc(eta, theta_vals)
    max_eta = np.max(eta)
    
    mean_C = mean_calc(C, theta_vals)
    max_C = np.max(C)

    print("[INFO] Results for coupling strength:")
    print(f"⟨η(θ, φ)⟩ = {mean_eta:.4f}, ηₘₐₓ = {max_eta:.4f}")
    print(f"⟨C(θ, φ)⟩=⟨η²(θ, φ)⟩ = {mean_C:.4f}, Cₘₐₓ = {max_C:.4f}")

    #Save data and all plots
    folder_ind_str = args.mode_ind
    # For spherical modes, use only n,p indices (skip m which adds degeneracy)
    if args.geometry == "spherical":
        indices = args.mode_ind.split(",")
        folder_ind_str = f"{indices[1]},{indices[2]}"

    dir_name = f"{args.geometry}_{args.mode_fam}_{folder_ind_str}"
    save_dir = os.path.join(args.save_dir, dir_name)
    os.makedirs(save_dir, exist_ok=True)

    # Build parameter string based on geometry
    if args.geometry == "cylindrical":
        params = f"L_{args.L:.2f}_R_{args.R:.2f}"
    elif args.geometry == "spherical":
        params = f"R_{args.R:.2f}"
    else:  # rectangular
        params = f"a_{args.a:.2f}_b_{args.b:.2f}_c_{args.c:.2f}"
    
    save_file = f"coupling_{args.geometry}_pol_{args.pol}_{args.mode_fam}_{args.mode_ind}_{params}_Ntheta_{args.N_theta}_Nphi_{args.N_phi}"

    np.savez(os.path.join(save_dir, save_file), cavity_type=args.geometry, mode_family=args.mode_fam, mode_indices=args.mode_ind, 
             L=(args.L if args.geometry == "cylindrical" else np.nan), 
             R=(args.R if args.geometry != "rectangular" else np.nan),
             a=(args.a if args.geometry == "rectangular" else np.nan),
             b=(args.b if args.geometry == "rectangular" else np.nan),
             c=(args.c if args.geometry == "rectangular" else np.nan),
             omega = mode.omega(), pol=args.pol, N_theta=args.N_theta, N_phi=args.N_phi,
             theta=theta_vals, phi=phi_vals, eta_a=eta_a, eta_b=eta_b, eta=eta, C=C)
    
    print(f"[INFO] Results saved to {save_dir}/{save_file}.npz")

if __name__ == "__main__":
    main()
