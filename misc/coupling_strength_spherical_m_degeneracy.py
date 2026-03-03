"""
Post-process cavity-GW coupling strength for spherical cavities with degenerate m modes (e.g. TM_m11 for m = -1, 0, 1).
Combines multiple coupling strength files via quadrature sum and computes statistics.
"""

import argparse
import glob
import os
import numpy as np

from coupling.utils import mean_calc
    
def parse_args():
    parser = argparse.ArgumentParser(description="Compute cavity-GW coupling strength η(θ,φ) for various geometries and modes.")
    # Parameters to select files
    parser.add_argument("--mode-fam", choices=["TM", "TE"], default="TM", help="Mode family")
    parser.add_argument("--mode-ind", default="1,1", help=("Mode indices, provide as comma-separated values for sphere: [n,p] n>=1,p>=1"))
    parser.add_argument("--R", default=None, type=float, help="Cavity radius [m]")
    parser.add_argument("--pol", choices=["plus", "cross"], default="cross", help="GW polarization: 'plus' or 'cross'")
    parser.add_argument("--N-theta", type=int, default=10, help="Number of theta angles to sample")
    parser.add_argument("--N-phi", type=int, default=0, help="Number of phi angles to sample")
    # Results directory
    parser.add_argument("--save-dir", type=str, default="results/coupling")
    
    return parser.parse_args()


def main():
    args = parse_args()

    # parse n,p from "n,p"
    np_parts = [x.strip() for x in args.mode_ind.split(",") if x.strip()]
    if len(np_parts) != 2:
        raise SystemExit("For spherical postprocess, --mode-ind must be 'n,p' (e.g. 1,1).")
    n_arg = int(np_parts[0])
    p_arg = int(np_parts[1])

    files_dir = os.path.join(args.save_dir, f"spherical_{args.mode_fam}_{args.mode_ind}")

    files = sorted(glob.glob(os.path.join(files_dir, "*.npz")))
    selected_files = [f for f in files 
                      if "coupling_spherical" in f
                      and f"{args.mode_fam}" in f
                      and f"R_{args.R:.2f}" in f 
                      and f"pol_{args.pol}" in f 
                      and f"Ntheta_{args.N_theta}" in f 
                      and f"Nphi_{args.N_phi}" in f]
    
    if len(selected_files) != 2*n_arg + 1:
        raise SystemExit(f"Expected {2*n_arg + 1} files for spherical mode with n={n_arg}, but found {len(selected_files)}. Check the files in {files_dir} and ensure they match the expected naming convention.")

    print(f"[INFO] Used {len(selected_files)} files for post-processing spherical degeneracy: {selected_files}")

    etas = []
    phi = []
    theta = []
    omega = []

    #MUST BE CHANGED IF THE SHAPE of grid IS DIFFERENT
    for path in selected_files:
        d = np.load(path, allow_pickle=True)

        theta_f = d["theta"]          
        phi_f   = d["phi"]    
        omega_f = d["omega"]    
        eta = d["eta"]        

        theta.append(theta_f)
        phi.append(phi_f)
        omega.append(omega_f)
        etas.append(eta)

    etas_arr = np.stack(etas, axis=0)      # shape: (Nfiles, ...gridshape...)
    C = np.sum(etas_arr**2, axis=0)
    eta_total = np.sqrt(np.sum(etas_arr**2, axis=0))
    
    mean_eta = mean_calc(eta_total, theta[0])
    max_eta = np.max(eta_total)
    
    mean_C = mean_calc(C, theta[0])
    max_C = np.max(C)


    #summarise
    print("[INFO] Results for coupling strength:")
    print(f"⟨η(θ, φ)⟩ = {mean_eta:.4f}, ηₘₐₓ = {max_eta:.4f}")
    print(f"⟨C(θ, φ)⟩=⟨η²(θ, φ)⟩ = {mean_C:.4f}, Cₘₐₓ = {max_C:.4f}")

    #save summary in a .npz in the same directory
    save_file = f"coupling_spherical_[degenerate_m]_pol_{args.pol}_{args.mode_fam}_{args.mode_ind}_R_{args.R:.2f}cm_Ntheta_{args.N_theta}_Nphi_{args.N_phi}.npz"
    np.savez(os.path.join(files_dir, save_file), 
             cavity_type="spherical",
             mode_family=args.mode_fam, 
             mode_indices=args.mode_ind, 
             L=(np.nan), 
             R=(args.R),
             a=(np.nan),
             b=(np.nan),
             c=(np.nan),
             omega = omega[0], 
             pol=args.pol, 
             N_theta=args.N_theta,
             N_phi=args.N_phi,
             theta=theta[0], 
             phi=phi[0], 
             eta=eta_total, 
             C=C,
             mean_eta=mean_eta,
             max_eta=max_eta,
             mean_C=mean_C,
             max_C=max_C,
             source_files=np.array(selected_files, dtype=object))
    
    print(f"[INFO] Summary results saved to {files_dir}/{save_file}.npz")

if __name__ == "__main__":    
    main()