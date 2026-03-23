import time, os, socket, json
from datetime import datetime, timezone
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import integrate, interpolate, stats
from gw.utils import load_waveform
from rhs.utils import compute_k_pol, decompose_B
from rhs.num_rhs_integration import compute_num_rhs, extract_mode, plot_3d
from misc.resonant_frequency_matches import find_chirp_match_time
from plotting.theme import new_figure, save_figure
from geometry import CylindricalCavity, SphericalCavity, RectangularCavity

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing the .npy data file and its config")
    parser.add_argument("--mode-dir", type=str, default="modes/numerical", help="Directory containing the .csv mode data file")
    parser.add_argument("--results-dir", type=str, default="results", help="Path to results directory")
    parser.add_argument("--Nt", type=int, default=10000, help="Number of time samples")
    parser.add_argument("--Ns", type=int, default=100, help="Number of spatial samples") # Don't choose Ns which is too small
    
    parser.add_argument("--freq-match", action="store_true", help="Match GW frequency to cavity resonant frequency and indicate it on the plot") 
    parser.add_argument("--pre-RHS", action="store_true", help="Compute RHS before one time derivative. Helps with accuracy in computing b(t)")

    parser.add_argument("--mode", type=str, default="TM010_5.119GHz", help="Path to .csv mode data file")
    parser.add_argument("--f", type=float, default=5.119, help="Frequency of the mode in GHz")
    parser.add_argument("--Bz", type=float, default=14.0, help="Magnetic field strength in Tesla")

    parser.add_argument("--data", type=str, required=True, help="Path to .npy data file")
    parser.add_argument("--theta", type=float, default=45.0, help="Polar angle of gravitational wave approach in degrees")
    parser.add_argument("--phi", type=float, default=0.0, help="Azimuthal angle of gravitational wave approach in degrees")

    parser.add_argument("--search-ell", action="store_true", help="TBD") # Write descriptions
    parser.add_argument("--ell", type=float, default=0.0015, help="TBD")
    parser.add_argument("--Nell", type=int, default=20, help="TBD")
    
    return parser.parse_args()

def main():

    args = parse_args()
    start = time.time()

    data_path = os.path.join(args.data_dir, args.data + ".npy")
    mode_path = os.path.join(args.mode_dir, args.mode + ".csv")

       
    dir1_name = f"{args.mode}_theta={args.theta}_phi={args.phi}"
    dir2_name = f"DATA_{args.data}"
    save_dir = os.path.join(args.results_dir, dir1_name, dir2_name)
    os.makedirs(save_dir, exist_ok=True)

    theta_rad = np.radians(args.theta)
    phi_rad = np.radians(args.phi)
    omega = 2 * np.pi * args.f * 1e9
    
    # Decompose B
    B = np.array([0., 0., args.Bz]) # in T
    k, e1, e2 = compute_k_pol(theta_rad, phi_rad)
    B_plus, B_cross = decompose_B(B, k, e1, e2)
    
    # Load all lines, skip COMSOL comment lines
    coords, Efield, norm, V = extract_mode(mode_path)
    num = len(coords)
    plot_3d(coords, Efield, save_dir)
    
    xpar = np.einsum('ij,j->i', coords, k)
    t_data, hplus_dd, hcross_dd = load_waveform(data_path, derivative=2)

    xps = np.linspace(np.min(xpar), np.max(xpar), args.Ns) # in m
    ts = np.linspace(t_data[0], t_data[-1], args.Nt)

    hplusDD = lambda t: np.interp(t, t_data, hplus_dd, left=0.0, right=0.0)
    hcrossDD = lambda t: np.interp(t, t_data, hcross_dd, left=0.0, right=0.0)

    print("[INFO] Computing RHS...")
    if args.search_ell:
        ell_start = np.linalg.norm(coords[0].flatten() - coords[1].flatten()) # in m
        ell_arr = ell_start * np.logspace(-2, 2, args.Nell)
    
        overlaps = np.empty(len(ell_arr), dtype=object)
        errors = np.empty(len(ell_arr), dtype=object)
        
        for i, ell in enumerate(tqdm(ell_arr, desc="RHS(t): ")):
            overlaps[i], errors[i] = compute_num_rhs(xpar, xps, ts, num, V, Efield, B_plus, B_cross, hplusDD, hcrossDD, ell)
        
        overlaps = np.stack(overlaps) 
        errors = np.stack(errors)

        mean_errors = np.array([err[int(0.98 * len(err)):].mean() for err in errors]) # Average over the end where the values are big
        
        min_index = mean_errors.argmin() - 1 # Do one value before minimum to make sure statistical error dominates
        min_value = mean_errors[min_index]
        ell = ell_arr[min_index]
        
        print(f"[INFO] Minimum mean error is {min_value:.2f} at index {min_index} and l = {ell:2f} m")
        RHS = overlaps[min_index]
        error = errors[min_index]
    else:
        ell = args.ell
        RHS, error = compute_num_rhs(xpar, xps, ts, num, V, Efield, B_plus, B_cross, hplusDD, hcrossDD, ell)

    xpar_len = np.abs(xpar[-1] - xpar[0])
    if ell > xpar_len / args.Ns:
        print(f"[WARN] The errors might be underestimated, as slices are no longer independent. Consider choosing Ns < {args.Ns}")

    print("[INFO] RHS computed in {:.2f} s".format(time.time() - start))

    start = time.time()
    print("[INFO] Computing pre_RHS...")
    if args.pre_RHS:
        _, hplus_d, hcross_d = load_waveform(data_path, derivative=1)
        hplusD = lambda t: np.interp(t, t_data, hplus_d, left=0.0, right=0.0)
        hcrossD = lambda t: np.interp(t, t_data, hcross_d, left=0.0, right=0.0)
        
        pre_RHS, error = compute_num_rhs(xpar, xps, ts, num, V, Efield, B_plus, B_cross, hplusD, hcrossD, ell)

        print("[INFO] pre_RHS computed in {:.2f} s".format(time.time() - start))
    else:
        pre_RHS = []
        
    # Save RHS to save_dir
    file_name = os.path.join(save_dir, f"RHS_{args.mode}_{args.data}.npz")
    np.savez(file_name, ts=ts, RHS=RHS, pre_RHS=pre_RHS, mode=args.mode, data=args.data)
    print("[INFO] Saved RHS array file to ", file_name)

    # Save run config
    run_info = {
        "args": vars(args),
        "omega": omega,
        "norm": norm, # Change this?
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname()
    }
    with open(os.path.join(save_dir, "run_config.json"), "w") as f:
        json.dump(run_info, f, indent=2)
    
    # Plot RHS array as a function of time
    fig, ax = new_figure()
    ax.plot(ts * 1e9, RHS, label="RHS (slicing method)")
    
    # Add vertical line for resonant time if frequency matching is enabled
    f_cavity = omega/(2*np.pi)
    t_match, _ = find_chirp_match_time(ts=ts, f_cavity=omega, data_dir=args.data_dir, data_file_name=args.data)
    if t_match is not None and args.freq_match:
        ax.axvline(t_match * 1e9, linestyle="--", linewidth=1.5, color="darkred", label=r"$f_{\rm GW} = f_{\rm cav}$")
    
    ax.set_xlabel(r"$t\,[\mathrm{ns}]$")
    ax.set_ylabel(r"$\mathrm{RHS}(t)$")
    ax.set_title(rf"$\mathrm{{RHS}}(t)$ for {args.mode};"+f"\n waveform file: {args.data}" )
    ax.legend()
    save_figure(
        fig,
        os.path.join(save_dir,  f"RHS(t)_{args.mode}_{args.data}.png"),
    )

    if args.pre_RHS:
        # Plot RHS array as a function of time
        fig, ax = new_figure()
        ax.plot(ts * 1e9, pre_RHS, label="preRHS (slicing method)")
        
        # Add vertical line for resonant time if frequency matching is enabled
        f_cavity = omega/(2*np.pi)
        t_match, _ = find_chirp_match_time(ts=ts, f_cavity=omega, data_dir=args.data_dir, data_file_name=args.data)
        if t_match is not None and args.freq_match:
            ax.axvline(t_match * 1e9, linestyle="--", linewidth=1.5, color="darkred", label=r"$f_{\rm GW} = f_{\rm cav}$")
        
        ax.set_xlabel(r"$t\,[\mathrm{ns}]$")
        ax.set_ylabel(r"$\mathrm{preRHS}(t)$")
        ax.set_title(rf"$\mathrm{{preRHS}}(t)$ for {args.mode};"+f"\n waveform file: {args.data}" )
        ax.legend()
        save_figure(
            fig,
            os.path.join(save_dir,  f"pre_RHS(t)_{args.mode}_{args.data}.png"),
        )


if __name__ == "__main__":
    main()
