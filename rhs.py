import os
from gw.utils import load_waveform
from rhs.utils import load_slice_integrals
from rhs.rhs_integration import compute_rhs_time_series
import numpy as np
import argparse

from plotting import new_figure, save_figure
from misc.resonant_frequency_matches import find_chirp_match_time, load_cavity_frequency_from_run_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to .npy data file")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing the .npy data file and its config")
    parser.add_argument("--results-dir", type=str, default="results", help="Path to results directory")
    parser.add_argument("--Nt", type=int, default=1000, help="Number of time samples")
    parser.add_argument("--Ns", type=int, default=100, help="Number of spatial samples")
    
    parser.add_argument("--NLGrid", action="store_true", help="Enable grid adaptation for VEGAS integration")
    parser.add_argument("--density-boost", type=float, default=5.0, help="Density boost factor for non-uniform time grid (if NLGrid is enabled)")
    
    parser.add_argument("--freq-match", action="store_true", help="Match GW frequency to cavity resonant frequency and indicate it on the plot")

    parser.add_argument("--method", choices=["quad", "vegas"], default="quad", help="Integration method")
    parser.add_argument("--nproc", type=int, default=1, help="Number of processors for parallel integration")
    
    parser.add_argument("--geometry", choices=["rectangular", "cylindrical", "spherical"], default="cylindrical", help="Cavity geometry type")
    parser.add_argument("--mode-fam", choices=["TE", "TM"], help="Mode family (TE or TM)")
    parser.add_argument("--mode-par", choices=["a", "b", None], help="Mode parameter")
    parser.add_argument("--mode-ind", default="0,1,0", help="Mode indices as comma-separated integers")
    
    parser.add_argument("--L", default=0.1, help="Cavity length scale parameter")

    parser.add_argument("--theta", type=float, default=45.0, help="Polar angle of gravitational wave approach in degrees")
    parser.add_argument("--phi", type=float, default=0.0, help="Azimuthal angle of gravitational wave approach in degrees")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.mode_par is not None:
        mode_name = args.mode_fam + args.mode_par
    else:
        mode_name = args.mode_fam

    data = os.path.join(args.data_dir, args.data + ".npy")

    save_dir, area_data = load_slice_integrals(data, args.results_dir, args.geometry, mode_name, args.mode_ind, args.theta, args.phi, args.Ns)
    x_par_arr, E_plus, E_cross = area_data
    t_data, hplus_dd, hcross_dd = load_waveform(data, derivative=2)

    ts, RHS = compute_rhs_time_series(
        t_data, x_par_arr, E_plus, E_cross, 
        hplus_dd, hcross_dd, Nt=args.Nt, L=args.L,
        method=args.method,
        nproc=args.nproc,
        NLGrid=args.NLGrid, 
        density_boost=args.density_boost
    )
    
    # Save RHS array to file
    file_name = os.path.join(save_dir, f"RHS_{args.geometry}_{mode_name}_{args.mode_ind}_{args.data}.npz")
    np.savez(file_name, ts=ts, RHS=RHS, mode_name=mode_name, mode_ind=args.mode_ind, data=args.data)
    print("[INFO] Saved RHS array file to ", file_name)

    # Plot RHS array as a function of time
    fig, ax = new_figure()
    ax.plot(ts * 1e9, RHS, label="RHS (cylinder slicing method)")

    # Add vertical line for resonant time if frequency matching is enabled
    f_cavity = load_cavity_frequency_from_run_config(args.results_dir, args.geometry, mode_name, args.mode_ind, args.theta, args.phi)
    t_match, _ = find_chirp_match_time(ts=ts, f_cavity=f_cavity, data_dir=args.data_dir, data_file_name=args.data)
    if t_match is not None and args.freq_match:
        ax.axvline(t_match * 1e9, linestyle="--", linewidth=1.5, color="darkred", label=r"$f_{\rm GW} = f_{\rm cav}$")

    ax.set_xlabel(r"$t\,[\mathrm{ns}]$")
    ax.set_ylabel(r"$\mathrm{RHS}(t)$")
    ax.set_title( f"$\mathrm{{RHS}}(t)$ for {args.geometry} cavity mode {mode_name} [{args.mode_ind}]; \n waveform file: {args.data}" )
    ax.legend()
    save_figure(
        fig,
        os.path.join(save_dir,  f"RHS(t)_{args.geometry}_{mode_name}_{args.mode_ind}_{args.data}.png"),
    )

if __name__ == "__main__":
    main()
