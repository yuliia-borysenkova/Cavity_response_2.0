import os
from gw.utils import load_waveform
from rhs.utils import load_slice_integrals
from rhs.rhs_integration import compute_rhs_time_series, CubicSplineInterp
import matplotlib.pyplot as plt
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--Nt", type=int, default=1000)
    parser.add_argument("--Ns", type=int, default=100)
    
    parser.add_argument("--NLGrid", action="store_true", help="Enable grid adaptation for VEGAS integration")
    parser.add_argument("--density-boost", type=float, default=5.0, help="Density boost factor for non-uniform time grid (if NLGrid is enabled)")
    
    parser.add_argument("--method", choices=["quad", "vegas"], default="quad")
    parser.add_argument("--nproc", type=int, default=1)
    
    parser.add_argument("--geometry", choices=["rectangular", "cylindrical", "spherical"], default="cylindrical")
    parser.add_argument("--mode-fam", choices=["TE", "TM"])
    parser.add_argument("--mode-par", choices=["a", "b", None])
    parser.add_argument("--mode-ind", default="0,1,0")
    
    parser.add_argument("--L", default=0.1)

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

    save_dir, area_data = load_slice_integrals(data, args.results_dir, args.geometry, mode_name, args.mode_ind, args.theta, args.phi, Ns=args.Ns)
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
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ts * 1e9, RHS, label="RHS (cylinder slicing method)")
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("RHS(t)")
    ax.set_title(f"RHS(t) for {args.geometry} cavity mode {mode_name} {args.mode_ind} and waveform file: {args.data}")
    ax.legend()
    ax.grid(True)
    fig.savefig(os.path.join(save_dir, f"RHS(t)_{args.geometry}_{mode_name}_{args.mode_ind}_{args.data}.png"))
    #plt.show()

if __name__ == "__main__":
    main()
