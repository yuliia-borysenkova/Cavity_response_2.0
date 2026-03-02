import os
import argparse
import matplotlib.pyplot as plt
from ode.solver import solve_mode_amplitude
from ode.utils import build_mode_from_config, load_rhs, save_amplitude, load_run_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results", help="Path to results directory")
    parser.add_argument("--data", type=str, help="Path to .npy data file")
    
    parser.add_argument("--theta", type=float, default=45.0, help="Polar angle of gravitational wave approach in degrees")
    parser.add_argument("--phi", type=float, default=0.0, help="Azimuthal angle of gravitational wave approach in degrees")

    parser.add_argument("--Q", type=float, default=1e5)
    parser.add_argument("--geometry", choices=["rectangular", "cylindrical", "spherical"], default="cylindrical")
    parser.add_argument("--mode-fam", choices=["TM", "TM"])
    parser.add_argument("--mode-par", choices=["a", "b", None], help="Cavity mode to excite")
    
    parser.add_argument("--mode-ind", type=str, default="0,1,0", help="Mode indices [n,p,q] as comma-separated values")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.mode_par is not None:
        mode_name = args.mode_fam + args.mode_par
    else:
        mode_name = args.mode_fam

    dir_name1 = f"{args.geometry}_{mode_name}_{args.mode_ind}_theta={args.theta}_phi={args.phi}"
    dir_name2 = f"DATA_{args.data}"
    
    run_dir = os.path.join(args.results_dir, dir_name1)
    save_dir = os.path.join(args.results_dir, dir_name1, dir_name2)
    rhs_filename = f"RHS_{args.geometry}_{mode_name}_{args.mode_ind}_{args.data}.npz"
    rhs_path = os.path.join(save_dir, rhs_filename)

    cfg = load_run_config(run_dir)
    omega = build_mode_from_config(cfg)

    ts, RHS, RHS_fn = load_rhs(rhs_path)

    print("[INFO] Solving the ODE with a given RHS(t)...")

    result = solve_mode_amplitude(
        ts=ts, RHS_fn=RHS_fn,
        omega=omega, Q=args.Q,
    )
    print("[INFO] ODE solved.")

    save_amplitude(save_dir, result)

    # --- Plot: Mode amplitude ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ts * 1e9, result["c"], label="Mode amplitude c(t)")
    ax.set_xlabel("t [ns]")
    ax.set_ylabel("Mode amplitude c(t)")
    ax.set_title(f"c(t) for {args.geometry} cavity mode {mode_name} {args.mode_ind} and waveform file: {args.data}")
    ax.grid(True)
    ax.legend()
    fig.savefig(os.path.join(save_dir, f"Mode_amplitude_{args.geometry}_{mode_name}_{args.mode_ind}_{args.data}.png"), dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    main()
