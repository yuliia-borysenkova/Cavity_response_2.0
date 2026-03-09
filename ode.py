import os
import argparse
import matplotlib.pyplot as plt
from ode.solver import solve_mode_amplitude
from ode.utils import load_rhs, save_amplitude, extend_rhs, compute_b, load_from_config, start_at_zero_crossing
from scipy.constants import epsilon_0

from plotting import new_figure, save_figure

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

    parser.add_argument("--extend", type=float, default=1.0, help="Extend the computation time to x the signal time")
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

    omega, norm = load_from_config(run_dir)

    ts, RHS, RHS_fn = load_rhs(rhs_path)

    ts, RHS = start_at_zero_crossing(ts, RHS)
    
    if args.extend != 1.0:
        ts, RHS, RHS_fn = extend_rhs(ts, RHS, args.extend)

    print("[INFO] Solving the ODE with a given RHS(t)...")
    result = solve_mode_amplitude(
        ts=ts, RHS_fn=RHS_fn,
        omega=omega, Q=args.Q,
    )
    print("[INFO] ODE solved.")
    
    print("[INFO] Computing magnetic mode coefficients...")
    c_t = result['c']
    b_t = compute_b(ts, c_t, omega)
    print("[INFO] Magnetic mode coefficients computed.")
    
    E = 1/2 * epsilon_0 * norm**2 * (c_t**2 + b_t**2)

    result['b'] = b_t
    result['E'] = E
    save_amplitude(save_dir, result)

    plots = [
    (
        result["c"],
        r"Mode amplitude $\mathrm{{c}}(t)$",
        r"Mode amplitude $\mathrm{{c}}(t)$",
        rf"$\mathrm{{c}}(t)$ for {args.geometry} cavity mode {mode_name} [{args.mode_ind}];"
        f"\n waveform file: {args.data}",
        f"Mode_c_amplitude_{args.geometry}_{mode_name}_{args.mode_ind}_{args.data}.png",
    ),
    (
        b_t,
        r"Mode amplitude $\mathrm{{b}}(t)$",
        r"Mode amplitude $\mathrm{{b}}(t)$",
        rf"$\mathrm{{b}}(t)$ for {args.geometry} cavity mode {mode_name} [{args.mode_ind}];"
        f"\n waveform file: {args.data}",
        f"Mode_b_amplitude_{args.geometry}_{mode_name}_{args.mode_ind}_{args.data}.png",
    ),
    (
        E,
        r"Energy $\mathrm{{E}}(t)$",
        r"Energy $\mathrm{{U}}$ [J]",
        rf"$\mathrm{{E}}(t)$ for {args.geometry} cavity mode {mode_name} [{args.mode_ind}];"
        f"\n waveform file: {args.data}",
        f"Energy_{args.geometry}_{mode_name}_{args.mode_ind}_{args.data}.png",
    ),]

    for y, label, ylabel, title, filename in plots:
        fig, ax = new_figure()
        ax.plot(ts * 1e9, y, label=label)
        ax.set_xlabel(r"$\mathrm{{t}}$ [ns]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        save_figure(fig, os.path.join(save_dir, filename))

if __name__ == "__main__":
    main()
