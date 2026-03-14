import os, argparse, time
import matplotlib.pyplot as plt
import numpy as np
from misc.resonant_frequency_matches import find_chirp_match_time, load_cavity_frequency_from_run_config
from ode.solver import solve_mode_amplitude
from ode.utils import load_rhs, save_amplitude, extend_rhs, compute_b, compute_U, load_from_config, clip_at_zero_crossing, taper_signal
from plotting import new_figure, save_figure

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results", help="Path to results directory")
    parser.add_argument("--data", type=str, required=True, help="Path to .npy data file")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing the .npy data file and its config")

    parser.add_argument("--theta", type=float, default=45.0, help="Polar angle of gravitational wave approach in degrees")
    parser.add_argument("--phi", type=float, default=0.0, help="Azimuthal angle of gravitational wave approach in degrees")

    parser.add_argument("--Q", type=float, default=1e5, help="Quality factor of the cavity")
    parser.add_argument("--geometry", choices=["rectangular", "cylindrical", "spherical"], default="cylindrical", help="Cavity geometry type")
    parser.add_argument("--mode-fam", choices=["TE", "TM"], default="TM", help="Mode family (TE or TM)")
    parser.add_argument("--mode-par", choices=["a", "b", None], default="b", help="Mode parity (even or odd)")
    parser.add_argument("--mode-ind", default="0,1,0", help="Mode indices as comma-separated values")

    parser.add_argument("--freq-match", action="store_true", help="Match GW frequency to cavity resonant frequency and indicate it on the plot")

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

    ts, RHS, pre_RHS = load_rhs(rhs_path)

    #ts, RHS = clip_at_zero_crossing(ts, RHS)
    RHS = taper_signal(RHS)
    
    ts_ext, RHS, RHS_fn = extend_rhs(ts, RHS, args.extend)
    
    if pre_RHS.size > 0:
        _, pre_RHS, pre_RHS_fn = extend_rhs(ts, pre_RHS, args.extend)
    else:
        pre_RHS = np.zeros_like(RHS)

    start = time.time()

    print("[INFO] Solving the ODE with a given RHS(t)...")
    result = solve_mode_amplitude(
        ts=ts_ext, RHS_fn=RHS_fn,
        omega=omega, Q=args.Q,
    )
    print(f"[INFO] ODE solved in {time.time()-start:.2f} s.")
    
    print("[INFO] Computing magnetic mode coefficients...")
    c_t = result['c']
    cD_t = result['cD']
    b_t = compute_b(c_t, cD_t, pre_RHS, args.Q, omega)
    U = compute_U(c_t, b_t)

    result['b'] = b_t
    result['U'] = U
    
    print("[INFO] Magnetic mode coefficients computed.")

    save_amplitude(save_dir, result)

    plots = [
    (
        c_t,
        r"Mode amplitude $c(t)$",
        r"Mode amplitude $c(t)$",
        rf"$c(t)$ for {args.geometry} cavity mode {mode_name} [{args.mode_ind}];"
        f"\n waveform file: {args.data}",
        f"Mode_c_amplitude_{args.geometry}_{mode_name}_{args.mode_ind}_{args.data}.png",
    ),
    (
        b_t,
        r"Mode amplitude $b(t)$",
        r"Mode amplitude $b(t)$",
        rf"$b(t)$ for {args.geometry} cavity mode {mode_name} [{args.mode_ind}];"
        f"\n waveform file: {args.data}",
        f"Mode_b_amplitude_{args.geometry}_{mode_name}_{args.mode_ind}_{args.data}.png",
    ),
    (
        U,
        r"Energy $U(t)$",
        r"Energy $U$, [J]",
        rf"$U(t)$ for {args.geometry} cavity mode {mode_name} [{args.mode_ind}];"
        f"\n waveform file: {args.data}",
        f"Energy_{args.geometry}_{mode_name}_{args.mode_ind}_{args.data}.png",
    ),]

    for y, label, ylabel, title, filename in plots:
        fig, ax = new_figure()
        ax.plot(ts_ext * 1e9, y, label=label)

        # Add vertical line for resonant time if frequency matching is enabled
        f_cavity = load_cavity_frequency_from_run_config(args.results_dir, args.geometry, mode_name, args.mode_ind, args.theta, args.phi)
        t_match, _ = find_chirp_match_time(ts=ts, f_cavity=f_cavity, data_dir=args.data_dir, data_file_name=args.data)
        if t_match is not None and args.freq_match:
            ax.axvline(t_match * 1e9, linestyle="--", linewidth=1.5, color="darkred", label=r"$f_{\rm GW} = f_{\rm cav}$")

        ax.set_xlabel(r"$t$, [ns]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        save_figure(fig, os.path.join(save_dir, filename))

if __name__ == "__main__":
    main()
