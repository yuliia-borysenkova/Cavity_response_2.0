import os, argparse, time
import matplotlib.pyplot as plt
import numpy as np
from misc.resonant_frequency_matches import find_chirp_match_time
from ode.solver import solve_mode_amplitude
from ode.utils import (
    load_rhs, save_amplitude, extend_rhs, compute_b, compute_U,
    load_from_config, clip_at_zero_crossing, taper_signal,
    apply_onset_smoothing,
)
from plotting import new_figure, save_figure

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")

    parser.add_argument("--theta", type=float, default=45.0)
    parser.add_argument("--phi",   type=float, default=0.0)
    parser.add_argument("--Ns",    type=int,   default=100)

    parser.add_argument("--Q",        type=float, default=1e5)
    parser.add_argument("--geometry", choices=["rectangular", "cylindrical", "spherical"], default="cylindrical")
    parser.add_argument("--mode-fam", choices=["TE", "TM"], default="TM")
    parser.add_argument("--mode-par", choices=["a", "b", None], default="b")
    parser.add_argument("--mode-ind", default="0,1,0")
    parser.add_argument("--mode",     type=str, default=None)

    parser.add_argument("--freq-match",      action="store_true")
    parser.add_argument("--extend",          type=float, default=1.0)
    parser.add_argument("--onset-smoothing", action="store_true",
                        help="Apply smooth onset ramp to RHS before ODE solve.")

    # Fine-grained onset control (both optional; auto-estimated when absent)
    parser.add_argument("--onset-threshold", type=float, default=1e-3,
                        help="Fraction of peak RHS used to detect signal onset (default 1e-3).")
    parser.add_argument("--onset-i0",    type=int, default=None,
                        help="Override onset index (samples). Auto-detected when absent.")
    parser.add_argument("--onset-width", type=int, default=None,
                        help="Override ramp width (samples). Auto-estimated when absent.")

    return parser.parse_args()


def main():
    args = parse_args()
    
    dir_name2 = f"DATA_{args.data}"
    
    if args.mode is not None:
        dir_name1 = f"{args.mode}_theta={args.theta}_phi={args.phi}_Ns={args.Ns}"
        rhs_description  = f"{args.mode}"
        mode_description = f"{args.mode}_{args.data}"
        run_dir  = os.path.join(args.results_dir, dir_name1, dir_name2)
    else:
        if args.mode_par is not None:
            mode_name = args.mode_fam + args.mode_par
        else:
            mode_name = args.mode_fam
        dir_name1        = f"{args.geometry}_{mode_name}_{args.mode_ind}_theta={args.theta}_phi={args.phi}_Ns={args.Ns}"
        rhs_description  = f"{args.geometry} cavity mode {mode_name} [{args.mode_ind}]"
        mode_description = f"{args.geometry}_{mode_name}_{args.mode_ind}_{args.data}"
        run_dir  = os.path.join(args.results_dir, dir_name1)
        
    rhs_filename = f"RHS_" + mode_description + ".npz"
    save_dir  = os.path.join(args.results_dir, dir_name1, dir_name2)
    rhs_path  = os.path.join(save_dir, rhs_filename)

    omega, norm = load_from_config(run_dir)

    ts, RHS, pre_RHS = load_rhs(rhs_path)

    # ------------------------------------------------------------------ #
    # Signal conditioning — ORDER MATTERS:                                 #
    #   1. taper tail  (avoid Gibbs at the end)                            #
    #   2. onset smooth on the *original* (non-extended) array             #
    #      → onset index is meaningful on the compact signal               #
    #   3. extend with zeros for longer ODE integration                    #
    # ------------------------------------------------------------------ #

    RHS = taper_signal(RHS)

    if args.onset_smoothing:
        ts, RHS, _ = apply_onset_smoothing(
            ts, RHS,
            i0=args.onset_i0,
            width=args.onset_width,
            threshold_ratio=args.onset_threshold,
            omega_cavity=omega,       # physics-informed ramp width
            verbose=True,
        )

    ts_ext, RHS, RHS_fn = extend_rhs(ts, RHS, args.extend)

    if pre_RHS.size > 0:
        if args.onset_smoothing:
            ts, pre_RHS, _ = apply_onset_smoothing(
                ts, pre_RHS,
                i0=args.onset_i0,
                width=args.onset_width,
                threshold_ratio=args.onset_threshold,
                omega_cavity=omega,
                verbose=False,
            )
        _, pre_RHS, pre_RHS_fn = extend_rhs(ts, pre_RHS, args.extend)
    else:
        pre_RHS = np.zeros_like(RHS)

    # ------------------------------------------------------------------ #

    start = time.time()
    print("[INFO] Solving the ODE with a given RHS(t)...")
    result = solve_mode_amplitude(
        ts=ts_ext, RHS_fn=RHS_fn,
        omega=omega, Q=args.Q,
    )
    print(f"[INFO] Computed in {time.time()-start:.2f} s.")
    
    print("[INFO] Computing magnetic mode coefficients...")
    c_t  = result['c']
    cD_t = result['cD']
    b_t  = compute_b(c_t, cD_t, pre_RHS, args.Q, omega)
    U    = compute_U(c_t, b_t)

    result['b'] = b_t
    result['U'] = U

    print("[INFO] Magnetic mode coefficients computed.")
    save_amplitude(save_dir, result)

    plots = [
        (c_t,  r"Mode amplitude $c(t)$", r"Mode amplitude $c(t)$",
         rf"$c(t)$ for " + rhs_description + f"\n waveform file: {args.data}",
         f"Mode_c_amplitude_" + mode_description + ".png"),
        (b_t,  r"Mode amplitude $b(t)$", r"Mode amplitude $b(t)$",
         rf"$b(t)$ for " + rhs_description + f"\n waveform file: {args.data}",
         f"Mode_b_amplitude_" + mode_description + ".png"),
        (U,    r"Energy $U(t)$",          r"Energy $U$, [J]",
         rf"$U(t)$ for " + rhs_description + f"\n waveform file: {args.data}",
         f"Energy_" + mode_description + ".png"),
    ]

    for y, label, ylabel, title, filename in plots:
        fig, ax = new_figure()
        ax.plot(ts_ext * 1e9, y, label=label)

        f_cavity = omega / (2 * np.pi)
        t_match, _ = find_chirp_match_time(
            ts=ts_ext, f_cavity=f_cavity,
            data_dir=args.data_dir, data_file_name=args.data,
        )
        if t_match is not None and args.freq_match:
            ax.axvline(t_match * 1e9, linestyle="--", linewidth=1.5,
                       color="darkred", label=r"$f_{\rm GW} = f_{\rm cav}$")

        ax.set_xlabel(r"$t$ [ns]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        save_figure(fig, os.path.join(save_dir, filename))


if __name__ == "__main__":
    main()
