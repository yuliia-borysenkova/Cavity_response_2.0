import os, argparse, time
import matplotlib.pyplot as plt
import numpy as np
from misc.resonant_frequency_matches import find_chirp_match_time
from ode.solver import solve_mode_amplitude
from ode.utils import (
    load_rhs, save_amplitude, extend_rhs, compute_b, compute_U,
    load_from_config,
    apply_onset_smoothing, update_config_with_Q, analytical_free_decay, compute_full_fourier,
)
from plotting import new_figure, save_figure

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Name of the .npy data file (without extension)")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing the .npy data file and its config")
    parser.add_argument("--results-dir", type=str, default="results", help="Path to results directory")

    parser.add_argument("--theta", type=float, default=45.0, help="Polar angle of GW incidence (deg)")
    parser.add_argument("--phi",   type=float, default=0.0, help="Azimuthal angle of GW incidence (deg) ")
    parser.add_argument("--Ns",    type=int,   default=100, help="Number of spatial steps")

    parser.add_argument("--Q",        type=float, default=1e5, help="Quality factor of the cavity mode")
    parser.add_argument("--geometry", choices=["rectangular", "cylindrical", "spherical"], default="cylindrical", help="Cavity geometry type")
    parser.add_argument("--mode-fam", choices=["TE", "TM"], default="TM", help="Mode family (TE or TM)")
    parser.add_argument("--mode-par", choices=["a", "b", None], default="b", help="Mode parity: 'a' for even, 'b' for odd, None for no parity")
    parser.add_argument("--mode-ind", default="0,1,0", help="Mode indices as comma-separated values, e.g. '0,1,0'")
    parser.add_argument("--mode",     type=str, default=None, help="Optional mode name for backward compatibility with old config structure. If provided, overrides geometry/mode-fam/mode-par/mode-ind arguments.")

    parser.add_argument("--freq-match", action="store_true", help="Match GW frequency to cavity resonant frequency and indicate it on the plot")

    parser.add_argument("--extend",          type=float, default=1.0, help="Extend RHS time series by this factor (default 1.0, i.e. no extension).")
    parser.add_argument("--onset-smoothing", action="store_true",
                        help="Apply smooth onset ramp to RHS before ODE solve.")

    #Onset smoothing parameters
    parser.add_argument("--onset-n-periods", type=int, default=3,
                        help="Number of oscillation periods for onset ramp (default 5).")

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

    if args.onset_smoothing:
        ts, RHS, RHS_fn = apply_onset_smoothing(
            ts, RHS,
            n_periods = args.onset_n_periods
        )

    ts_ext, RHS, RHS_fn = extend_rhs(ts, RHS, args.extend)

    if pre_RHS.size > 0:
        if args.onset_smoothing:
            ts, pre_RHS, _ = apply_onset_smoothing(
                ts, pre_RHS,
                n_periods = args.onset_n_periods
            )
        _, pre_RHS, pre_RHS_fn = extend_rhs(ts, pre_RHS, args.extend)
    else:
        pre_RHS = np.zeros_like(RHS)

    #Plot RHS after filters
    fig, ax = new_figure()
    ax.plot(ts * 1e9, RHS[:len(ts)], label="RHS", linewidth=1.5)
    ax.set_xlabel(r"$t\,[\mathrm{ns}]$")
    ax.set_ylabel(r"$\mathrm{RHS}(t)$")
    ax.set_title(rf"$\mathrm{{RHS}}(t)$ after filters for {args.geometry} cavity mode {mode_name} [{args.mode_ind}];"+f"\n waveform file: {args.data}" )
    ax.legend()
    save_figure(fig, os.path.join(save_dir, f"RHS(t)_filtered.png"))

    #Plot preRHS after filters
    fig, ax = new_figure()
    ax.plot(ts * 1e9, pre_RHS[:len(ts)], label="pre_RHS", linewidth=1.5)
    ax.set_xlabel(r"$t\,[\mathrm{ns}]$")
    ax.set_ylabel(r"$\mathrm{preRHS}(t)$")
    ax.set_title(rf"$\mathrm{{preRHS}}(t)$ after filters for {args.geometry} cavity mode {mode_name} [{args.mode_ind}];"+f"\n waveform file: {args.data}" )
    ax.legend()
    save_figure(fig, os.path.join(save_dir, f"pre_RHS(t)_filtered.png"))

    # ------------------------------------------------------------------ #

    start = time.time()
    print("[INFO] Solving the ODE with a given RHS(t)...")
    # result = solve_mode_amplitude(
    #     ts=ts_ext, RHS_fn=RHS_fn,
    #     omega=omega, Q=args.Q,
    # )

    result = solve_mode_amplitude(
        ts=ts, RHS_fn=RHS_fn,        # <-- ts, not ts_ext
        omega=omega, Q=args.Q,
        )   
    # extend the solution in free-decay region using analytical formula
    result = analytical_free_decay(result, ts_ext, omega, args.Q)

    print(f"[INFO] Computed in {time.time()-start:.2f} s.")

    #Furier transform of the solution for mode amplitude c(t)
    freqs, c_hat_num, c_hat_ana, c_hat_total = compute_full_fourier(result, ts_ext, n_driven=len(ts))
    result['freqs']            = freqs
    result['c_hat_numerical']  = c_hat_num
    result['c_hat_analytical'] = c_hat_ana
    result['c_hat_total']      = c_hat_total

    #Furier plot for c(t)
    fig, ax = new_figure()
    ax.plot(freqs / (2 * np.pi) * 1e-9, np.abs(c_hat_total),   lw=1.5, label=r"Total $|\hat{c}(\nu)|$")
    ax.plot(freqs / (2 * np.pi) * 1e-9, np.abs(c_hat_num),     lw=1.0, linestyle='--', label=r"Numerical (driven)")
    ax.plot(freqs / (2 * np.pi) * 1e-9, np.abs(c_hat_ana),     lw=1.0, linestyle='--', label=r"Analytical (tail)")
    ax.axvline(omega / (2 * np.pi) * 1e-9, color='k', linestyle=':', lw=1.0, label=r"$f_{\rm cav}$")
    f_cav_GHz = omega / (2 * np.pi) * 1e-9
    ax.set_xlim(f_cav_GHz * 0.999, f_cav_GHz * 1.001)  # ±0.1% around resonance
    ax.set_xlabel(r"$f$ [GHz]")
    ax.set_ylabel(r"$|\hat{c}(\nu)|$")
    ax.set_title(rf"Fourier transform of $c(t)$ for " + rhs_description + f"\n waveform file: {args.data}")
    ax.legend()
    save_figure(fig, os.path.join(save_dir, f"Fourier_c_" + mode_description + ".png"))

    print("[INFO] Computing magnetic mode coefficients...")
    c_t  = result['c']
    cD_t = result['cD']
    b_t  = compute_b(c_t, cD_t, pre_RHS, args.Q, omega)
    U    = compute_U(c_t, b_t)

    result['b'] = b_t
    result['U'] = U

    print("[INFO] Magnetic mode coefficients computed.")

    #editing json file to add Q
    output_file = os.path.join(save_dir, f"config_{args.geometry}_{mode_name}_{args.mode_ind}_{args.data}.json")
    update_config_with_Q(output_file, args)

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
