import os, argparse, time
import numpy as np
import scipy.io
from scipy.constants import epsilon_0, mu_0
from output.utils import create_plot, compare_contributions, rebuild_s11_grid, fourier_interp

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--data", type=str, required=True)

    parser.add_argument("--geometry", choices=["rectangular", "cylindrical", "spherical"], default="cylindrical")

    parser.add_argument("--theta", type=float, default=45.0)
    parser.add_argument("--phi", type=float, default=0.0)

    parser.add_argument("--Ns", type=int, default=100)

    parser.add_argument("--epsilon-r", type=float, default=2.08)

    parser.add_argument("--a", type=float, default=2.11e-3)
    parser.add_argument("--b", type=float, default=0.635e-3)

    return parser.parse_args()
 
def main():

    args = parse_args()

    print("[INFO] Computation started...")

    start = time.time()

    os.makedirs(args.results_dir, exist_ok=True)

    s11_vals = scipy.io.loadmat("S11 sweep - step 2 kHz.mat")["S11"].flatten()
    s11_freqs = scipy.io.loadmat("frequency sweep - step 2 kHz.mat")["freq_vec"].flatten()

    F_dict = {"TMb_0,1,0": -0.0081, "TMb_0,1,1": -0.0116,"TMb_0,1,2": -0.0120}
    freq_dict = {"TMb_0,1,0": 0.5e9, "TMb_0,1,1": 582.960e6,"TMb_0,1,2": 780.685e6}

    s11_freqs, s11_vals = rebuild_s11_grid(s11_freqs, s11_vals, freq_dict)
    I_gw_freq = np.zeros_like(s11_freqs, dtype=np.complex128)

    for mode_name, F_m1 in F_dict.items():

        folder_name = (f"{args.geometry}_{mode_name}_theta={args.theta}_phi={args.phi}_Ns={args.Ns}")

        mode_dir = os.path.join(args.results_dir, folder_name)
        data_dir = os.path.join(mode_dir, f"DATA_{args.data}")
        file_path = os.path.join(data_dir, "amplitude_package.npz")

        pkg = np.load(file_path, allow_pickle=True)

        freqs = pkg["freqs"]
        dt = pkg["ts"][1] - pkg["ts"][0]
        c_hat_num = pkg["c_hat_numerical"]

        threshold = 5e-2
        _, freqs_exceed = compare_contributions(pkg, c_hat_num, freqs, s11_freqs, threshold)

        omegas = s11_freqs * 2 * np.pi
        s = pkg['alpha'] + 1j * omegas
        c_hat_ana = np.exp(-1j * omegas * pkg['t0']) * \
                        (pkg['A_n'] * s + pkg['B_n'] * pkg['omega_d']) / \
                        (s**2 + pkg['omega_d']**2)

        if freqs_exceed.size == 0:
            print("[INFO] No frequencies exceed the threshold. Using analytical part only.")
            c_hat = c_hat_ana

        else:
            print(f"[INFO] {len(freqs_exceed)} frequencies exceed {threshold:g}. Using Fourier interpolation for the numerical part.")
            c_hat_num_interp = fourier_interp(c_hat_num, freqs * 2 * np.pi, s11_freqs * 2 * np.pi, dt)
            c_hat = c_hat_num_interp + c_hat_ana

        I_gw_freq += F_m1 * c_hat * (freq_dict[mode_name] / s11_freqs) * 1j

        print("[INFO] Mode", mode_name, "done.")

    Y_w = np.sqrt(epsilon_0 * args.epsilon_r / mu_0)
    Y_c = Y_w * (1 - s11_vals) / (1 + s11_vals)
    V_c = I_gw_freq / (Y_w + Y_c)

    V_meas = V_c * np.sqrt(np.log(args.a / args.b) / (2 * np.pi))

    P_w = (np.abs(I_gw_freq) ** 2 / (2 * np.abs(Y_w + Y_c) ** 2) * np.real(Y_w))

    np.save(
        os.path.join(args.results_dir, "output.npy"),
        {
            "freqs": s11_freqs,
            "V_meas": V_meas,
            "P_w": P_w,
            "I_gw_freq": I_gw_freq
        }
    )

    plots = [

        {
            "filename": os.path.join(args.results_dir, "voltage_spectrum.png"),
            "xlabel": "Frequency [GHz]",
            "ylabel": r"$|V_{\mathrm{meas}}|$ [V]",
            "title": "Measured Voltage Spectrum",
            "xlim": (0.45, 0.9),
            "yscale": "log",
            "curves": [
                {
                    "x": s11_freqs / 1e9,
                    "y": np.abs(V_meas),
                    "plot_kwargs": {
                        "color": "#a00000"
                    },
                }
            ],
        },

        {
            "filename": os.path.join(args.results_dir, "power_spectrum.png"),
            "xlabel": "Frequency [GHz]",
            "ylabel": "Power [W]",
            "title": "Detected Power Spectrum",
            "xlim": (0.45, 0.9),
            "yscale": "log",
            "curves": [
                {
                    "x": s11_freqs / 1e9,
                    "y": P_w,
                    "plot_kwargs": {
                        "color": "#a00000"
                    },
                }
            ],
        },

        {
            "filename": os.path.join(args.results_dir, "I_gw_mag.png"),
            "xlabel": "Frequency [GHz]",
            "ylabel": r"$|I_\mathrm{gw}|$ [A]",
            "title": "Gravitational Wave Current Spectrum",
            "xlim": (0.46, 0.56),
            "ylim": (1e-17, 1e-12),
            "yscale": "log",
            "legend": True,
            "curves": [
                {
                    "x": s11_freqs / 1e9,
                    "y": np.abs(I_gw_freq),
                    "label": "interpolated",
                    "plot_kwargs": {
                        "marker": ".",
                        "markersize": 0.7,
                        "linestyle": "None",
                        "color": "#a00000"
                    },
                },
            ],
        },

        {
            "filename": os.path.join(args.results_dir, "I_gw_phase.png"),
            "xlabel": "Frequency [GHz]",
            "ylabel": r"$\mathrm{phase}[I_\mathrm{gw}], \mathrm{degrees}$",
            "title": "Gravitational Wave Current Spectrum",
            "xlim": (0.46, 0.56),
            "legend": True,
            "curves": [
                {
                    "x": s11_freqs / 1e9,
                    "y": np.angle(I_gw_freq) / np.pi * 180,
                    "label": "interpolated",
                    "plot_kwargs": {
                        "marker": ".",
                        "markersize": 0.7,
                        "linestyle": "None",
                        "color": "#a00000"
                    },
                },
            ],
        },

        {
            "filename": os.path.join(args.results_dir, "s11_mag.png"),
            "xlabel": "Frequency [GHz]",
            "ylabel": r"$|S_{11}|, \mathrm{dB}$",
            "title": r"$|S_{11}|$ as a Function of Frequency",
            "curves": [
                {
                    "x": s11_freqs / 1e9,
                    "y": 20 * np.log10(np.abs(s11_vals)),
                    "plot_kwargs": {
                        "color": "#a00000"
                    },
                }
            ],
        },

        {
            "filename": os.path.join(args.results_dir, "s11_phase.png"),
            "xlabel": "Frequency [GHz]",
            "ylabel": r"$\mathrm{phase}[S_{11}], \mathrm{degrees}$",
            "title": r"$S_{11}$ phase as a Function of Frequency",
            "curves": [
                {
                    "x": s11_freqs / 1e9,
                    "y": np.angle(s11_vals) / np.pi * 180,
                    "plot_kwargs": {
                        "color": "#a00000"
                    },
                }

            ],
        },

    ]

    for spec in plots:
        create_plot(spec)

    end = time.time()

    print(f"[INFO] Computation complete, elapsed time = {end - start:.2f} s.")
    print(f"[INFO] Results saved to: {args.results_dir}")


if __name__ == "__main__":

    main()
