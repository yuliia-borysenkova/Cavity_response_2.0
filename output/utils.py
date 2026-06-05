import numpy as np
from tqdm import tqdm
from plotting import new_figure, save_figure

def fourier_interp(c_hat, omega_k, omega, dt):
    N = len(c_hat)

    c_hat_interp = np.zeros_like(omega, dtype=np.complex128)

    for i, c_hat_i in tqdm(enumerate(c_hat), total=N):
        x = (omega_k[i] - omega) * dt

        small = np.abs(x) < 1e-7
        kernel = np.zeros_like(x, dtype=np.complex128)
        kernel[~small] = (1 - np.exp(1j * N * x[~small])) / (1 - np.exp(1j * x[~small]))
        kernel[small] = N

        c_hat_interp += c_hat_i * kernel
        
    return 1/N * c_hat_interp

def fourier_interp_narrow(c_hat, omega_k, omega, dt):
    N = len(c_hat)

    c_hat_interp = np.zeros_like(omega, dtype=np.complex128)

    # only keep near-resonant indices
    tol = 10000 / (N)

    for i, c_hat_i in tqdm(enumerate(c_hat), total=N):

        x = (omega_k[i] - omega) * dt

        # drop highly oscillatory contributions
        if np.any(np.abs(x) > tol):
            continue

        c_hat_interp += c_hat_i * (1 - np.exp(1j * N * x)) / (1 - np.exp(1j * x))

    return c_hat_interp / N

def rebuild_s11_grid(s11_freqs, s11_vals, freq_dict, coarse_step=1000, df=5e6):

    important = np.zeros_like(s11_freqs, dtype=bool)

    for f0 in freq_dict.values():
        important |= (np.abs(s11_freqs - f0) <= df)

    keep = np.zeros_like(s11_freqs, dtype=bool)
    keep[::coarse_step] = True

    keep |= important

    s11_freqs = s11_freqs[keep]
    s11_vals = s11_vals[keep]

    return s11_freqs, s11_vals

def compare_contributions(pkg, c_hat_num, freqs, s11_freqs, threshold = 5e-2):

    freq_mask = (freqs <= np.max(s11_freqs)) & (freqs >= np.min(s11_freqs))
    freqs_red = freqs[freq_mask]

    c_hat_num_red = c_hat_num[freq_mask]
    omegas = freqs_red * 2 * np.pi
    s = pkg['alpha'] + 1j * omegas
    c_hat_ana_red = np.exp(-1j * omegas * pkg['t0']) * \
                    (pkg['A_n'] * s + pkg['B_n'] * pkg['omega_d']) / \
                    (s**2 + pkg['omega_d']**2)

    ratio = np.abs(c_hat_num_red) / np.abs(c_hat_ana_red)
    max_ratio = np.max(ratio)
    idx_max = np.argmax(ratio)

    print(f"[INFO] Maximum |c_hat_num|/|c_hat_ana| = {max_ratio:.3e} occurs at f = {freqs_red[idx_max]:.6g} Hz.")

    mask = (ratio > threshold)

    freqs_exceed = freqs_red[mask]
    ratio_exceed = ratio[mask]

    for f, r in zip(freqs_exceed, ratio_exceed):
        print(f"[INFO] f = {f:.6g} Hz, ratio = {r:.3e}")

    return ratio_exceed, freqs_exceed

def create_plot(spec):
    fig, ax = new_figure(figsize=spec.get("figsize"))

    for curve in spec["curves"]:

        ax.plot(
            curve["x"],
            curve["y"],
            label=curve.get("label"),
            **curve.get("plot_kwargs", {})
        )

    ax.set_xlabel(spec["xlabel"])
    ax.set_ylabel(spec["ylabel"])
    ax.set_title(spec["title"])

    if spec.get("xlim") is not None:
        ax.set_xlim(*spec["xlim"])

    if spec.get("ylim") is not None:
        ax.set_ylim(*spec["ylim"])

    if spec.get("xscale") is not None:
        ax.set_xscale(spec["xscale"])

    if spec.get("yscale") is not None:
        ax.set_yscale(spec["yscale"])

    ax.grid(True)

    if spec.get("legend", False):
        ax.legend()

    save_figure(fig, spec["filename"])