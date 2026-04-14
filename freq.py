import numpy as np
from scipy.special import jn_zeros, jnp_zeros
from scipy.constants import c
import os
import argparse

def roots_TM(n, pmax):
    return jn_zeros(n, pmax)

def roots_TE(n, pmax):
    return jnp_zeros(n, pmax)

def k_qpn(root_np, q, L, R):
    return np.sqrt((q*np.pi/L)**2 + (root_np/R)**2)

def is_zero_mode(mode_family, n, p, q):
    if mode_family == 'TEa' and (p == 0 or n == 0 or q == 0):
        return True
    if mode_family == 'TEb' and (p == 0 or q == 0):
        return True
    if mode_family == 'TMa' and (p == 0 or n == 0):
        return True
    if mode_family == 'TMb' and (p == 0):
        return True
    return False

def scan_modes(
    mode_family="TM",  # "TM" or "TE"
    R=0.04,            # cavity radius [m]
    L=0.24,            # cavity length [m]
    nmax=5,            # azimuthal index range n = 0..nmax
    pmax=3,            # radial index range p = 1..pmax
    qmax=3             # longitudinal index range q = 0..qmax
):
    # Select the correct Bessel root function
    get_roots = roots_TM if mode_family.upper()=="TM" else roots_TE

    rows = []
    for n in range(nmax+1):
        rts = get_roots(n, pmax)  # list of pmax roots (p = 1..pmax)
        for p in range(1, pmax+1):
            for q in range(qmax+1):
                k = k_qpn(rts[p-1], q, L, R)
                omega = c * k                    # angular frequency [rad/s]
                freq = omega / (2*np.pi)         # frequency [Hz]
                
                # store also a/b zero flags
                fam = mode_family.upper()
                is_zero_a = is_zero_mode(fam + "a", n, p, q)
                is_zero_b = is_zero_mode(fam + "b", n, p, q)

                rows.append((fam, n, p, q, omega, freq, is_zero_a, is_zero_b))

    # Sort modes by increasing frequency
    rows.sort(key=lambda r: r[5])
    return rows

def print_table(rows, top=15):
    print(f"{'Mode':<4} {'n':>2} {'p':>2} {'q':>2} "
          f"{'ω [GHz]':>14} {'f [GHz]':>12} "
          f"{'a':>3} {'b':>3}")
    print("-"*75)

    for fam, n, p, q, omega, freq, zero_a, zero_b in rows[:top]:
        a_mark = "0" if zero_a else ""
        b_mark = "0" if zero_b else ""

        print(
            f"{fam:<4} {n:>2} {p:>2} {q:>2} "
            f"{omega/1e9:>14.6f} {freq/1e9:>12.6f} "
            f"{a_mark:>3} {b_mark:>3}"
        )

#OPTIONAL: save functions
def save_modes(filename, rows):
    data = np.array(
        [(fam, n, p, q, freq) for fam, n, p, q, _, freq, *_ in rows],
        dtype=object
    )
    np.save(filename, data)

def save_modes_npz(filename, rows):
    fam = np.array([r[0] for r in rows], dtype="U2")   # "TM" or "TE"
    n   = np.array([r[1] for r in rows], dtype=np.int16)
    p   = np.array([r[2] for r in rows], dtype=np.int16)
    q   = np.array([r[3] for r in rows], dtype=np.int16)
    omega = np.array([r[4] for r in rows], dtype=np.float64)
    freq  = np.array([r[5] for r in rows], dtype=np.float64)
    zero_a = np.array([r[6] for r in rows], dtype=bool)
    zero_b = np.array([r[7] for r in rows], dtype=bool)

    np.savez(filename, fam=fam, n=n, p=p, q=q, omega=omega, freq=freq, zero_a=zero_a, zero_b=zero_b)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan and list cavity modes for a cylindrical cavity.")
    # --- Cavity parameters ---
    parser.add_argument("--R", type=float, default=0.01, help="Cavity radius in [m]")
    parser.add_argument("--L", type=float, default=0.02, help="Cavity length in [m]")
    parser.add_argument("--top", type=int, nargs="?", default=15, help="Number of lowest modes to display")
    args = parser.parse_args()
    R = args.R  # [m]
    L = args.L  # [m]

    tm_rows = scan_modes("TM", R=R, L=L, nmax=2, pmax=3, qmax=3)
    te_rows = scan_modes("TE", R=R, L=L, nmax=2, pmax=3, qmax=3)

    tg_rows = tm_rows + te_rows
    tg_rows.sort(key=lambda r: r[5]) # sort modes again by increasing frequency

    # print("\n=== Lowest TM modes ===")
    # print_table(tm_rows, top=15)
    # print("\n=== Lowest TE modes ===")
    # print_table(te_rows, top=15)
    print("\n=== Lowest general modes ===")
    print_table(tg_rows, top=args.top)

    #OPTIONAL: Save to the file:
    save_dir = f"modes_info"
    os.makedirs(save_dir, exist_ok=True)
    #save_modes(os.path.join(save_dir, "modes.npy"), tg_rows)
    save_modes_npz(os.path.join(save_dir, f"cyl_modes_R{R:.3f}_L{L:.4f}.npz"), tg_rows) 