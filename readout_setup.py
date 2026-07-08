import argparse
import numpy as np
from scipy import integrate

from geometry import CylindricalCavity
from modes import CylindricalMode


def h_TEM(rho, a_coax, b_coax):
    """
    Phi-component of the normalized magnetic vector mode function of the
    fundamental TEM mode of a coaxial line with outer/inner conductor
    radii a_coax, b_coax: h_TEM = 1/sqrt(2 pi ln(a_coax/b_coax)) * 1/rho.
    """
    return 1.0 / (np.sqrt(2 * np.pi * np.log(a_coax / b_coax)) * rho)


def port_coupling(mode, a_coax, b_coax, z=0.0, epsabs=1e-12, epsrel=1e-8):
    """
    Coupling coefficient of `mode` to a coaxial TEM port:

        F_m = int_{S(port)} B_m(r') . h_TEM(r') dS'

    integrated over the annular aperture b_coax <= rho <= a_coax at
    z = z_port, i.e. the cross-section of the coax between its inner and
    outer conductor where it penetrates the cavity wall.
    Requires mode.normalize() to have been called first.
    """
    if mode.norm_B is None:
        raise RuntimeError("Mode not normalized")

    def integrand(rho, phi):
        Bphi = mode.B(np.array([rho, phi, z]))[1]
        return Bphi * h_TEM(rho, a_coax, b_coax) * rho

    val, _ = integrate.dblquad(
        integrand, 0.0, 2 * np.pi, b_coax, a_coax,
        epsabs=epsabs, epsrel=epsrel
    )

    return val


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute the cavity-to-port TEM coupling coefficients "
                     "F_m = int_S B_m(r') . h_TEM(r') dS' for a set of cylindrical "
                     "cavity modes, integrated over the coaxial cable aperture."
    )

    parser.add_argument("--R", type=float, default=0.230, help="Cavity radius [m]")
    parser.add_argument("--L", type=float, default=0.500, help="Cavity length [m]")

    parser.add_argument("--a", type=float, default=2.11e-3, help="Outer conductor radius of the coaxial cable [m]")
    parser.add_argument("--b", type=float, default=0.634e-3, help="Inner conductor radius of the coaxial cable [m]")
    parser.add_argument("--z-port", type=float, default=13.5e-3, help="z position (depth) of the coaxial port aperture [m]")

    parser.add_argument(
        "--modes", nargs="+", default=["TMb_0,1,0", "TMb_0,1,1", "TMb_0,1,2"],
        help="Mode names as 'family_n,p,q', e.g. TMb_0,1,0 (= TM010)"
    )

    parser.add_argument("--output", type=str, default="F_readout.npz")

    return parser.parse_args()


def parse_mode_name(mode_name):
    family, idx_str = mode_name.split("_")
    indices = tuple(int(x) for x in idx_str.split(","))
    return family, indices


def main():
    args = parse_args()

    cavity = CylindricalCavity(R=args.R, L=args.L)

    mode_names = []
    F_values = []

    z_port = args.L - args.z_port  # Convert from depth to z-coordinate
    for mode_name in args.modes:
        family, indices = parse_mode_name(mode_name)

        mode = CylindricalMode(indices=indices, mode_name=family, cavity=cavity, mu_r=1.0, epsilon_r=1.0, sigma_w=0.0)
        mode.normalize()

        F = port_coupling(mode, args.a, args.b, z=z_port)

        mode_names.append(mode_name)
        F_values.append(F)

        #print(f"[INFO] {mode_name}: F = {F:.6f}")

    F_values = np.array(F_values)
    mode_names = np.array(mode_names)
    freq_values = []
    for mode_name in args.modes:
        family, indices = parse_mode_name(mode_name)
        mode = CylindricalMode(indices=indices, mode_name=family, cavity=cavity, mu_r=1.0, epsilon_r=1.0, sigma_w=0.0)
        freq_values.append(mode.omega() / (2 * np.pi))
    freq_values = np.array(freq_values)

    np.savez(
        args.output,
        F_values=F_values,
        freq_values=freq_values,
        R=args.R,
        L=args.L,
        a=args.a,
        b=args.b,
        z_port=z_port,
        mode_names=mode_names
    )

    print(f"[INFO] Results saved to: {args.output}")

    F_dict_str = "F_dict = {" + ", ".join(
        f'"{name}": {F:.6f}' for name, F in zip(mode_names, F_values)
    ) + "}"
    print(F_dict_str)

    print(f"[INFO] Port parameters: a = {args.a * 1e3:.4f} mm, "
          f"b = {args.b * 1e3:.4f} mm, z_port = {z_port * 1e3:.4f} mm")


if __name__ == "__main__":
    main()
