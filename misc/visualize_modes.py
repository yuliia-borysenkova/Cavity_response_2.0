"""
visualize_modes.py — Visualize electromagnetic cavity eigenmodes.

Produces four figures per run:
  1. Volumetric scatter: |E|² field intensity throughout the cavity volume.
  2–4. Three orthogonal cross-sections (XY, XZ, YZ) with |E|² colour-map and
       Re(E) field vectors (snapshot at t = 0).

Usage example:
  python visualize_modes.py --geometry cylindrical --mode-fam TM --mode-par b \
      --mode-ind 0,1,0 --R 0.05 --L 0.05
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("MacOSX")   # interactive window on macOS; change to "TkAgg" if this fails
import matplotlib.pyplot as plt

from geometry import CylindricalCavity, SphericalCavity, RectangularCavity
from modes import RectangularMode, CylindricalMode, SphericalMode
from plotting.theme import apply_style, save_figure


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Visualize cavity eigenmodes in 3D")

    p.add_argument("--geometry", choices=["rectangular", "cylindrical", "spherical"],
                   default="cylindrical")
    p.add_argument("--mode-fam", choices=["TE", "TM"], default="TM",
                   help="Mode family")
    p.add_argument("--mode-par", choices=["a", "b", "None"], default="b",
                   help="Azimuthal parity (a = sin nφ, b = cos nφ); ignored for rectangular")
    p.add_argument("--mode-ind", default="0,1,1",
                   help="Mode indices n,p,q (cylindrical/spherical) or m,n,p (rectangular)")

    # Cavity dimensions
    p.add_argument("--a", type=float, default=0.1, help="Rectangular: x-dimension [m]")
    p.add_argument("--b", type=float, default=0.1, help="Rectangular: y-dimension [m]")
    p.add_argument("--c", type=float, default=0.1, help="Rectangular: z-dimension [m]")
    p.add_argument("--R", type=float, default=0.05, help="Cavity radius [m]")
    p.add_argument("--L", type=float, default=0.05, help="Cylindrical cavity length [m]")

    # Visualisation resolution
    p.add_argument("--grid-size",  type=int, default=30,
                   help="Points per axis for the volumetric scatter")
    p.add_argument("--slice-size", type=int, default=50,
                   help="Points per axis for the cross-section surfaces")

    p.add_argument("--output",  type=str, default=None,
                   help="Override default output directory")
    p.add_argument("--no-show", action="store_true",
                   help="Skip interactive display")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def native_to_cartesian(V_native, Y_native, geometry):
    """
    Rotate a vector field from native (cavity) basis to Cartesian.

    Parameters
    ----------
    V_native : (3, N) or (3,) array
        Vector components in native basis [Vρ, Vφ, Vz] or [Vr, Vθ, Vφ].
    Y_native : (3, N) or (3,) array
        Native coordinates matching V_native.
    geometry : str
    """
    if geometry == "rectangular":
        return V_native

    if geometry == "cylindrical":
        phi = Y_native[1]
        Vr, Vp, Vz = V_native
        return np.array([
            Vr * np.cos(phi) - Vp * np.sin(phi),
            Vr * np.sin(phi) + Vp * np.cos(phi),
            Vz,
        ])

    if geometry == "spherical":
        r, th, ph = Y_native
        Vr, Vth, Vph = V_native
        sth, cth = np.sin(th), np.cos(th)
        sph, cph = np.sin(ph), np.cos(ph)
        return np.array([
            Vr * sth * cph + Vth * cth * cph - Vph * sph,
            Vr * sth * sph + Vth * cth * sph + Vph * cph,
            Vr * cth         - Vth * sth,
        ])

    raise ValueError(f"Unknown geometry '{geometry}'")


# ---------------------------------------------------------------------------
# Cavity outline drawing
# ---------------------------------------------------------------------------

def draw_cavity_outline(ax, cavity, geometry, color="gray", alpha=0.3):
    """Draw a dashed wireframe outline of the cavity boundary."""
    kw = dict(color=color, alpha=alpha, linestyle="--")

    if geometry == "rectangular":
        a, b, c = cavity.a, cavity.b, cavity.c
        for x in (0, a):
            for y in (0, b):
                ax.plot([x, x], [y, y], [0, c], **kw)
        for x in (0, a):
            for z in (0, c):
                ax.plot([x, x], [0, b], [z, z], **kw)
        for y in (0, b):
            for z in (0, c):
                ax.plot([0, a], [y, y], [z, z], **kw)

    elif geometry == "cylindrical":
        R, L = cavity.R, cavity.L
        theta = np.linspace(0, 2 * np.pi, 100)
        cx, cy = R * np.cos(theta), R * np.sin(theta)
        ax.plot(cx, cy, 0, color=color, alpha=alpha)
        ax.plot(cx, cy, L, color=color, alpha=alpha)
        for phi in np.linspace(0, 2 * np.pi, 5)[:-1]:
            ax.plot([R * np.cos(phi)] * 2, [R * np.sin(phi)] * 2, [0, L],
                    color=color, alpha=alpha)

    elif geometry == "spherical":
        R = cavity.R
        u = np.linspace(0, 2 * np.pi, 100)
        ax.plot(R * np.cos(u), R * np.sin(u), np.zeros_like(u), color=color, alpha=alpha)
        ax.plot(R * np.cos(u), np.zeros_like(u), R * np.sin(u), color=color, alpha=alpha)
        ax.plot(np.zeros_like(u), R * np.cos(u), R * np.sin(u), color=color, alpha=alpha)


# ---------------------------------------------------------------------------
# Field evaluation helpers
# ---------------------------------------------------------------------------

def eval_field_on_grid(pts_cart, cavity, mode, geometry):
    """
    Evaluate the electric field at a list of Cartesian points.

    Returns
    -------
    intensity : (N,) float   — |E|² at each point (zero outside cavity)
    e_cart    : (N, 3) float — Re(E) in Cartesian coordinates (zero outside)
    inside    : (N,) bool
    """
    N = len(pts_cart)
    intensity = np.zeros(N)
    e_cart    = np.zeros((N, 3))
    inside    = np.zeros(N, dtype=bool)

    for i, pt in enumerate(pts_cart):
        pt_native = cavity.cart_to_native(pt)
        if not cavity.inside(pt_native):
            continue
        e_native = mode.E(pt_native)
        e_c      = native_to_cartesian(e_native, pt_native, geometry)
        inside[i]    = True
        intensity[i] = float(np.sum(np.abs(e_native) ** 2))
        e_cart[i]    = np.real(e_c)

    return intensity, e_cart, inside


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _add_colorbar(fig, mappable, ax, label):
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(label)
    return cbar


def plot_volumetric(pts_inside, intensity, mode_label, cavity, geometry):
    """Scatter plot of |E|² inside the cavity volume."""
    fig = plt.figure(figsize=(8, 7.5))
    ax  = fig.add_subplot(111, projection="3d")
    draw_cavity_outline(ax, cavity, geometry)

    if len(pts_inside):
        norm_i = intensity / intensity.max()
        sc = ax.scatter(
            pts_inside[:, 0], pts_inside[:, 1], pts_inside[:, 2],
            c=intensity, s=80 * norm_i + 5,
            cmap="plasma", alpha=0.5, edgecolors="none",
        )
        _add_colorbar(fig, sc, ax, r"$|\mathbf{E}|^2$ [arb. units]")

    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(f"{mode_label}  —  volumetric $|E|^2$")
    return fig


def plot_slice(X, Y, Z, intensity, e_cart, inside, norm, cavity, geometry,
               mode_label, plane_label, quiver_step=4):
    """
    2-D cross-section: colour-mapped |E|² surface with Re(E) quiver overlay.

    Parameters
    ----------
    plane_label : str   e.g. "XY  (z = L/2)"
    quiver_step : int   subsample factor for quiver arrows
    """
    fig = plt.figure(figsize=(8, 7))
    ax  = fig.add_subplot(111, projection="3d")
    draw_cavity_outline(ax, cavity, geometry)

    # Surface colour
    rgba = plt.cm.plasma(norm(intensity))
    rgba[~inside, 3] = 0.0   # transparent outside cavity
    rgba[ inside, 3] = 0.65
    ax.plot_surface(X, Y, Z, facecolors=rgba,
                    rstride=1, cstride=1, shade=False, antialiased=True)

    # Quiver overlay (Re(E) field vectors, snapshot at t = 0)
    s = quiver_step
    xs_q = X[::s, ::s].ravel();  ys_q = Y[::s, ::s].ravel();  zs_q = Z[::s, ::s].ravel()
    vx   = e_cart[::s, ::s, 0].ravel()
    vy   = e_cart[::s, ::s, 1].ravel()
    vz   = e_cart[::s, ::s, 2].ravel()
    ins_q = inside[::s, ::s].ravel()

    mag = np.sqrt(vx**2 + vy**2 + vz**2)
    keep = ins_q & (mag > 1e-10)
    if keep.any():
        scale = 0.06 * max(np.ptp(X), np.ptp(Y), np.ptp(Z))
        ax.quiver(xs_q[keep], ys_q[keep], zs_q[keep],
                  vx[keep] / mag[keep] * scale,
                  vy[keep] / mag[keep] * scale,
                  vz[keep] / mag[keep] * scale,
                  color="black", pivot="middle", alpha=0.9, linewidth=1.1)

    # Colorbar — plot_surface with facecolors has no mappable, so make one
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
    sm.set_array([])
    _add_colorbar(fig, sm, ax, r"$|\mathbf{E}|^2$ [arb. units]")

    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(f"{mode_label}  —  {plane_label} slice\n"
                 r"colour: $|\mathbf{E}|^2$,  arrows: $\mathrm{Re}(\mathbf{E})$")
    ax.view_init(elev=25, azim=45)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # 1. Build mode name -------------------------------------------------------
    if args.geometry == "rectangular" or args.mode_par in ("None", None):
        mode_name = args.mode_fam
    else:
        mode_name = args.mode_fam + args.mode_par

    mode_ind = [int(x) for x in args.mode_ind.split(",")]
    # Human-readable label, e.g. "TM₀₁₀ (cylindrical)"
    subscript = "".join(str(i) for i in mode_ind)
    mode_label = f"{mode_name}$_{{{subscript}}}$ ({args.geometry})"

    # 2. Instantiate cavity and mode -------------------------------------------
    if args.geometry == "rectangular":
        cavity = RectangularCavity(a=args.a, b=args.b, c=args.c)
        mode_class = RectangularMode
        bounds = [(0.0, args.a), (0.0, args.b), (0.0, args.c)]
    elif args.geometry == "cylindrical":
        cavity = CylindricalCavity(R=args.R, L=args.L)
        mode_class = CylindricalMode
        bounds = [(-args.R, args.R), (-args.R, args.R), (0.0, args.L)]
    else:  # spherical
        cavity = SphericalCavity(R=args.R)
        mode_class = SphericalMode
        bounds = [(-args.R, args.R), (-args.R, args.R), (-args.R, args.R)]

    center = cavity.center()
    print(f"Cavity: {args.geometry}  |  Mode: {mode_name} {mode_ind}")

    mode = mode_class(indices=mode_ind, mode_name=mode_name, cavity=cavity)
    mode.normalize()
    print(f"Mode normalised  (norm = {mode.norm:.4e})")

    # Apply custom style if available
    try:
        apply_style()
    except Exception:
        pass

    # 3. Volumetric grid -------------------------------------------------------
    Nv = args.grid_size
    xs = np.linspace(*bounds[0], Nv)
    ys = np.linspace(*bounds[1], Nv)
    zs = np.linspace(*bounds[2], Nv)
    pts_vol = np.column_stack([g.ravel() for g in np.meshgrid(xs, ys, zs, indexing="ij")])

    vol_int, vol_e, vol_in = eval_field_on_grid(pts_vol, cavity, mode, args.geometry)
    pts_inside = pts_vol[vol_in]
    int_inside = vol_int[vol_in]

    # 4. Orthogonal cross-sections --------------------------------------------
    Ns = args.slice_size
    xs_s = np.linspace(*bounds[0], Ns)
    ys_s = np.linspace(*bounds[1], Ns)
    zs_s = np.linspace(*bounds[2], Ns)

    def make_slice(X2d, Y2d, Z2d):
        """Evaluate field on a 2-D surface defined by three (Ns,Ns) arrays."""
        pts = np.column_stack([X2d.ravel(), Y2d.ravel(), Z2d.ravel()])
        I, E, ins = eval_field_on_grid(pts, cavity, mode, args.geometry)
        return (I.reshape(Ns, Ns),
                E.reshape(Ns, Ns, 3),
                ins.reshape(Ns, Ns))

    X_xy, Y_xy = np.meshgrid(xs_s, ys_s, indexing="ij")
    Z_xy       = np.full_like(X_xy, center[2])
    int_xy, e_xy, in_xy = make_slice(X_xy, Y_xy, Z_xy)

    X_xz, Z_xz = np.meshgrid(xs_s, zs_s, indexing="ij")
    Y_xz       = np.full_like(X_xz, center[1])
    int_xz, e_xz, in_xz = make_slice(X_xz, Y_xz, Z_xz)

    Y_yz, Z_yz = np.meshgrid(ys_s, zs_s, indexing="ij")
    X_yz       = np.full_like(Y_yz, center[0])
    int_yz, e_yz, in_yz = make_slice(X_yz, Y_yz, Z_yz)

    # Shared normalisation across all slices
    vmax = max(int_xy.max(), int_xz.max(), int_yz.max(), 1e-30)
    norm = plt.Normalize(vmin=0, vmax=vmax)

    # 5. Build figures ---------------------------------------------------------
    step = max(1, Ns // 12)   # quiver subsampling

    fig_vol = plot_volumetric(pts_inside, int_inside, mode_label, cavity, args.geometry)

    fig_xy = plot_slice(X_xy, Y_xy, Z_xy, int_xy, e_xy, in_xy, norm,
                        cavity, args.geometry, mode_label,
                        f"XY  (z = {center[2]:.3g} m)", step)
    fig_xz = plot_slice(X_xz, Y_xz, Z_xz, int_xz, e_xz, in_xz, norm,
                        cavity, args.geometry, mode_label,
                        f"XZ  (y = {center[1]:.3g} m)", step)
    fig_yz = plot_slice(X_yz, Y_yz, Z_yz, int_yz, e_yz, in_yz, norm,
                        cavity, args.geometry, mode_label,
                        f"YZ  (x = {center[0]:.3g} m)", step)

    # 6. Save + Show ----------------------------------------------------------
    tag     = f"{mode_name}_{'_'.join(map(str, mode_ind))}"
    out_dir = args.output or os.path.join("results", args.geometry, tag)
    os.makedirs(out_dir, exist_ok=True)

    for fig, name in [
        (fig_vol, "volumetric"),
        (fig_xy,  "slice_xy"),
        (fig_xz,  "slice_xz"),
        (fig_yz,  "slice_yz"),
    ]:
        path = os.path.join(out_dir, f"{args.geometry}_{tag}_{name}.png")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")

    print(f"Saved figures to {out_dir}/")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
