import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as c_cnst
from scipy import integrate, interpolate, stats

#old compute_num_rhs, not fully vectorised, with lambda interp wrappers for hplus and hcross. Kept for reference.

def compute_num_rhs(xpar, xps, ts, num, V, Efield, B_plusdir, B_crossdir, hplus, hcross, ell):

    xp_num = len(xps)
    
    EE = lambda xp: V * np.mean(stats.norm.pdf(xpar, loc = xp, scale = ell)[:,None] * np.conjugate(Efield), axis = 0) # in V*m
    EEs = np.array([EE(xp) for xp in xps])
            
    E_plus = np.einsum('ij,j->i', np.conjugate(EEs), B_plusdir)
    E_cross = np.einsum('ij,j->i', np.conjugate(EEs), B_crossdir)
                
    EEsDD = (EEs[2:] - 2 * EEs[1:-1] + EEs[:-2]) / (xps[1] - xps[0]) ** 2 # in V/m
    EEsqr = lambda xp: V * np.mean(stats.norm.pdf(xpar, loc = xp, scale = ell)[:,None] ** 2 * np.abs(Efield) ** 2, axis = 0) # in V^2
    EEsqrs = np.array([EEsqr(xp) for xp in xps])
                
    EE_sys_errs = np.abs(EEsDD) * ell ** 2 / 2
    EE_stat_errs = np.sqrt((V * EEsqrs - np.abs(EEs) ** 2) / num)
    EE_errs = np.sqrt(EE_sys_errs ** 2 + EE_stat_errs[1:-1] ** 2)
            
    overlap_sum1d = lambda t: (xps[-1] - xps[0]) * np.mean(hplus(t - xps / c_cnst) * E_plus + hcross(t - xps / c_cnst) * E_cross)
    overlaps_sum1d = np.array([overlap_sum1d(t) for t in ts])
                
    err_sum1d = lambda t: (xps[-1] - xps[0]) / xp_num * np.sqrt(np.vdot(np.abs(EE_errs) ** 2, (hplus(t - xps[1:-1] / c_cnst)[:,None] * B_plusdir + hcross(t - xps[1:-1] / c_cnst)[:,None] * B_crossdir)**2))
    
    errs_sum1d = np.array([err_sum1d(t) for t in ts])

    return overlaps_sum1d.real, errs_sum1d

#vectorised compute_num_rhs, with vectorised interpolation of hplus and hcross. More efficient and cleaner
"""
def compute_num_rhs(xpar, xps, ts, num, V, Efield, B_plusdir, B_crossdir, hplus_arr, hcross_arr, t_data, ell):
    xp_num = len(xps)

    # --- E-field slice averages (unchanged, already vectorised) ---
    EE = V * np.mean(
        stats.norm.pdf(xpar[:, None], loc=xps[None, :], scale=ell)[:, :, None]
        * np.conj(Efield[:, None, :]),
        axis=0,
    )  # shape (Ns, 3)

    E_plus  = EE @ np.conj(B_plusdir)   # shape (Ns,)
    E_cross = EE @ np.conj(B_crossdir)  # shape (Ns,)

    # --- Vectorised retarded-time interpolation ---
    # t_ret[i_t, i_x] = ts[i_t] - xps[i_x] / c
    t_ret = ts[:, None] - xps[None, :] / c_cnst   # shape (Nt, Ns)

    hp  = np.interp(t_ret.ravel(), t_data, hplus_arr,  left=0.0, right=0.0).reshape(t_ret.shape)
    hc  = np.interp(t_ret.ravel(), t_data, hcross_arr, left=0.0, right=0.0).reshape(t_ret.shape)

    # --- Overlap integral ---
    # shape (Nt,)
    overlap = (xps[-1] - xps[0]) / xp_num * np.sum(
        hp * E_plus[None, :] + hc * E_cross[None, :], axis=1
    )

    # --- Error (vectorised) ---
    EEsDD   = (EE[2:] - 2*EE[1:-1] + EE[:-2]) / (xps[1] - xps[0])**2
    EEsqr   = V * np.mean(
        stats.norm.pdf(xpar[:, None], loc=xps[None, :], scale=ell)[:, :, None]**2
        * np.abs(Efield[:, None, :])**2,
        axis=0,
    )
    EE_sys  = np.abs(EEsDD) * ell**2 / 2
    EE_stat = np.sqrt((V * EEsqr - np.abs(EE)**2) / num)
    EE_errs = np.sqrt(EE_sys**2 + EE_stat[1:-1]**2)

    hp_mid = np.interp((ts[:, None] - xps[1:-1][None, :] / c_cnst).ravel(),
                       t_data, hplus_arr, left=0.0, right=0.0).reshape(len(ts), -1)
    hc_mid = np.interp((ts[:, None] - xps[1:-1][None, :] / c_cnst).ravel(),
                       t_data, hcross_arr, left=0.0, right=0.0).reshape(len(ts), -1)

    integrand_err = (hp_mid[:, :, None] * B_plusdir[None, None, :]
                   + hc_mid[:, :, None] * B_crossdir[None, None, :])  # (Nt, Ns-2, 3)
    errs = (xps[-1] - xps[0]) / xp_num * np.sqrt(
        np.einsum('txi,xi->t', integrand_err**2, np.abs(EE_errs)**2)
    )

    return overlap.real, errs
"""
    
def extract_mode(mode_path, threshold=True):

    def conv(fld):
        if fld.endswith('i'):
            return complex(fld[:-1] + 'j')
        else:
            return float(fld)

    cavity_data = np.loadtxt(mode_path, delimiter = ',', comments = r'%', converters = conv, dtype = complex)
    cavity_data[:,:3] *= 1e-3 # rescale coordinates from mm to m

    coords = []
    Efield = []
    for data in cavity_data:
        if not np.isnan(data[3]):
            coords.append(data[:3].real)
            Efield.append(data[3:6])
    coords = np.array(coords) # in m
    Efield = np.array(Efield) # in V/m

    dx = (cavity_data[1,0] - cavity_data[0,0]).real # in m
    assert(np.allclose(cavity_data[1,1:3], cavity_data[0,1:3]))
    dy_ind = np.where(cavity_data[:,1] != cavity_data[0,1])[0][0]
    dy = (cavity_data[dy_ind,1] - cavity_data[0,1]).real # in m
    assert(np.allclose(cavity_data[dy_ind,[0,2]], cavity_data[0,[0,2]]))
    dz_ind = np.where(cavity_data[:,2] != cavity_data[0,2])[0][0]
    dz = (cavity_data[dz_ind,2] - cavity_data[0,2]).real # in m
    assert(np.allclose(cavity_data[dz_ind,:2], cavity_data[0,:2]))
    dV = dx * dy * dz # in m^3
    
    if threshold:
        coords, Efield = filter_E(coords, Efield)
    
    norm = dV * np.linalg.norm(Efield) ** 2 # in V^2*m

    Efield = Efield / np.sqrt(norm)

    V = len(Efield) * dV

    return coords, Efield, norm, V

def filter_E(coords, Efield, threshold=1e-3):
    """
    Discard points where |Efield| < threshold.

    coords: (N,3) array
    Efield: (N,3) array of complex numbers
    threshold: float, minimum magnitude
    """
    E_mag = np.linalg.norm(Efield, axis=1)  # magnitude of each E vector
    mask = E_mag >= threshold               # keep only points above threshold

    return coords[mask], Efield[mask]

def plot_3d(coords, Efield, directory):
    """
    3D scatter plot of points, colored by |Efield| magnitude.
    """
    coords = np.asarray(coords)
    Efield = np.asarray(Efield)

    x = coords[:,0]
    y = coords[:,1]
    z = coords[:,2]

    # Magnitude of the E-field at each point
    E_mag = np.linalg.norm(Efield, axis=1)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter points, color by E-field magnitude
    sc = ax.scatter(x, y, z, c=E_mag, cmap='viridis', s=1, alpha=0.8)
    ax.tick_params(axis='both', labelsize=8)

    ax.set_xlabel("X [m]", fontsize=12)
    ax.set_ylabel("Y [m]", fontsize=12)
    ax.set_zlabel("Z [m]", fontsize=12, labelpad=15)

    # Add colorbar for E-field magnitude
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("|E| [V/m]", fontsize=12)

    # Adjust margins to prevent clipping
    fig.subplots_adjust(left=0.0, right=0.85, bottom=0.05, top=0.95)
    fig.savefig(os.path.join(directory, "thresholded_cavity.png"), dpi=300, bbox_inches='tight')

    plt.close()
