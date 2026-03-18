import numpy as np
from scipy.constants import c as c_cnst
from scipy import integrate, interpolate, stats

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

def extract_mode(mode_path):

    with open(mode_path, 'r') as f:
        lines = [line for line in f if not line.startswith('%')]
    
    data_str = [line.strip().split(',') for line in lines]
    
    # Convert to a NumPy array of strings
    data_str = np.array(data_str)
    
    # Coordinates (first 3 columns) → float
    coords = data_str[:, 0:3].astype(float) / 1000   # mm → m
    
    # E-field (columns 3-5) → complex
    # Replace 'i' with 'j' and convert to complex
    Efield_str = np.char.replace(data_str[:, 3:6], 'i', 'j')
    Efield = Efield_str.astype(np.complex128)
    
    # Replace any NaNs with 0
    Efield = np.nan_to_num(Efield)

    return coords, Efield