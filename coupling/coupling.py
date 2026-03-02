import numpy as np
import os
from tqdm import tqdm
from gw.utils import h_monochromatic
from rhs.utils import compute_k_pol, make_jeff

class CouplingStrength:
    def __init__(self, cavity, mode, theta_vals, phi_vals: int = 0, B=(0.0, 0.0, 1.0), pol: str = "cross", nproc: int = 1):
        self.cavity = cavity
        self.mode = mode
        self.B = np.asarray(B, dtype=float)
        self.pol = str(pol)
        self.nproc = int(nproc)
        self.theta_vals = theta_vals
        self.phi_vals = phi_vals

        self.omega = mode.omega()
        self.hplus = lambda tau: h_monochromatic(amplitude=1.0, tau=tau, omega=self.omega)
        self.hcross = lambda tau: h_monochromatic(amplitude=1.0, tau=tau, omega=self.omega)
        
    def compute_coupling(self, t, k=None):
        directions = []
        for phi in self.phi_vals:
            for theta in self.theta_vals:
                k, e1, e2 = compute_k_pol(theta, phi)
                directions.append((k, e1, e2))
        
        eta_k = []
        mode_numbers = ''.join(map(str, self.mode.indices))
        desc = f"[INFO] Computing coupling for mode {self.mode.mode_name}_{mode_numbers} with {self.pol} polarization"
        for k, e1, e2 in tqdm(directions, desc=desc, unit="k"):
            jeff = make_jeff(B=self.B, cavity=self.cavity, hplus=self.hplus, hcross=self.hcross, k=k, e1=e1, e2=e2)[self.pol]
            V = self.cavity.volume()
            E1 = lambda Y: self.mode.E(Y)
            E2 = lambda Y: jeff(Y, t)
            # coupling = self.cavity.overlap_integral(E1, E2, method="vegas", neval=50_000, nitn_warmup=5, nitn_main=10) / np.sqrt(V)
            coupling = self.cavity.overlap_integral(E1, E2, method="nquad", epsabs=1e-8, epsrel=1e-6, limit=80, complex_value=True) / np.sqrt(V)
            eta_k.append(coupling)

        eta_k = np.array(eta_k)
        eta_reshaped = eta_k.reshape(len(self.phi_vals), len(self.theta_vals))

        return eta_reshaped