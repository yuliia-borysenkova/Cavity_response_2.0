import numpy as np
from .base import CavityMode
from scipy.constants import c as c_cnst
from scipy.constants import mu_0, epsilon_0

class RectangularMode(CavityMode):
    """
    Rectangular cavity modes:
    TM, TE
    Indices: m,n,p
    """
    def __init__(self, indices, mode_name, epsilon_r, mu_r, sigma_w, cavity):
        self.m, self.n, self.p = indices
        self.epsilon_r = epsilon_r
        self.mu_r = mu_r
        self.sigma_w = sigma_w
        super().__init__(indices, mode_name, epsilon_r, mu_r, sigma_w, cavity)
        self.k = self.k_calc()
        
    # ---------------- mode index validation ----------------
    def _is_zero_mode(self):
        return False


    def _validate(self):
        m, n, p = self.m, self.n, self.p

        if self.mode_name not in ("TE", "TM"):
            raise ValueError(f"Invalid mode_name {self.mode_name} for rectangular cavity")

        if any(i < 0 for i in (m, n, p)):
            raise ValueError("Mode indices must be non-negative")
            
        if self.mode_name == "TM":
            if m == 0 or n == 0:
                    raise ValueError("Rectangular TM modes require m >= 1 and n >= 1")

        if self.mode_name == "TE":
            if p == 0:
                raise ValueError("Rectangular TE modes require p >= 1")
            

    # ---------------- wavevector ----------------
    def k_calc(self):
        a, b, c = self.cavity.a, self.cavity.b, self.cavity.c

        return np.sqrt((self.m*np.pi/a)**2 + (self.n*np.pi/b)**2 + (self.p*np.pi/c)**2)

    def omega(self):
        return c_cnst * self.k
    
    # ---------------- quality factor ----------------
    def calculate_Q(self):
        a, b, c = self.cavity.a, self.cavity.b, self.cavity.c
        m, n, p = self.m, self.n, self.p

        mu = mu_0 * self.mu_r
        epsilon = epsilon_0 * self.epsilon_r
        eta = np.sqrt(mu/epsilon)
        Rs = np.sqrt(mu*self.omega()/(2*self.sigma_w))

        kx = m * np.pi / a
        ky = n * np.pi / b
        kz = p * np.pi / c

        kxy2 = kx**2 + ky**2


        if self.mode_name == "TE":
            if m == 0:
                 Q = eta * a * b * c * self.k**3 / (2 * Rs * (b * c * self.k**2 + 2 * a * c * ky**2 + 2 * a * b * kz**2))
            elif n == 0:
                 Q = eta * a * b * c * self.k**3 / (2 * Rs * (a * c * self.k**2 + 2 * b * c * kx**2 + 2 * a * b * kz**2))
            else:
                 Q = eta * a * b * c * kxy2 * self.k**3 / (4 * Rs * (b * c * (kxy2**2 + ky**2 * kz**2) + a * c * (kxy2**2 + kx**2 * kz**2) + a * b * kxy2 * kz**2))

        elif self.mode_name == "TM":
            if p == 0:
                Q =  eta * a * b * c * self.k**3 / (2 * Rs * (a * b * self.k**2 + 2 * b * c * kx**2 + 2 * a * c * ky**2))
            else:
                Q = eta * a * b * c * kxy2 * self.k / (4 * Rs * (kx**2 * b * (a + c) + ky**2 * a * (b + c)))

        return Q

    # ---------------- prenormalized E field ----------------
    def E_prenorm(self, Y):

        if self._is_zero_mode():
            return np.zeros(3, dtype=complex)
        
        k = self.k
        a, b, c = self.cavity.a, self.cavity.b, self.cavity.c
        m, n, p = self.m, self.n, self.p
        
        if self.mode_name == 'TE':

            Ex = - 1j / (k**2 - (p * np.pi / c)**2 ) * (n * np.pi / b) * np.cos(m * np.pi / a * Y[0]) * np.sin(n * np.pi / b * Y[1]) * np.sin(p * np.pi / c * Y[2])
            Ey = 1j / (k**2 - (p * np.pi / c)**2 )  * (m * np.pi / a) * np.sin(m * np.pi / a * Y[0]) * np.cos(n * np.pi / b * Y[1]) * np.sin(p * np.pi / c * Y[2])
            Ez =  0.0

        elif self.mode_name == 'TM':

            Ex = -1 / (k**2 - (p * np.pi / c)**2 ) * (m * np.pi / a) * (p * np.pi / c)  * np.cos(m * np.pi / a * Y[0]) * np.sin(n * np.pi / b * Y[1]) * np.sin(p * np.pi / c * Y[2])
            Ey = 1 / (k**2 - (p * np.pi / c)**2 ) * (n * np.pi / b) * (p * np.pi / c)  * np.sin(m * np.pi / a * Y[0]) * np.cos(n * np.pi / b * Y[1]) * np.sin(p * np.pi / c * Y[2])
            Ez = np.sin(m * np.pi / a * Y[0]) * np.sin(n * np.pi / b * Y[1]) * np.cos(p * np.pi / c * Y[2])

        return np.array([Ex, Ey, Ez])
    
    def B_prenorm(self, Y):

        if self._is_zero_mode():
            return np.zeros(3, dtype=complex)
        
        k = self.k
        a, b, c = self.cavity.a, self.cavity.b, self.cavity.c
        m, n, p = self.m, self.n, self.p
        
        if self.mode_name == 'TE':

            Bx = - 1 / (k**2 - (p * np.pi / c)**2 ) * (m * np.pi / a) * (n * np.pi / b) * np.sin(m * np.pi / a * Y[0]) * np.cos(n * np.pi / b * Y[1]) * np.cos(p * np.pi / c * Y[2])
            By = 1 / (k**2 - (p * np.pi / c)**2 ) * (m * np.pi / a) * (p * np.pi / c) * np.cos(m * np.pi / a * Y[0]) * np.sin(n * np.pi / b * Y[1]) * np.sin(p * np.pi / c * Y[2])
            Bz =  np.cos(m * np.pi / a * Y[0]) * np.cos(n * np.pi / b * Y[1]) * np.sin(p * np.pi / c * Y[2])

        elif self.mode_name == 'TM':

            Bx = - 1j / (k**2 - (p * np.pi / c)**2 ) * (n * np.pi / b) * np.sin(m * np.pi / a * Y[0]) * np.cos(n * np.pi / b * Y[1]) * np.cos(p * np.pi / c * Y[2])
            By = 1j / (k**2 - (p * np.pi / c)**2 )  * (m * np.pi / a) * np.cos(m * np.pi / a * Y[0]) * np.sin(n * np.pi / b * Y[1]) * np.cos(p * np.pi / c * Y[2])
            Bz = 0.0

        return np.array([Bx, By, Bz])

    # ---------------- normalized fields ----------------
    def E(self, Y):
        if self.norm_E is None:
            raise RuntimeError("Mode not normalized")
        return self.E_prenorm(Y)/np.sqrt(self.norm_E)
        
    def B(self, Y):
        if self.norm_B is None:
            raise RuntimeError("Mode not normalized")
        
        return self.B_prenorm(Y)/np.sqrt(self.norm_B)
    
# ---------------- normalization ----------------
    def normalize(self):
        if self._is_zero_mode():
            self.norm_E = 1
            self.norm_B = 1
        else:    
            def E1(Y): return self.E_prenorm(Y)
            self.norm_E = self.cavity.overlap_integral(E1, E1)
            def E2(Y): return self.B_prenorm(Y)
            self.norm_B = self.cavity.overlap_integral(E2, E2)

            if self.norm_E == 0:
                self.norm_E = 1
            if self.norm_B == 0:
                self.norm_B = 1

    