import numpy as np
from .base import Cavity
from scipy import integrate
from geometry.geometry import SphericalGeometry

class SphericalCavity(Cavity):
    def __init__(self, R: float, **params):
        super().__init__(**params)
        self.R = float(R)
        self._geometry = SphericalGeometry(R=self.R)

    @property
    def geometry(self) -> SphericalGeometry:
        return self._geometry

    # ---------------- coordinate conversion ----------------
    def cart_to_native(self, X):
        x, y, z = X
        r = np.sqrt(x*x + y*y + z*z)
        if r < 1e-12:
            return np.array([r, 0.0, 0.0])
        theta = np.arccos(z/r)
        phi = np.arctan2(y, x)
        if phi < 0:
            phi += 2*np.pi
        return np.array([r, theta, phi])
    
    def native_to_cart(self, Y):
        r, theta, phi = Y
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.array([x, y, z])

    def cart_vec_to_native(self, V, coords):
        r, theta, phi = coords
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_th, s_th = np.cos(theta), np.sin(theta)
        Vx, Vy, Vz = V
        
        Jr = s_th*c_phi*Vx + s_th*s_phi*Vy + c_th*Vz
        Jtheta = c_th*c_phi*Vx + c_th*s_phi*Vy - s_th*Vz
        Jphi = -s_phi*Vx + c_phi*Vy
        
        return np.array([Jr, Jtheta, Jphi])
        
    # ---------------- center ----------------
    def center(self):
        return np.array([0.0, 0.0, 0.0])

    # ---------------- domain ----------------
    def inside(self, Y):
        r, _, _ = Y
        return r <= self.R

    # ---------------- slice ----------------
    def slice_limits(self, k):
        # For spheres, x_parallel runs from -R to +R along k
        return -self.R, self.R

    def slice_integral(self, x_par_vec, integrand, k, e1, e2,
                       epsabs=1e-8, epsrel=1e-4, limit=200):
        lim = self.R

        def wrapped(s, t):
            X = x_par_vec + s*e1 + t*e2
            
            Y = self.cart_to_native(X)
            r, theta, phi = Y
            
            if not self.inside(Y):
                return 0.0
            return integrand(r, theta, phi)

        opts = [{'limit': limit, 'epsabs': epsabs, 'epsrel': epsrel}]*2
        result, _ = integrate.nquad(wrapped, [[-lim, lim], [-lim, lim]], opts=opts)
        return result

    # ---------------- volume ----------------
    def volume(self):
        return 4/3 * np.pi * self.R**3
