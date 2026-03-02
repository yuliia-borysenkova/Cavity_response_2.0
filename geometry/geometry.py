import numpy as np
from abc import ABC, abstractmethod

class Geometry(ABC):
    def __init__(self, bounds):
        self._bounds = bounds
        
    @property
    def bounds(self):
        """Return list of (min, max) for each integration variable."""
        return self._bounds

    def make_Y(self, *params):
        """Convert integration variables to coordinates Y for E-field evaluation."""
        return np.asarray(params, float)

    @abstractmethod
    def jacobian(self, Y):
        """Return the Jacobian factor for integration at coordinates Y."""
        ...

class CylindricalGeometry(Geometry):
    def __init__(self, R, L, rho_min=1e-30):
        super().__init__([
            (rho_min, R),       # rho
            (0.0, 2*np.pi),     # phi
            (0.0, L),           # z
        ])

    def jacobian(self, Y):
        return Y[0]
    
class SphericalGeometry(Geometry):
    def __init__(self, R, r_min=1e-30, theta_fix=1e-3):
        super().__init__([
            (r_min, R),         # r
            (theta_fix, np.pi-theta_fix),       # theta
            (0.0, 2*np.pi),     # phi
        ])

    def jacobian(self, Y):
        r, theta, _ = Y
        return r**2 * np.sin(theta)
    
class RectangularGeometry(Geometry):
    def __init__(self, Lx, Ly, Lz):
        super().__init__([
            (0.0, Lx),          # x
            (0.0, Ly),          # y
            (0.0, Lz),          # z
        ])

    def jacobian(self, Y):
        return 1.0