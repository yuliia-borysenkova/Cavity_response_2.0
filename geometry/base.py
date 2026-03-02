from abc import ABC, abstractmethod
import numpy as np
from .integration import overlap_integral as _overlap_integral

class Cavity(ABC):
    def __init__(self, **params):
        self.params = params

    # ---------- coordinate transforms ----------
    @abstractmethod
    def cart_to_native(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def native_to_cart(self, Y: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def cart_vec_to_native(self, V: np.ndarray, coords: np.ndarray) -> np.ndarray:
        pass

    # ---------- domain checks ----------
    @abstractmethod
    def inside(self, coords: np.ndarray) -> bool:
        pass
    
    # ---------- cavity centre ----------
    @abstractmethod
    def center(self):
        """Return the Cartesian center of the cavity."""
        pass

    # ---------- slicing ----------
    @abstractmethod
    def slice_limits(self, k: np.ndarray):
        """Return (x_par_min, x_par_max)."""
        pass

    @abstractmethod
    def slice_integral(self, x_par_vec, integrand, k, e1, e2):
        pass

    def overlap_integral(self, E1, E2, *, method="nquad", complex_value=False, **opts):
        return _overlap_integral(E1, E2, self.geometry, method=method, complex_value=complex_value, **opts)

    @property
    @abstractmethod
    def geometry(self):
        """Object providing bounds(), make_Y(*vars), jacobian(Y) for integration."""
        pass

    # ---------- volume ----------
    @abstractmethod
    def volume(self):
        pass