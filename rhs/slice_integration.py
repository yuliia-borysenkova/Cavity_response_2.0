# simulation.py
import os
import numpy as np
import matplotlib.pyplot as plt
from rhs.utils import decompose_B, save_plot, compute_k_pol
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

def compute_area_integral(cavity, mode, Bvec, x_vec, k, e1, e2):
    """
    Compute area integral at a single parallel position.
    """
    def integrand(*coords):
        Y = np.array(coords)

        # Convert Cartesian B into cavity-native components
        B_native = cavity.cart_vec_to_native(Bvec, Y)

        return np.vdot(mode.E(Y), B_native)

    return cavity.slice_integral(x_vec, integrand, k, e1, e2)

# Move to slice_integration
def compute_slice_integrals(cavity, mode, Bvec, k, e1, e2, Ns=100, nproc=1):
    """
    Compute slice integrals over all parallel positions in a cavity.
    Returns: x_par_vals, area_vals
    """
    # Plane vectors
    if hasattr(cavity, "slice_limits"):
        x_min, x_max = cavity.slice_limits(k)
    else:
        raise ValueError("Cavity must implement slice_limits()")
    
    x_par_vals = np.linspace(x_min, x_max, Ns)
    
    x0 = cavity.center()
    
    x_par_vecs = np.array([x0 + x*k for x in x_par_vals])

    # Plane vectors
    func = partial(compute_area_integral, cavity, mode, Bvec, k=k, e1=e1, e2=e2)
    
    with Pool(nproc) as pool:
        area_vals = list(tqdm(pool.imap(func, x_par_vecs), total=len(x_par_vecs)))
    
    return x_par_vals, np.array(area_vals)
    
class SliceIntegration:
    """
    Handles slice integrals of cavity modes for plus and cross polarizations.
    Works with any geometry and mode subclass.
    """
    def __init__(self, cavity, mode, theta, phi, B, Ns=100, nproc=1, save_dir="results"):
        self.cavity = cavity
        self.mode = mode
        self.theta = theta
        self.phi = phi
        self.B = B
        self.Ns = Ns
        self.nproc = nproc
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Decompose B field
        self.k, self.e1, self.e2 = compute_k_pol(self.theta, self.phi)
        self.B_plus, self.B_cross = decompose_B(self.B, self.k, self.e1, self.e2)

    def run(self):
        """
        Compute plus and cross slice integrals, plot and save results
        """
        outputs = {}
        for pol, Bvec in [("plus", self.B_plus), ("cross", self.B_cross)]:
            print(f"[INFO] Computing {pol} polarization integrals...")
            x_par_vals, area_vals = compute_slice_integrals(
                self.cavity, self.mode, Bvec, self.k, self.e1, self.e2,
                Ns=self.Ns, nproc=self.nproc
            )
            print("[INFO] Slice integration complete.")

            # Plot
            plt.figure(figsize=(10,5))
            plt.plot(x_par_vals, area_vals)
            plt.xlabel("X parallel [m]")
            plt.ylabel(f"E · B_{pol}")
            plt.title(f"Cavity mode {self.mode.mode_name} {self.mode.indices}")
            plt.grid(True, alpha=0.3)
            save_plot(self.save_dir, f"slice_integrals_{pol}.png")

            # Save data
            file_name = os.path.join(self.save_dir, f"slice_integrals_{pol}.npz")
            np.savez(file_name, x_par=x_par_vals, area=area_vals, k=self.k)
            print("[INFO] Saved results to ", file_name)
            
            outputs[pol] = area_vals
        
        return outputs
