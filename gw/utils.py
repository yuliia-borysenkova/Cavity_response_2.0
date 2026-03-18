import os, json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict
from plotting import new_figure, save_figure

def load_waveform(directory, derivative=0):

    data = np.load(directory)
    t_data, hpl, hcr = data

    for i in range(derivative):
        hpl = np.gradient(hpl, t_data)
        hcr = np.gradient(hcr, t_data)

    return t_data, hpl, hcr

def h_monochromatic(amplitude, tau, omega):
    return amplitude * np.exp(1j * omega * tau)

def plot_waveform(data, labels=(r"$h_+$", r"$h_\times$"), title=None, save_path=None, display=False):

    fig, ax = new_figure()
    
    ax.set_title(title)
    ax.plot(data[0], data[1], label=labels[0])
    ax.plot(data[0], data[2], label=labels[1])
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel("Strain $h$")
    ax.legend()
    
    if save_path is not None:
        save_figure(fig, save_path+".png")
        print(f"[INFO] Saved plot to {save_path}")
        
    if display:    
        plt.show()

def prepare_output_path(data_folder, output_stem, M_abs, q):
    
    os.makedirs(data_folder, exist_ok=True)
    
    if not output_stem:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        M_str = f"{M_abs:.2e}"
        q_str = f"{q:.2f}"
        output_stem = f"GW_M={M_str}_q={q_str}_{timestamp}"
        
    return os.path.join(data_folder, output_stem)

def save_config_to_json(cfg, output_path):
    # Use asdict to convert dataclass to dictionary
    cfg_dict = asdict(cfg)
    
    # Use output filename if provided, otherwise default
    json_file = output_path + "_config.json"
    
    # Save to JSON
    with open(json_file, "w") as f:
        json.dump(cfg_dict, f, indent=4)
    print(f"[INFO] Saved configuration to {json_file}")
