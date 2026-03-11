import os, json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict

def load_waveform(directory, derivative=0):

    data = np.load(directory)
    t_data, hpl, hcr = data

    for i in range(derivative):
        hpl = np.gradient(hpl, t_data)
        hcr = np.gradient(hpl, t_data)

    return t_data, hpl, hcr

def h_monochromatic(amplitude, tau, omega):
    return amplitude * np.exp(1j * omega * tau)

def plot_waveform(data, labels=("h+", "hx"), title=None, save_path=None):
    plt.figure(figsize=(10,5))
    plt.title(title)
    plt.plot(data[0], data[1], label=labels[0])
    plt.plot(data[0], data[2], label=labels[1])
    plt.xlabel("Time [s]")
    plt.ylabel("Strain")
    plt.legend()
    plt.grid()
    
    if save_path is not None:
        plt.savefig(save_path+".png", dpi=200)
        print(f"[INFO] Saved plot to {save_path}")
        
    #plt.show()

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
