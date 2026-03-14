import argparse, json, time
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from gw.generator import WaveformPipeline, WaveformConfig
from gw.utils import prepare_output_path, save_config_to_json
import numpy as np

def parse_arguments() -> WaveformConfig:
    parser = argparse.ArgumentParser(
         description="Generate gravitational waveforms from compact binary systems "
                    "and optionally add GW memory, clipping, and plotting."
    )

    parser.add_argument("--m-absolute", type=float, default=1e-6,
                        help="Total mass scale of the system in solar masses (M_sun).")
    
    parser.add_argument("--q", type=float, default=1,
                        help="Mass ratio m1/m2 (larger mass over smaller mass).")
    
    parser.add_argument("--spin-1", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        help="Dimensionless spin vector of larger object [x y z].")
    
    parser.add_argument("--spin-2", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        help="Dimensionless spin vector of smaller object [x y z].")

    parser.add_argument("--eccentricity", type=float, default=0.0,
                        help="Eccentricity of the orbit.")

    parser.add_argument("--low_freq", type=float, default=1e8,
                        help="Lower frequency bound of waveform (Hz).")
    
    parser.add_argument("--high_freq", type=float, default=1e11,
                        help="Upper frequency bound of waveform (Hz).")
    
    parser.add_argument("--tc", type=float, default=0.0,
                        help="Time of coalescence in seconds.")
    
    parser.add_argument("--distance", type=float, default=1e-5,
                        help="Distance to the source in parsecs.")
    
    parser.add_argument("--inclination", type=float, default=0.0,
                        help="Inclination angle of the binary orbit (radians).")
    
    parser.add_argument("--polarization-angle", type=float, default=0.0,
                        help="Polarization angle of the gravitational wave (radians).")

    parser.add_argument("--phi0", type=float, default=0.0,
                        help="Initial orbital phase of the binary (radians).")
    
    parser.add_argument("--approximant", type=str, default="IMRPhenomD",
                        help="Waveform approximant to use (e.g., IMRPhenomD, TaylorF2, Newtonian).")
    
    parser.add_argument("--clip", action="store_true",
                        help="Apply waveform clipping to remove small-amplitude tails.")
    
    parser.add_argument("--memory", action="store_true",
                        help="Add gravitational-wave memory effect to waveform.")

    parser.add_argument("--plot", action="store_true",
                        help="Plot waveforms at each stage of generation.")

    parser.add_argument("--display", action="store_true",
                        help="Display waveforms at each stage of generation.")

    parser.add_argument("--data-dir", type=str, default='data',
                        help="Folder where output files will be saved. Created if it does not exist.")
    
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename stem (without extension).")

    parser.add_argument("--density-factor", type=float, default=5.0, help="Density scaling factor applied to the waveform.")

    parser.add_argument("--clip-th1", type=float, default=0.2, help="Primary clipping threshold (relative amplitude).")

    parser.add_argument("--clip-th2", type=float, default=1e-4, help="Secondary clipping threshold for tail removal.")

    args = parser.parse_args()
    
    return WaveformConfig(**vars(args))
    

def main():
    cfg = parse_arguments()
    output_path = prepare_output_path(cfg.data_dir, cfg.output, cfg.m_absolute, cfg.q)
    save_config_to_json(cfg, output_path)

    start = time.time()
    pipeline = WaveformPipeline(cfg, output_path)
    data = pipeline.run()
    print(f"[INFO] Computed in {time.time()-start: .2f} s.")
    
    np.save(output_path + ".npy", data)
    print(f"[INFO] Saved waveform data to {output_path}.npy")

if __name__ == "__main__":
    main()
