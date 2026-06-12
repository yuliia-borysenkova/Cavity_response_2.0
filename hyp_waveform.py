import argparse, time, json
from gw.utils import prepare_output_path, save_config_to_json
import numpy as np
from gw.models import get_hyp_waveform
from gw.processing import rotate_polarization
from gw.utils import plot_waveform, prepare_output_path, save_config_to_json

AU_TO_PC = 1 / 206265  # 1 AU in parsecs

# Need to add safety procedures and a config file generation

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-total", type=float, default=1e-6,
                        help="Total mass of the system in solar masses (M_sun).")
    
    parser.add_argument("--q", type=float, default=1,
                        help="Mass ratio m1/m2 (larger mass over smaller mass).")

    parser.add_argument("--initial-eccentricity", type=float, default=1.1,
                        help="Initial eccentricity of the encounter.")
    
    parser.add_argument("--b", type=float, default=70.0,
                        help="Impact parameter in GM/c^2.")
    
    
    parser.add_argument("--time-initial", type=float, default=-1e-8,
                        help="Initial time of the simulation (s).")
    
    parser.add_argument("--time-final", type=float, default=1e-8,
                        help="Final time of the simulation (s).")   

    parser.add_argument("--Nt", type=int, default=10000,
                        help="Number of time steps.")   
    

    parser.add_argument("--distance", type=float, default=AU_TO_PC,
                    help="Distance to the source in parsecs (default: 1 AU).")
    
    parser.add_argument("--inclination", type=float, default=0.0,
                        help="Inclination angle of the binary orbit (radians).")
    
    parser.add_argument("--polarization-angle", type=float, default=0.0,
                        help="Polarization angle of the gravitational wave (radians).")
    

    parser.add_argument("--plot", action="store_true",
                        help="Plot waveforms at each stage of generation.")

    parser.add_argument("--display", action="store_true",
                        help="Display waveforms at each stage of generation.")
    

    parser.add_argument("--data-dir", type=str, default='data',
                        help="Folder where output files will be saved. Created if it does not exist.")
    
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename stem (without extension).")
    
    return parser.parse_args()

def main():
    args = parse_args()

    output_path = prepare_output_path(args.data_dir, args.output, args.m_total, args.q, hyperbolic=True)

    run_info = {
        **vars(args),
        "hyperbolic": True,
    }

    with open(output_path+"_config.json", "w") as f:
        json.dump(run_info, f, indent=2)

    start = time.time()

    times, h_plus, h_cross, f_peak, _, _ = get_hyp_waveform(args.m_total, args.q, args.initial_eccentricity,
                                              args.b, args.time_initial, args.time_final,
                                              args.Nt, args.inclination, args.distance, order=3, estimatepeak=True)
    

    data = [times, h_plus, h_cross]

    # plot_waveform
    plot_waveform(data, labels=(r"$h_+$", r"$h_\times$"), title="PBH hyperbolic encounter", save_path=None, display=args.display)

    data = rotate_polarization(data, args.polarization_angle, plot=args.plot, display=args.display)
    
    print(f"[INFO] Signal peaks at {f_peak/1e9:.2f} GHz.")
    print(f"[INFO] Computed in {time.time()-start: .2f} s.")

    np.save(output_path + ".npy", data)
    print(f"[INFO] Saved waveform data to {output_path}.npy")

if __name__ == "__main__":
    main()
