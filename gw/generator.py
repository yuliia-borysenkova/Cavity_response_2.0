import json, argparse
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field, asdict
from gw.models import generate_waveform
from gw.processing import (
    clip_waveform, add_gw_memory, rotate_polarization
)
from gw.utils import plot_waveform, prepare_output_path


@dataclass
class WaveformConfig:
    # Masses and spins
    m_absolute: float = 1e-7
    q: float = 1.0
    spin_1: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    spin_2: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    
    # Orbital parameters
    eccentricity: float = 0.0
    phi0: float = 0.0
    tc: float = 0.0
    inclination: float = 0.0
    polarization_angle: float = 0.0
    
    # Frequency settings
    low_freq: float = 1e9
    high_freq: float = 1e12
    
    # Distance
    distance: float = 1e-11
    
    # Waveform options
    approximant: str = "IMRPhenomD"
    clip: bool = False
    memory: bool = False
    plot: bool = False
    
    # Output
    data_dir: str = "data"
    output: Optional[str] = None

    # Optional hyperparameters
    density_factor: Optional[float] = 1.0
    clip_th1: Optional[float] = 0.5e1
    clip_th2: Optional[float] = 1e4


class WaveformPipeline:
    def __init__(self, cfg, output_path):  # accept one argument in addition to self
        self.cfg = cfg
        self.output_path = output_path
        
    def run(self):
        #self.save_config_to_json()
        
        ratio = 10 / self.cfg.m_absolute

        m2 = self.cfg.m_absolute / (1 + self.cfg.q) * ratio
        m1 = self.cfg.q * m2

        distance_mpc = self.cfg.distance / 1e6

        print(f"[INFO] Generating waveform for m1 = {m1 /ratio}, m2 = {m2 / ratio} at ratio {ratio} and distance {distance_mpc} Mpc")

        data = generate_waveform(
            self.cfg.approximant,
            m1, m2,
            self.cfg.spin_1, self.cfg.spin_2,
            1 / (self.cfg.density_factor * self.cfg.high_freq / ratio),
            distance_mpc,
            self.cfg.inclination,
            self.cfg.low_freq / ratio,
            self.cfg.high_freq / ratio,
            self.cfg.eccentricity,
            self.cfg.phi0,
        ) / ratio

        if self.cfg.plot:
            plot_waveform(data, ("h+ " + self.cfg.approximant, "hx " + self.cfg.approximant))

        if self.cfg.clip:
            print("[INFO] Clipping waveform")
            data = clip_waveform(data, self.cfg.plot, self.cfg.clip_th1, self.cfg.clip_th2)

        if self.cfg.polarization_angle != 0: # Should this go before or after memory?
            print(f"========== Rotating by polarization angle {self.cfg.polarization_angle} rad ========== ")
            data = rotate_polarization(data, self.cfg.polarization_angle, self.cfg.plot)

        if self.cfg.memory:
            print("[INFO] Adding GW memory")
            data += add_gw_memory(
                data, m1 + m2, self.cfg.q,
                self.cfg.spin_1, self.cfg.spin_2,
                distance_mpc,
                self.cfg.inclination,
                self.cfg.phi0,
                self.cfg.approximant,
                ratio,
                self.cfg.plot,
            )

        if self.cfg.plot:
            plot_waveform(data, ("h+ final " + self.cfg.approximant, "hx final " + self.cfg.approximant),
                          title = f"GW waveform for a m1 = {m1 / ratio}, m2 = {m2 / ratio} solar mass BH merger",
                          save_path=self.output_path)

        return data

    