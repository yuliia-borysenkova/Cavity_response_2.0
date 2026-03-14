import numpy as np
import matplotlib.pyplot as plt
from gwmemory import time_domain_memory
from gw.utils import plot_waveform

def clip_waveform(data, clip_th1=0.2, clip_th2=1e-4, plot=False, display=False):
    
    h1 = np.abs(data[1])
    h2 = np.abs(data[2])

    peak = max(h1.max(), h2.max())

    start_th = clip_th1 * peak
    end_th = clip_th2 * peak

    # ---- find start ----
    start_mask = (h1 > start_th) | (h2 > start_th)


    if not np.any(start_mask):
        raise ValueError("Start threshold never crossed")

    start = np.argmax(start_mask)

    data = data[:, start:]

    # ---- find end ----
    h1 = np.abs(data[1])
    h2 = np.abs(data[2])

    end_mask = (h1 > end_th) | (h2 > end_th)
    if not np.any(end_mask):
        raise ValueError("End threshold never crossed")

    end = np.where(end_mask)[0][-1] + 1

    data = data[:, :end]

    if plot:
        plot_waveform(data, (r"$h_+$ clipped", r"$h_\times$ clipped"), display=display)

    return data


def add_gw_memory(data, M_tot, q, spin_1, spin_2,
                  dist_mpc, incl, phase, approximant, ratio, plot=False, display=False):

    times = data[0] * ratio
    h_mem, times = time_domain_memory(
        total_mass=M_tot, q=q, model=approximant,
        spin_1=spin_1, spin_2=spin_2,
        distance=dist_mpc, inc=incl,
        phase=phase, times=times
    )

    memory = np.array([times, h_mem["plus"], h_mem["cross"]]) / ratio

    if plot:
        plot_waveform(memory, (r"$h_+$ memory", r"$h_\times$ memory"), display=display)

    return memory


def rotate_polarization(data, angle, plot=False, display=False):
    s, c = np.sin(2*angle), np.cos(2*angle)
    hp = data[1]*c - data[2]*s
    hc = data[1]*s + data[2]*c
    data[1], data[2] = hp, hc

    if plot:
        plot_waveform(data, (r"$h_+$ rotated", r"$h_\times$ rotated"), display=display)
    
    return data

    
