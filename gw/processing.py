import numpy as np
import matplotlib.pyplot as plt
from gwmemory import time_domain_memory
from gw.utils import plot_waveform

def clip_waveform(data, clip_th1=0.5e1, clip_th2=1e4, plot=False):
    threshold = np.max(data[1] / clip_th1)
    threshold_2 = np.max(data[1] / clip_th2)

    start = min(
        np.argmax(np.abs(data[1]) > threshold),
        np.argmax(np.abs(data[2]) > threshold),
    )

    data = data[:, start:]

    # --- find end (from the right) ---
    n = data.shape[1]

    end = min(
        n - np.argmax(np.abs(data[1][::-1]) > threshold_2),
        n - np.argmax(np.abs(data[2][::-1]) > threshold_2),
    )
    data = data[:, :end]

    if plot:
        plot_waveform(data, ("h+ clipped", "hx clipped"))

    return data


def add_gw_memory(data, M_tot, q, spin_1, spin_2,
                  dist_mpc, incl, phase, approximant, ratio, plot):

    times = data[0] * ratio
    h_mem, times = time_domain_memory(
        total_mass=M_tot, q=q, model=approximant,
        spin_1=spin_1, spin_2=spin_2,
        distance=dist_mpc, inc=incl,
        phase=phase, times=times
    )

    memory = np.array([times, h_mem["plus"], h_mem["cross"]]) / ratio

    if plot:
        plot_waveform(memory, ("h+ memory", "hx memory"))

    return memory


def rotate_polarization(data, angle, plot):
    s, c = np.sin(2*angle), np.cos(2*angle)
    hp = data[1]*c - data[2]*s
    hc = data[1]*s + data[2]*c
    data[1], data[2] = hp, hc

    if plot:
        plot_waveform(data, ("h+ rotated", "hx rotated"))
    
    return data

    
