import numpy as np
from pycbc.waveform import get_td_waveform
from pyseobnr.generate_waveform import GenerateWaveform

def simple_td_model(m_1, m_2, f_lower, f_higher, delta_t, inclination, phi0, dist_mpc):
    c_const = 3e8
    G_const = 6.67e-11
    M_sun = 2e30

    m_1 *= M_sun
    m_2 *= M_sun
    distance = dist_mpc * 3.0857e22

    M_chirp = (m_1 * m_2)**(3/5) / (m_1 + m_2)**(1/5)

    t_start = 5/256 * (np.pi * f_higher)**(-8/3) * (G_const * M_chirp / c_const**3)**(-5/3)
    t_end   = 5/256 * (np.pi * f_lower )**(-8/3) * (G_const * M_chirp / c_const**3)**(-5/3)

    t_obs = np.linspace(t_start, t_end, int(1e7))
    Phi = -2.0 * ((5 * G_const * M_chirp) / c_const**3)**(-5/8) * t_obs**(5/8) + phi0

    hc = (1 / distance) * ((G_const * M_chirp) / c_const**2)**(5/4) * (5 / (c_const * t_obs))**(1/4)

    h_plus  = hc * (1 + np.cos(inclination)**2)/2 * np.cos(Phi)
    h_cross = hc * np.cos(inclination) * np.sin(Phi)

    return -t_obs, h_plus, h_cross


def generate_waveform(model, m1, m2, spin1, spin2, del_t,
                      dist_mpc, incl, f_low, f_high, ecc, phi0):

    if model.lower() == "newtonian":
        times, hp, hc = simple_td_model(m1, m2, f_low, f_high, del_t, incl, phi0, dist_mpc)

    elif ecc == 0.0:
        hp, hc = get_td_waveform(
            approximant=model,
            mass1=m1, mass2=m2,
            spin1x=spin1[0], spin1y=spin1[1], spin1z=spin1[2],
            spin2x=spin2[0], spin2y=spin2[1], spin2z=spin2[2],
            delta_t=del_t,
            distance=dist_mpc,
            inclination=incl,
            phi=phi0,
            f_lower=f_low,
            f_higher=f_high,
        )
        times = hp.sample_times

    else:
        gen = GenerateWaveform(
            approximant=model,
            mass1=m1, mass2=m2,
            spin1x=spin1[0], spin1y=spin1[1], spin1z=spin1[2],
            spin2x=spin2[0], spin2y=spin2[1], spin2z=spin2[2],
            eccentricity=ecc,
            deltaT=del_t,
            phi_ref=phi0,
            distance=dist_mpc,
            inclination=incl,
            f22_start=f_low,
            f_max=f_high,
        )

        hp, hc = gen.generate()
        times = hp.epoch + hp.deltaT * np.arange(hp.data.length)

    data = np.array([times, hp, hc])
        
    return data
