import numpy as np
from pycbc.waveform import get_td_waveform
from pyseobnr.generate_waveform import GenerateWaveform
from numpy import sin, cos
from gw.utils_hyperbolic import *

# Takes in mass in solar masses, b in units of GM/c^2, ti, tf in seconds, t_step as an integer, inc in radians, distance in pc
def get_hyp_waveform(M, q, et0, b, ti, tf, t_step, inc, distance, order, estimatepeak=False):

        distance = distance * pc

        closest_approach = b * np.sqrt((et0 - 1)/(et0 + 1))

        if closest_approach < 2:
            print("[WARNING]: Waveform is unphysical, as black holes get closer to each other than two Schwartzschild radii.")

        eta=q/(1+q)**2
        Time=M*tsun
        dis=M*dsun
        scale=distance/dis
        x0=get_x(et0,eta,b,3)[0]
        n0=x0**(3/2)
        tarr=np.linspace(ti,tf,t_step)
        t_arr=tarr/Time
        t_i=t_arr[0]
        t_f=t_arr[len(t_arr)-1]
        l_i=n0*t_i

        u_i=get_u(l_i,et0,eta,b,3)
        y0=[et0,n0,u_i]
        sol=solve_rr(eta,b,y0,t_i,t_f,t_arr)

        uarr=sol[2]
        earr=sol[0]
        narr=sol[1]

        step=len(tarr)
        hp_arr=np.zeros(step)
        hx_arr=np.zeros(step)
        X=np.zeros(step)
        Y=np.zeros(step)

        rt_arr=np.zeros(step)
        for i in range(step):
            et=earr[i]
            u=uarr[i]
            x=narr[i]**(2/3) 
            phi=phiv(eta,et,u,x,order)
            r1=rx(eta,et,u,x,order)
            z=1/r1
            phit=phitx(eta,et,u,x,order)
            rt=rtx(eta,et,u,x,order)

            rt_arr[i] = rt

            phi=phiv(eta,et,u,x,order)
            r1=rx(eta,et,u,x,order)
            X[i]=r1*cos(phi)
            Y[i]=r1*sin(phi)
            hp_arr[i]=(-eta*(sin(inc)**2*(z-r1**2*phit**2-rt**2)+(1+cos(inc)**2)*((z
            +r1**2*phit**2-rt**2)*cos(2*phi)+2*r1*rt*phit*sin(2*phi))))
            hx_arr[i]=(-2*eta*cos(inc)*((z+r1**2*phit**2-rt**2)*sin(2*phi)-2*r1*rt*phit*cos(2*phi)))

        maximum_velocity = np.max(np.abs(rt_arr))

        print(f"[INFO] Maximum velocity during approach was {maximum_velocity: .3f} c.")
        if np.max(np.abs(rt_arr)) > 0.5:
            print("[WARNING] The PN approximation may not be convergent.")
        Hp=hp_arr/scale
        Hx=hx_arr/scale

        #Eliminate DC offset term at -infinity
        if estimatepeak == True:
            dimless_peak = get_max(eta,b,et0)
            peak = dimless_peak / (2*np.pi*Time)
            return t_arr*Time, Hp-Hp[0], Hx-Hx[0], peak, X, Y
        else:
            return t_arr*Time, Hp-Hp[0], Hx-Hx[0]
        

def simple_td_model(m_1, m_2, f_lower, f_higher, delta_t, inclination, phi0, dist_mpc):
    # This is broken currently
    c_const = 3e8
    G_const = 6.67e-11
    M_sun = 2e30

    m_1 *= M_sun
    m_2 *= M_sun
    distance = dist_mpc * 3.0857e22

    M_chirp = (m_1 * m_2)**(3/5) / (m_1 + m_2)**(1/5)

    t_start = 5/256 * (np.pi * f_higher)**(-8/3) * (G_const * M_chirp / c_const**3)**(-5/3)
    t_end   = 5/256 * (np.pi * f_lower )**(-8/3) * (G_const * M_chirp / c_const**3)**(-5/3)

    t_obs = np.linspace(t_start, t_end, int((t_end-t_start)/delta_t))
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
        )

        hp, hc = gen.generate()
        times = hp.epoch + hp.deltaT * np.arange(hp.data.length)

    data = np.array([times, hp, hc])
        
    return data
