import numpy as np
import pandas as pd

def h_monochromatic(amplitude, tau, omega, phase=0):
    return amplitude * np.exp(1j * (omega * tau + phase))

def load_B(filename):

    df = pd.read_excel(filename)
    df = df.apply(pd.to_numeric)

    df["z"] /= 100.0   # cm -> m
    df["r"] /= 100.0   # cm -> m

    df["Bz"] /= 10000.0   # G -> T
    df["Br"] /= 10000.0   # G -> T

    z = np.sort(df["z"].unique())
    r = np.sort(df["r"].unique()) 

    Bz = (
        df.pivot(index="z", columns="r", values="Bz")
        .loc[z, r]
        .to_numpy()
    ) 

    Br = (
        df.pivot(index="z", columns="r", values="Br")
        .loc[z, r]
        .to_numpy()
    ) 

    return r, z, Br, Bz

def mean_calc(eta, theta):
    eta_sum, sin_sum = 0.0, 0.0
    for row in eta:          
        for i, element in enumerate(row):
            sin_theta = np.sin(theta[i])
            eta_sum += element * sin_theta
            sin_sum += sin_theta
    
    result = eta_sum / sin_sum if sin_sum > 0 else 0.0

    return result