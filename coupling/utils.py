import numpy as np

def h_monochromatic(tau, omega):
    return np.exp(1j * omega * tau)

def mean_calc(eta, theta):
    eta_sum, sin_sum = 0.0, 0.0
    for row in eta:          
        for i, element in enumerate(row):
            sin_theta = np.sin(theta[i])
            eta_sum += element * sin_theta
            sin_sum += sin_theta
    
    result = eta_sum / sin_sum if sin_sum > 0 else 0.0

    return result