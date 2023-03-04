import resampling
import numpy as np
from random import sample
import math
from scipy.stats import multivariate_normal

def calc_weight(Zt, Q, Zt_1):
    # Q = 1                               # measument covariance
    # Zt_1 = 10                           # predicted Zt

    # w = target(xt) / proposal(xt)

    Wt = (2*math.pi * Q)**(-1/2) * math.exp(-1/2 * (Zt - Zt_1)**2 / Q * (Zt - Zt_1))
    # how to compute weight? Gaussian with replace sigma to Q and x - u with Zt and predicted Zt
    return 1

def varience(mu_d, J): # sigma calc eq
    return (1 / J - 1) * mu_d


def creating_partcles(pre_particle, Ut, Zt, Ct): # form of Ut and Zt gotta be different format. This is just for referace.

    # every sample should have : pose(x,y,z), weight, landmarks(list of 2 by 2 kalman filter)
    
    N = 10    # N number of sample, particle created.
    sigma = 3 # Tune or calc??
    Particle_bar = 0
    mu_d = 0
    
    particle_set = np.zeros(N,1) #[x,y,theta, weight]

    for n in range(N):
        x = np.random.normal(pre_particle + Ut, sigma, 1) # Obtain new x value for new sample #starts from uniform distribution
        y = np.random.normal(pre_particle + Ut, sigma, 1) #Obtains new y value for new sample
        theta = np.random.normal(pre_particle + Ut, sigma, 1) #Between 0 and 2pi radians
        particle = [x,y,theta]
        
        mu_d += (particle - (pre_particle + Ut))**2 # if we have to do sigma calc, NOT BEING USED IRRELEVANT DONT PAY ATTENTION


    #     ----------------------landmark, Ct-------------------------------
    #     if Ct never seen before: 
    #         1. initialize mean = mu
    #         2. calculate Jacobian = H 
    #         3. initialize covariance
    #         4. default importance weight

    #     else:
    #         <EKF-update> // update landmark # first compute weight and update the map??
    #             measurement prediction = Z_hat
    #             calculate Jacobian = H
    #             measurment covariance = Q
    #             calculate Kalman gain = K
    #             update mean = mu
    #             update covariance
        
            # Wt = calc_weight(Zt, prev_Q, Zt_1) # calc weight
    #         # keep unobserved ladmark unchanged
    # end for loop
    # resampling and get Xt

        Wt = calc_weight(Zt, 1, 1) # calc weight
        Particle_bar += np.dot(particle, Wt) # predicted pos. assume Wt is probability. We can ignore this(can be replaced by resampling step)

        particle.append(Wt)
        particle_set[N] = particle
        
        
    sigma = varience(mu_d, N)
    Particle = resampling(Particle_bar, Particle_bar[n][1]) # resampling from sample set. Need to be fixed

    return particle_set, Particle_bar, sigma


if __name__ == '__main__':

    particle_set = creating_partcles(50, 50, 1, 1)
    print("particle_set =\n", particle_set)
