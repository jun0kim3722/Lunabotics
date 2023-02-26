import resampling
import numpy as np
from random import sample
import math


def calc_weight(Zt, Q, Zt_1):
    # Q = 1                               # measument covariance
    # Zt_1 = 10                           # predicted Zt

    # w = target(xt) / proposal(xt)

    Wt = (2*math.pi * Q)**(-1/2) * math.exp(-1/2 * (Zt - Zt_1)**2 / Q * (Zt - Zt_1))
    # how to compute weight? Gaussian with replace sigma to Q and x - u with Zt and predicted Zt
    return 1

def varience(mu_d, J): # sigma calc eq
    return (1 / J - 1) * mu_d


def creating_partcles(pre_Xt, Ut, Zt, Ct): # form of Ut and Zt gotta be different format. This is just for referace.

    # every sample should have : pose(x,y,z), weight, landmarks(list of 2 by 2 kalman filter)
    
    J = 10    # J number of sample, particle created.
    sigma = 3 # Tune or calc??
    X_bar = 0
    mu_d = 0

    for j in range(J):
        xt = np.random.normal(pre_Xt + Ut, sigma, 1) # Obtain new sample #starts from uniform distribution
        
        mu_d += (xt - (pre_Xt + Ut))**2 # if we have to do sigma calc   
        
        # so Do you get array of samples in every j? so J number of array set? I mean usually X_bar means mean of samples so this is the answer?
        # or just one sample at a time which J number of sample


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
        X_bar += np.dot(xt, Wt) # predicted pos. assume Wt is probability. We can ignore this(can be replaced by resampling step)

    sigma = varience(mu_d, J)
    Xt = resampling(X_bar, X_bar[j][1]) # resampling from sample set. Need to be fixed

    return Xt, X_bar, sigma


if __name__ == '__main__':

    Xt = creating_partcles(50, 50, 1, 1)
    print("Xt =\n", Xt)
