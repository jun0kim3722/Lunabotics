import numpy as np
import math
from scipy.stats import multivariate_normal
from numpy.random import random
from functools import reduce

def calc_weight(Zt, Q, Zt_1):
    Wt = (2 * math.pi * Q) ** (-1/2) * np.exp(-1/2 * (Zt - Zt_1) ** 2 / Q * (Zt - Zt_1))
    return Wt

class particle_filter:
    def __init__(self, sigma, N):
        self.pre_particle = [0, 0, 0]
        self.Ct = None
        self.sigma = sigma
        self.N = N
        self.weight = []
        self.particle = []

    def creating_particles(self, Ut, Zt): # form of Ut and Zt gotta be different format. This is just for referace.

        particle_set = np.zeros((self.N, 4)) #[x, y, theta, Weight]

        for n in range(self.N):
            x = np.random.normal(self.pre_particle[0] + Ut, self.sigma[0], 1) # Obtain new x value for new sample #starts from uniform distribution
            y = np.random.normal(self.pre_particle[1] + Ut, self.sigma[1], 1) #Obtains new y value for new sample
            theta = np.random.normal(self.pre_particle[2] + Ut, self.sigma[2], 1) #Between 0 and 2pi radians

            particle = np.concatenate((x, y, theta))

        #     ----------------------landmark, Ct-------------------------------
            if self.Ct == True: # Ct never seen before: Ct = Matrix that discribe how to map the state Xt to an observation Zt
        #   1. initialize mean = mu
                
        #   2. calculate Jacobian = H
                H = calc_jacobian(particle, Z_hat)
        #   3. initialize covariance

        #   4. default importance weight

            else:   #<EKF-update> // update landmark # first compute weight and update the map??
        #       measurement prediction = Z_hat
                Z_hat = h(particle, Ut)
        #       state transition update = Zt_1
                Zt_1 = f(self.pre_particle, Ut, Wt)
        #       calculate Jacobian = H
                H = calc_jacobian(particle, Z_hat)
        #       measurment covariance = Q and R
                Q = calc_covariance_Q(particle, Wt)
                R = calc_covariance_R(Z_hat)    
        
            
            Wt = calc_weight(Zt, Q, Zt_1) # calc weight
            self.weight.append(Wt)
            particle_set = np.concatenate((x, y, theta, np.array([Wt])))
            self.particle.append(particle_set)

        particle = list(map(lambda x: self.particle[x], resampling(self))) # list of particles
        self.pre_particle = particle # update previous list of particles
        particle_bar = reduce((lambda x,y : x + y), [particle[i][0:3] * particle[i][3] for i in range(len(particle))])

        return particle_set, particle_bar

# State transition function
def f(pre_particle, Ut, Wt):
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[0.5], [1]])
    Zt_1 = np.dot(A, pre_particle) + np.dot(B, Ut) + Wt
    return Zt_1

# Observation function
def h(particle, Ut):
    C = np.array([1, 0])
    Z_hat = np.dot(C, particle) + Ut
    return Z_hat

# Jacobian calculation function
def calc_jacobian(particle, Z_hat):
    num_diff = 1e-6
    N = particle.shape[0]
    m = Z_hat(particle).shape[0]
    H = np.zeros((m,N))
    for i in range(N):
        particle_i = particle.copy()
        particle_i[i] += num_diff
        H[:,i] = (Z_hat(particle_i) - Z_hat(particle)) / num_diff
    return H

# Calculates covariance matrix for process noise
def calc_covariance_Q(particle, Wt):
    Q = np.cov(particle, aweights=Wt)
    return Q

# Calculates covariance matrix for observation noise
def calc_covariance_R(Z_hat):
    z_mean = np.mean(Z_hat, axis=1)
    z_diff = Z_hat - z_mean.reshape(-1, 1)
    R = np.cov(z_diff)
    return R


def resampling(self): # Broken gotta fix.
    
    N = self.N
    positions = (random(N) + range(N)) / N
    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(self.weight)
    i , j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
        
    return indexes



if __name__ == '__main__':

    particle = particle_filter([10,10,10], 10)

    particle_set = particle.creating_particles(1, 1)

    # particle.weight = [0.1]*10
    # resample = resampling(particle)
    # print(resample)

