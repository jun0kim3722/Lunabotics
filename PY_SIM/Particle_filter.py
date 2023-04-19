import numpy as np
from math import pi
from numpy.random import random
from functools import reduce
import pdb

class landmark:
    num_land = 0
    l_pos = []
    l_sig = []
    l_Qt = []
    
    def __init__ (self, l_pos, sig, Qt):
        landmark.num_land += 1
        landmark.l_sig.append(sig)
        landmark.l_pos.append(l_pos)
        landmark.l_Qt.append(Qt)
    
    def update_landmark(index, l_pos, sig, Qt):
        landmark.l_sig[index] = sig
        landmark.l_pos[index] = l_pos
        landmark.l_Qt[index] = Qt

    def prev_val (idx):
        l_pos = landmark.l_pos[idx]
        sig = landmark.l_sig[idx]
        Qt = landmark.l_Qt[idx]
        return l_pos, sig, Qt

class particle_filter:

    def __init__(self, m_sigma, l_sigma, N, map):
        self.prev_R_pos = []
        self.Ct = []
        self.m_sigma = m_sigma
        self.l_sigma = l_sigma
        self.N = N
        self.weight = []
        self.l_weight = []
        self.map_size = map

    def creating_particles(self, Ut, Zt): # form of Ut and Zt gotta be different format. This is just for referace.

        particle_set = np.zeros((self.N, 4)) #[x, y, theta, Weight]

        for n in range(self.N):
            if self.prev_R_pos != []:
                x = np.random.normal(self.prev_R_pos[0] + Ut[0], self.m_sigma[0], 1) # Obtain new x value for new sample
                y = np.random.normal(self.prev_R_pos[1] + Ut[1], self.m_sigma[1], 1) #Obtains new y value for new sample
                theta = np.random.normal(self.prev_R_pos[2] + Ut[2], self.m_sigma[2], 1) #Between 0 and 2pi radians

            else: #starts from uniform distribution
                x = np.random.uniform(0, self.map_size[0], 1)
                y = np.random.uniform(0, self.map_size[1], 1)
                theta = np.random.uniform(0, 2, 1)

            particle = np.concatenate((x, y, theta))
            
        #     ----------------------landmark, Ct-------------------------------
            for i, zt in enumerate(Zt):
                if self.Ct[i][0]: # Ct never seen before: Ct = Matrix that discribe how to map the state Xt to an observation Zt
                    # initialize mean = mu => list of landmarks
                    l_pos = landmark_pos(particle, zt)
                    Z_hat, delta = h(particle, l_pos)
                    Z_hat = Z_hat[:, np.newaxis]

                    # calculate Jacobian = H 
                    H = calc_jacobian(Z_hat, delta)

                    # initialize covariance => list of uncertainty of landmarks
                    Qt = init_Qt(self)
                    inv_H = np.linalg.pinv(H)
                    sig = inv_H @ Qt @ inv_H.T

                    # default importance weight
                    Wt = 1/self.N # tune value I believe
                    self.l_weight.append(Wt)

                    # init landmark class
                    landmark(l_pos, sig, Qt)

                else:   #<EKF-update> // update landmark
                    # Get prev value
                    prev_l_pos, prev_sig, prev_Qt = landmark.prev_val(self.Ct[i][1])

                    # measurement prediction = Z_hat
                    Z_hat, delta = h(particle, prev_l_pos)
                    Z_hat_t = Z_hat[:, np.newaxis]

                    # calculate Jacobian = H
                    H = calc_jacobian(Z_hat_t, delta)

                    # measurment covariance = Q
                    Q = H @ prev_sig @ H.T + prev_Qt

                    # calculate Kalman gain = K
                    K = calc_kalmangain(prev_sig, H, Q)

                    # update mean = mu ==> mu + K(Zt - z_hat)
                    l_pos = prev_l_pos + (K @ (zt - Z_hat))

                    # update covariance ==> ()
                    sig = (np.identity(H.shape[1]) - K @ H) @ prev_sig

                    # update landmark class
                    landmark.update_landmark(self.Ct[i][1], l_pos, sig, Qt)

                    # calc weight
                    Wt = calc_weight(zt, Q, Z_hat)
                    self.l_weight.append(Wt)
                    if (np.isnan(Wt).any()):
                        pdb.set_trace()
            
            self.weight.append(sum(self.l_weight)/len(self.l_weight)); self.l_weight = []
            particle_set[n] = np.concatenate((x, y, theta, np.array([Wt])))
            # self.particle.append(particle_set)

        # particle_list = list(map(lambda x: particle_set[x], resampling(self.N, self.weight))) # list of particles

        particle_bar = reduce((lambda x, y : np.add(x, y)), [[particle_set[i][j] * particle_set[0][3] for j in range(3)] for i in resampling(self.N, self.weight)])
        # self.prev_R_pos = particle_bar # update previous list of particles
        self.Ct = []; self.weight = []

        return particle_set, particle_bar

def landmark_pos(particle, Zt):
    R_x = particle[0] # Robot pos x
    R_y = particle[1] # Robot pos y
    R_th = particle[2] # Robot pos theta

    L_th = Zt[1] # Angle btw robot to landmark
    L_d = Zt[0] # distance from robot to landmark

    L_pos = np.array([[R_x], [R_y]]) + np.array([[L_d * np.cos(L_th + R_th)], [L_d * np.sin(L_th + R_th)]])
    
    return L_pos # returning x, y of landmark

# Observation function
def h(particle, L_pos):
    R_x = particle[0]
    R_y = particle[1]
    R_th = particle[2]

    L_x = L_pos[0] # landmark x
    L_y = L_pos[1] # landmark y

    delta = np.array([L_x - R_x, L_y - R_y])
    q = delta.T @ delta
    q = q[0][0]

    Z_hat = np.array([np.sqrt(q), np.arctan2(delta[1][0], delta[0][0]) - R_th])

    return Z_hat, delta # returning expected observation

# Jacobian calculation function
def calc_jacobian(Z_hat, delta):
    sqrt_q = Z_hat[0][0]
    q = sqrt_q ** 2
    x = np.array([-delta[0][0], -delta[1][0], 0, delta[0][0], delta[1][0]])
    y = np.array([ delta[1][0], -delta[0][0], -q, -delta[1][0], delta[0][0]])

    # id_3 = np.concatenate((np.identity(3), np.zeros((2,3))), axis=0)
    # zero = np.zeros((5, 2*j - 2))
    # id_2 = np.concatenate((np.zeros((3,2)), np.identity(2)), axis=0)
    # M_high = np.concatenate((id_3, zero, id_2), axis= 1)

    H = (1/q) * np.array([sqrt_q * x,y]) #@ M_high
    return H

def init_Qt(self):
    Qt = np.array([[self.l_sigma[0]**2, 0], [0, self.l_sigma[1]]])
    return Qt

def calc_kalmangain(sig ,H, Q):
    K = sig @ H.T @ np.linalg.inv(Q)
    return K

def calc_weight(Zt, Q, Z_hat):
    Wt = (2 * pi * Q) ** (-1/2) * np.exp(-1/2 * (Zt - Z_hat) ** 2 / Q * (Zt - Z_hat))
    return Wt

def resampling(N, weight):
    positions = (random(N) + range(N)) / N
    indexes = np.zeros(N, 'i')
    weight /= np.sum(weight)
    cumulative_sum = np.cumsum(weight)
    i , j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
        
    return indexes

if __name__ == '__main__':
    particle = particle_filter([10,10,10], [10,10,10], 10, [100, 100])

    particle.Ct = [True, True]

    particle_set, robot_pos = particle.creating_particles([0.3,3.6, 0.2], [[2, pi/2], [1.3, pi/2]])
    print(particle_set)
    print(robot_pos)
