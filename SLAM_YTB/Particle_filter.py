import numpy as np
import math
from numpy.random import random
from functools import reduce

class particle_filter:
    prev_sig = []
    prev_l_pos = []
    Qt = None
    l_num = 0

    def __init__(self, m_sigma, l_sigma, N, map):
        self.prev_particle = []
        self.Ct = []
        self.m_sigma = m_sigma
        self.l_sigma = l_sigma
        self.N = N
        self.weight = []
        self.particle = []
        self.map_size = map

    def creating_particles(self, Ut, Zt): # form of Ut and Zt gotta be different format. This is just for referace.

        particle_set = np.zeros((self.N, 4)) #[x, y, theta, Weight]

        for n in range(self.N):
            if self.prev_particle != []:
                x = np.random.normal(self.prev_particle[0] + Ut[0], self.m_sigma[0], 1) # Obtain new x value for new sample
                y = np.random.normal(self.prev_particle[1] + Ut[1], self.m_sigma[1], 1) #Obtains new y value for new sample
                theta = np.random.normal(self.prev_particle[2] + Ut[2], self.m_sigma[2], 1) #Between 0 and 2pi radians

            else: #starts from uniform distribution
                x = np.random.uniform(0, self.map_size[0], 1)
                y = np.random.uniform(0, self.map_size[1], 1)
                theta = np.random.uniform(0, 2, 1)

            particle = np.concatenate((x, y, theta))
            
        #     ----------------------landmark, Ct-------------------------------
            for i, zt in enumerate(Zt):
                if self.Ct[i]: # Ct never seen before: Ct = Matrix that discribe how to map the state Xt to an observation Zt
                    # initialize mean = mu => list of landmarks
                    self.l_num += 1
                    l_pos = landmark_pos(particle, zt); self.prev_l_pos = l_pos
                    Z_hat, delta = h(particle, l_pos)
                    Z_hat = Z_hat[:, np.newaxis]

                    # calculate Jacobian = H 
                    H = calc_jacobian(Z_hat, delta, self.l_num) # Wrong eq ig??
                    # H = h'(L_pos, particle) ???

                    # initialize covariance => list of uncertainty of landmarks
                    Qt = init_Qt(self); self.Qt = Qt
                    inv_H = np.linalg.pinv(H)
                    sig = inv_H @ Qt @inv_H.T ; self.prev_sig = sig

                    # default importance weight
                    Wt = 1/self.N # turn value I believe
                    self.weight.append(Wt)

                else:   #<EKF-update> // update landmark
                    # Zt_1 = f(self.prev_particle, Ut, Wt) # state transition update = Zt_1
                    print("LANDMARK DETECT")
                    # measurement prediction = Z_hat
                    Z_hat, delta = h(particle, self.prev_l_pos)
                    Z_hat = Z_hat[:, np.newaxis]

                    # calculate Jacobian = H
                    H = calc_jacobian(Z_hat, delta, self.l_num)

                    # measurment covariance = Q
                    Q = H @ sig @ H.T + Qt
                    #self.Qt = update_Qt(Qt)

                    # calculate Kalman gain = K
                    K = calc_kalmangain(self.prev_sig ,H, particle, Q)   

                    # update mean = mu ==> mu + K(Zt - z_hat)
                    l_pos = self.prev_l_pos + K @ (zt - Z_hat)
                    self.prev_l_pos = l_pos

                    # update covariance ==> ()
                    sig = (np.identity(2) - K @ H) @ self.prev_sig
                    self.prev_sig = sig

                    # calc weight
                    Wt = calc_weight(zt, Q, Z_hat)
                    self.weight.append(Wt)
                
            particle_set[n] = np.concatenate((x, y, theta, np.array([Wt])))
            self.particle.append(particle_set)

        particle_list = list(map(lambda x: self.particle[x], resampling(self))) # list of particles
        particle_bar = reduce((lambda x,y : np.add(x, y)), [[particle_set[i][j] * particle_set[0][3] for j in range(3)] for i in range(len(particle_list))])
        self.prev_particle = particle_bar # update previous list of particles
        # print("done", self.Ct, Zt)
        self.Ct = []

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
def calc_jacobian(Z_hat, delta, j):
    sqrt_q = Z_hat[0][0]
    q = sqrt_q ** 2
    x = np.array([-delta[0][0], -delta[1][0], 0, delta[0][0], delta[1][0]])
    y = np.array([ delta[1][0], -delta[0][0], -q , -delta[1][0], delta[0][0]])

    id_3 = np.concatenate((np.identity(3), np.zeros((2,3))), axis=0)
    zero = np.zeros((5, 2*j - 2))
    id_2 = np.concatenate((np.zeros((3,2)), np.identity(2)), axis=0)
    M_high = np.concatenate((id_3, zero, id_2), axis= 1)

    H = (1/q) * np.array([sqrt_q * x,y]) #@ M_high

    return H

def init_Qt(self):
    Qt = np.array([[self.l_sigma[0]**2, 0], [0, self.l_sigma[1]]])
    return Qt

#def update_Qt(Qt):
    #Qt1 = [Qt , 0],

def calc_kalmangain(sig ,H, particle, Q):
    K = sig @ H.T @ np.linalg.inv(Q)
    return K

def calc_weight(Zt, Q, Z_hat):
    Wt = (2 * math.pi * Q) ** (-1/2) * np.exp(-1/2 * (Zt - Z_hat) ** 2 / Q * (Zt - Z_hat))
    return Wt

def resampling(self):
    
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
    particle = particle_filter([10,10,10], [10,10,10], 10, [100, 100])

    particle.Ct = [True, True]

    particle_set, robot_pos = particle.creating_particles([0.3,3.6, 0.2], [[2, math.pi/2], [1.3, math.pi/2]])
    print(particle_set)
    print(robot_pos)
