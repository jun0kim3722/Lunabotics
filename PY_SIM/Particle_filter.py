import numpy as np
from numpy.random import random
import pdb
from math import pi

class landmark:
    num_land = 0
    l_pos = []
    l_sig = []
    Qt = 0

    def add_landmark(N):
        landmark.num_land += 1
        landmark.l_pos.append(np.zeros((N,2)))
        landmark.l_sig.append(np.zeros((N,2,2)))
    
    def update_landmark(n, idx, l_pos, sig):
        landmark.l_sig[idx][n] = sig
        landmark.l_pos[idx][n] = l_pos.T

    def prev_val (n, idx):
        l_pos = landmark.l_pos[idx][n]
        sig = landmark.l_sig[idx][n]
        return l_pos, sig

class particle_filter:

    def __init__(self, m_sigma, l_sigma, N, map):
        self.prev_R_pos = np.array([])
        self.prev_particle = None
        self.Ct = []
        self.m_sigma = m_sigma
        self.N = N
        self.weight = []
        self.map_size = map
        landmark.Qt = np.array([[l_sigma[0]**2, 0], [0, l_sigma[1]**2]])

    def creating_particles(self, Ut, Zt):

        particle_set = np.zeros((self.N, 3)) #[x, y, theta, Weight]
    
        for n in range(self.N):
            if self.prev_R_pos != []:
                x = np.random.normal(self.prev_particle[n][0] + Ut[0], self.m_sigma[0], 1) # Obtain new x value for new sample
                y = np.random.normal(self.prev_particle[n][1] + Ut[1], self.m_sigma[1], 1) #Obtains new y value for new sample
                theta = np.random.normal((self.prev_particle[n][2] + Ut[2]) % 2, self.m_sigma[2], 1) #Between 0 and 2pi radians //////////////work needed

            else: #starts from uniform distribution
                x = np.random.uniform(0, self.map_size[0], 1)
                y = np.random.uniform(0, self.map_size[1], 1)
                theta = np.random.uniform(0, 2, 1)
                
                if(n == 3):
                    x = np.array([200])
                    y = np.array([200])
                    theta = np.array([0])

            particle = np.concatenate((x, y, theta))
            l_weight = np.zeros(len(Zt))

            for i, z in enumerate(Zt): # Landmark by landmark
                zt = np.array([z]).T

                if self.Ct[i][0]: # Ct never seen before. Ct = [[seen? T or F], [idx of landmark if true]]
                    # add landmark
                    l_idx = self.Ct[i][1]
                    if (n == 0):
                        landmark.add_landmark(self.N)

                    # initialize mean = mu => list of landmarks
                    l_pos = landmark_pos(particle, zt)
                    Z_hat, delta, q = h(particle, l_pos) # Z_hat = (2,1) delta = (2,1)

                    # calculate Jacobian = H 
                    H = calc_jacobian(delta, q)

                    # initialize covariance => list of uncertainty of landmarks
                    sig = np.linalg.inv(H) @ landmark.Qt @ np.linalg.inv(H).T
                        # (2,2) @ (2,2) @ (2,2)

                    # default importance weight
                    Wt = 1 /self.N
                    l_weight[i] = Wt

                    # landmark init
                    landmark.update_landmark(n, l_idx, l_pos, sig)

                else:   #<EKF-update> // update landmark
                    # Get prev value
                    l_idx = self.Ct[i][1]
                    prev_l_pos, prev_sig = landmark.prev_val(n, l_idx)
                    prev_l_pos = prev_l_pos[:, np.newaxis]

                    # measurement prediction = Z_hat
                    Z_hat, delta, q = h(particle, prev_l_pos) # Z_hat = (2,1) delta = (2,1)

                    # calculate Jacobian = H
                    H = calc_jacobian(delta, q) # (2,2)

                    # measurment covariance = Q
                    Q = H @ prev_sig @ H.T + landmark.Qt
                    #   (2,2) @ (2,2) @ (2,2) + (2,2) = (2,2)

                    # calculate Kalman gain = K
                    K = calc_kalmangain(prev_sig, H, Q)
                    #      (2,2) @ (2,2) @ (2,2) = (2,2)

                    # update mean = mu ==> mu + K(Zt - z_hat)
                    l_pos = prev_l_pos + (K @ (zt - Z_hat)) # l_pos = (2,2)

                    # update covariance ==> ()
                    sig = (np.identity(2) - K @ H) @ prev_sig
                         # [(2,2) - (2,2) @ (2,2) ] @ (2,2) = (2,2)

                    # update landmark class
                    landmark.update_landmark(n, l_idx, l_pos, sig)

                    # calc weight
                    Wt = calc_weight(zt, Q, Z_hat) # weight of landamrk[j] with particle[n]
                    l_weight[i] = Wt * 10 ** 5
                    
                    if np.isnan(Wt) or np.isinf(Wt):
                        print("Individual nan or inf")
                        pdb.set_trace()

            if np.isnan(np.prod(l_weight[l_weight > 0.01])) or np.isinf(np.prod(l_weight[l_weight != 0])):
                print("Weight of particle is nan or inf")
                pdb.set_trace()
            
            self.weight.append(np.prod(l_weight[l_weight != 0]))
            # self.weight.append(sum(l_weight))

            particle_set[n] = np.concatenate((x, y, theta))

        # Normalize weight
        if sum(self.weight) != 0:
            a = self.weight / np.sum(self.weight)
            if np.isnan(a).any() or np.isinf(a).any():
                print("Normalize Nan or Inf Error")
                pdb.set_trace()
            else:
                self.weight = a
        else:
            self.weight = [1 /self.N] * self.N #//////need work better normalization needed
        
        if (sum(self.weight) == 0):
            print("ZERO weight")
            pdb.set_trace()
        if (len(self.weight) != self.N):
            print("len weight ERROR")
            pdb.set_trace()

        # Resampling & est position
        if sum(self.weight) != 0:
            print('resampling!!!')
            idx = resampling(self.N, self.weight)
        else:
            idx = range(self.N)
        new_particle = np.array(list(map(lambda x: particle_set[x], idx)))
        particle_bar = new_particle.mean(0)

        np.set_printoptions(suppress = True)
        # print('set', np.array(list(map(lambda x: [round(i,1) for i in x], particle_set))))
        print("set", particle_set)
        print('Weight', self.weight)
        print("idx: ", idx)
        # print("Max_weight", particle_set[np.argmax(self.weight)])
        # print("new_particle", new_particle)

        # pdb.set_trace()
        self.prev_R_pos = particle_bar # update previous pos of robot
        self.prev_particle = new_particle
        self.Ct = []
        self.weight = []

        return particle_set, particle_bar

def landmark_pos(particle, Zt):
    R_x = particle[0] # Robot pos x
    R_y = particle[1] # Robot pos y
    R_th = particle[2] # Robot pos theta

    L_th = Zt[1][0] # Angle btw robot to landmark
    L_d = Zt[0][0] # distance from robot to landmark

    L_pos = np.array([[R_x], [R_y]]) + np.array([[L_d * np.cos(L_th + R_th)], [-L_d * np.sin(L_th + R_th)]]) # negative bc 0,0 is left top

    # pdb.set_trace()

    return L_pos # returning x, y of landmark

# Observation function
def h(particle, L_pos):
    R_x = particle[0]
    R_y = particle[1]
    R_th = particle[2]
    L_x = L_pos[0][0] # landmark x
    L_y = L_pos[1][0] # landmark y

    delta = np.array([[L_x - R_x], [L_y - R_y]])
    q = (delta.T @ delta)[0][0]

    Z_hat = np.array([[np.sqrt(q)], [np.arctan2(delta[1][0], delta[0][0]) - R_th]])

    return Z_hat, delta, q # returning expected observation


def calc_jacobian(delta, q):

    x = np.array([delta[0][0], delta[1][0]])
    y = np.array([-delta[1][0], delta[0][0]])

    H = (1/q) * np.array([np.sqrt(q) * x, y])

    return H

def calc_kalmangain(sig ,H, Q):
    K = sig @ H.T @ np.linalg.inv(Q)

    return K

def calc_weight(Zt, Q, Z_hat):

    Zx = (Zt - Z_hat)
    inv_Q = np.linalg.inv(Q)
    Wt = np.exp(-0.5 * Zx.T @ inv_Q @ Zx) / np.sqrt(2 * pi * np.linalg.det(2 * pi * Q))
    #              0.5 * (1,2) @ (2,2) @ (2,1) = (1,1)

    if (np.isnan(Wt) or np.isinf(Wt)):
        print("Nan or Inf Wt ERROR")
        pdb.set_trace()

    return Wt[0][0]

def resampling(N, weight):
    positions = (random(N) + range(N)) / N
    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weight)
    i , j = 0, 0
    while i < N:
        try:
            a = positions[i] < cumulative_sum[j]
        except IndexError:
                print("Idx Error")
                pdb.set_trace()

        if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
        else:
            j += 1
        
    # pdb.set_trace()
    return indexes





















if __name__ == '__main__':
    # particle = particle_filter([10,10,10], [10,10,10], 10, [100, 100])

    # particle.Ct = [True, True, False]
    # for i in range(3):
    #     particle_set, robot_pos = particle.creating_particles([1, 1, 0.5], [[3,0], [5,2]])


    # for i in range(2):
    #     if i == 0:
    #         x = np.array([210])
    #         y = np.array([200])
    #         theta = np.array([0])

    # a = np.array([1.0, 1.0, 2.0 ,2.0 ,1.0])
    # a /= sum(a)
    # print(a)
    # a = 5
    # print(5 // 2)
    print(np.zeros(5))


