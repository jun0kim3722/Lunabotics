# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 16:58:58 2023

@author: c5all
"""

import numpy as np
from scipy.stats import multivariate_normal

"""
p = particle set
N = number of particles
x = state vector
z = observation set
t = time
h = observation model
f = state transition model
u = control inputs
w = weight
Q = covariance matrix of process noise (uncertainty in state transition model)
R = covariance matrix of observation noise (uncertainty in observation model)
H = jacobian that linearizes the measurement model around the particle estimate
"""

def calc_weight(N, t, z, h, p):
    w = np.ones(N) / N
    for i in range(N):
        likelihood = np.exp(-0.5 * (z[t] - h(p[i, t, :], np.random.normal(0, 0.1)))**2)
        w[i, t] = w[i, t-1] * likelihood / np.sum(w[:, t-1] * likelihood)
    return w

# State transition function
def f(x_prev, u, w):
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[0.5], [1]])
    f = np.dot(A, x_prev) + np.dot(B, u) + w
    return f

# Observation function
def h(x, v):
    C = np.array([1, 0])
    h = np.dot(C, x) + v
    return h

# Jacobian calculation function
def calc_jacobian(x, h):
    num_diff = 1e-6
    N = x.shape[0]
    m = h(x).shape[0]
    H = np.zeros((m,N))
    for i in range(N):
        x_i = x.copy()
        x_i[i] += num_diff
        H[:,i] = (h(x_i) - h(x)) / num_diff
    return H

# Calculates covariance matrix for process noise
def calc_covariance_Q(x,w):
    Q = np.cov(x, aweights=w)
    return Q

# Calculates covariance matrix for observation noise
def calc_covariance_R(z):
    z_mean = np.mean(z, axis=1)
    z_diff = z - z_mean.reshape(-1, 1)
    R = np.cov(z_diff)
    return R

# Particle filter algorithm
def particle_update(p, N, u, t, h, f, Q, R, H):
    if t == 0:
        for i in range(N):
            p[i,0] = np.array([0., 0., 0.])
            p[i,1:-1] = np.random.normal(0., 1., size=(p.shape[1]-2)//2)
            p[i,-1] = 1./N
    
    # Predict step
    for i in range(N):
        p[i,0:3] = f(p[i,0:3], p[i,3:])
        p[i,3:] = np.random.multivariate_normal(np.zeros(Q.shape[0]), Q)
    
    # Update step
    for obs in u:
        for i in range(N):
            # Compute weight for particle i
            z_hat = h(p[i,0:3])
            J = H(p[i,0:3])
            S = np.dot(np.dot(J, p[i,3:-1].reshape(-1,2)), J.T) + R
            K = np.dot(np.dot(p[i,3:-1].reshape(-1,2), J.T), np.linalg.inv(S))
            p[i,-1] = multivariate_normal.pdf(obs, mean=z_hat, cov=S) # likelihood
            p[i,3:-1] = np.reshape(p[i,3:-1].reshape(-1,2) + np.dot(K, (obs - z_hat).reshape(-1,1)).flatten(), (-1,)) # update landmark estimates
        
        # Resample particles
        w = p[:,-1]
        w /= np.sum(w)
        new_particles = np.zeros_like(p)
        idx = np.random.choice(range(N), size=N, replace=True, p=w)
        new_particles[:,:] = p[idx,:]
        p = []
        p[:,:] = new_particles
    
    return p

    