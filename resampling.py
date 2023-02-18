import random
import numpy as np

def resampling(Xt, Wt):
    X_sm = np.array()
    N = len(Wt)
    positions = (random(N) + range(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(Wt)
    i , j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
        
    return Xt[indexes]

