import numpy as np
from scipy.stats import multivariate_normal as mn

import pdb
def rvs(dim=3):
     random_state = np.random
     H = np.eye(dim)
     D = np.ones((dim,))
     for n in range(1, dim):
         x = random_state.normal(size=(dim-n+1,))
         D[n-1] = np.sign(x[0])
         x[0] -= D[n-1]*np.sqrt((x*x).sum())
         # Householder transformation
         Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
         mat = np.eye(dim)
         mat[n-1:, n-1:] = Hx
         H = np.dot(H, mat)
         # Fix the last sign such that the determinant is 1
     D[-1] = (-1)**(1-(dim % 2))*D.prod()
     # Equivalent to np.dot(np.diag(D), H) but faster, apparently
     H = (D*H.T).T
     return H
    
def generate_new_particles(center, w_ref, data_gradients, data_idx, lr, num_need):
    new_particles = []
    gradient = data_gradients[data_idx: data_idx + 1, ...]
    gradients_cache = lr * lr * np.sum(np.square(data_gradients), axis = (1, 2))
    num_trail = 0
    while len(new_particles) < num_need:
        num_trail += 1
        valid = True
        # idx = np.random.randint(old_particles.shape[0])
        particle = mn.rvs(center.flatten(), cov = 0.05)
        particle = particle.reshape((1, center.shape[1], center.shape[2]))
        val_target = gradients_cache[data_idx] - 2 * lr * np.sum((w_ref - particle) * gradient)
        for j in range(data_gradients.shape[0]):
            val_cmp = gradients_cache[j] - 2 * lr * np.sum((w_ref - particle) * data_gradients[j: j + 1, ...])
            if val_cmp < val_target:
                valid = False
                break
        if valid:
            new_particles.append(particle)
    return new_particles
