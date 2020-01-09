import copy
import numpy as np
import tensorflow as tf
from easydict import EasyDict as edict

import pdb

class LearnerS:
    def __init__(self, config):
        self.config_ = config
        self.particles_ = np.random.uniform(-1, 1, size = [self.config_.particle_num, self.config_.data_dim + 1])
        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)

    def reset(self, init_ws):
        self.particles_ = copy.deepcopy(init_ws)
        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)

    def learn(self, data_pool, data_y, data_idx, gradients, smarter = False):
        for i in range(self.config_.particle_num):
            gradient = np.matmul(data_pool[data_idx: data_idx + 1, :].T,
                                 (np.sum(data_pool[data_idx: data_idx + 1, :] * self.particles_[i: i + 1, :], 1, keepdims = True) - data_y[data_idx]))
            self.particles_[i, :] -= self.config_.lr * gradient[:, 0]

        eliminate = 0
        #if smarter:
        gradient = gradients[data_idx: data_idx + 1, :]
        new_center = self.current_mean_ - self.config_.lr * gradient
        val_target = self.config_.lr * self.config_.lr * np.sum(np.square(gradient)) -\
                            2 * self.config_.lr * np.sum((self.current_mean_ - self.particles_) * gradient, 1)
        
        gradients_cache = self.config_.lr * self.config_.lr * np.sum(np.square(gradients), 1)
        for i in range(self.config_.particle_num):
            particle_cache = self.current_mean_ - self.particles_[i: i + 1, :]
            for j in range(data_pool.shape[0]):
                if j != data_idx:
                    val_cmp = gradients_cache[j] - 2 * self.config_.lr * np.sum(particle_cache * gradients[j, :])
                    if not smarter:
                        rd = np.random.choice(2, p = [0.15, 0.85])
                        if rd == 0:
                            break
                    if not smarter or (smarter and val_cmp < val_target[i]):
                        noise = np.random.normal(scale = 0.1, size = [1, self.config_.data_dim + 1])
                        self.particles_[i: i + 1, :] = new_center + noise
                        eliminate += 1
                        break
        

        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)
        return self.current_mean_, eliminate

    def get_grads(self, data_pool, data_y):
        gradients = []
        for i in range(data_pool.shape[0]):
            gradient = np.matmul(data_pool[i: i + 1, :].T,
                                 (np.sum(data_pool[i: i + 1, :] * self.current_mean_, 1, keepdims = True) - data_y[i]))
            gradients.append(gradient)
        return np.concatenate(gradients, 1).T
        

def main():
    return

if __name__ == '__main__':
    main()