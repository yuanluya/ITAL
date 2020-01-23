import copy
import numpy as np
import tensorflow as tf
from easydict import EasyDict as edict

import pdb

class LearnerSM:
    def __init__(self, sess, config):
        self.sess_ = sess
        self.config_ = config
        self.particles_ = np.random.uniform(-1, 1, size = [self.config_.particle_num,\
                                            self.config_.num_classes, self.config_.data_dim + 1])
        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)

        self.X_ = tf.placeholder(dtype = tf.float32, shape = [None, self.config_.data_dim + 1])
        self.y_ = tf.placeholder(dtype = tf.int32, shape = [None, self.config_.num_classes])
        self.W_ = tf.placeholder(dtype = tf.float32, shape = [None, self.config_.num_classes, self.config_.data_dim + 1])

        self.linear_val_ = tf.reduce_sum(tf.expand_dims(self.X_, 1) * self.W_, 2)
        self.probs_ = tf.nn.softmax(self.linear_val_)
        self.losses_ = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(self.y_, self.linear_val_)
        self.loss_ = tf.reduce_sum(self.losses_ + 0.5 * self.config_.reg_coef * tf.reduce_sum(tf.square(self.W_[:, :, 0: -1]), axis = (1, 2)))
        
        self.gradient_w_ = tf.gradients(self.loss_, [self.W_])
        self.gradient_lv_ = tf.gradients(self.loss_, [self.linear_val_])

    def reset(self, init_ws):
        self.particles_ = copy.deepcopy(init_ws)
        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)

    def learn(self, data_pool, data_y, data_idx, gradients, random_prob = None):
        gradient_tf = self.sess_.run(self.gradient_w_, {self.X_: data_pool[data_idx: data_idx + 1, ...],
                                                        self.W_: self.particles_,
                                                        self.y_: data_y[data_idx: data_idx + 1, :]})
        self.particles_ -= self.config_.lr * gradient_tf[0]

        eliminate = 0
        #if smarter:
        gradient = gradients[data_idx: data_idx + 1, ...]
        new_center = self.current_mean_ - self.config_.lr * gradient
        val_target = self.config_.lr * self.config_.lr * np.sum(np.square(gradient)) -\
                            2 * self.config_.lr * np.sum((self.current_mean_ - self.particles_) * gradient, axis = (1, 2))
        
        gradients_cache = self.config_.lr * self.config_.lr * np.sum(np.square(gradients), axis = (1, 2))
        for i in range(self.config_.particle_num):
            if random_prob is not None:
                rd = np.random.choice(2, p = [1 - random_prob, random_prob])
                if rd == 1:
                    if random_prob != 1:
                        noise = np.random.normal(scale = 0.05, size = [1, self.config_.num_classes, self.config_.data_dim + 1])
                    self.particles_[i: i + 1, ...] = new_center + (noise if random_prob != 1 else 0)
                    eliminate += 1
                continue
            particle_cache = self.current_mean_ - self.particles_[i: i + 1, ...]
            for j in range(data_pool.shape[0]):
                if j != data_idx:
                    val_cmp = gradients_cache[j] - 2 * self.config_.lr * np.sum(particle_cache * gradients[j: j + 1, ...])
                    if val_cmp < val_target[i]:
                        # rd = np.random.choice(2, p = [0.1, 0.9])
                        # if rd == 1:
                        if True:
                            noise = np.random.normal(scale = 0.05, size = [1, self.config_.num_classes, self.config_.data_dim + 1])
                            self.particles_[i: i + 1, ...] = new_center + noise
                            eliminate += 1
                            break
        
        #pdb.set_trace()
        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)

        return self.current_mean_, eliminate

    def learn_sur(self, data_pool, data_y, data_idx, gradients, prev_loss):
        new_particle_losses = []
        gradient_tf = self.sess_.run(self.gradient_w_, {self.X_: data_pool[data_idx: data_idx + 1, ...],
                                                        self.W_: self.particles_,
                                                        self.y_: data_y[data_idx: data_idx + 1, :]})
        
        self.particles_ -= self.config_.lr * gradient_tf[0]
        for i in range(self.config_.particle_num):
            losses = self.sess_.run(self.losses_, {self.X_: data_pool,
                                                    self.W_: self.particles_[i: i + 1, ...],
                                                    self.y_: data_y})
            new_particle_losses.append(losses)

        eliminate = 0
        #if smarter:
        gradient = gradients[data_idx: data_idx + 1, ...]
        new_center = self.current_mean_ - self.config_.lr * gradient
        val_target = self.config_.lr * self.config_.lr * np.sum(np.square(gradient))
        
        gradient_cache = self.config_.lr * self.config_.lr * np.sum(np.square(gradients), axis = (1, 2))
        for i in range(self.config_.particle_num):
            val_target_temp = val_target - 2 * self.config_.lr * (prev_loss[data_idx] - new_particle_losses[i][data_idx])
            val_cmps = gradient_cache - 2 * self.config_.lr * (prev_loss - new_particle_losses[i])
            for j in range(data_pool.shape[0]):
                if j != data_idx and val_cmps[j] < val_target_temp:
                    noise = np.random.normal(scale = 0.1, size = [1, self.config_.num_classes, self.config_.data_dim + 1])
                    self.particles_[i: i + 1, ...] = new_center + noise
                    eliminate += 1
                    break

        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)

        return self.current_mean_, eliminate

    def get_grads(self, data_pool, data_y, w_param = None):
        gradients = []
        gradient_tfs = []
        losses = []
        if w_param is None:
            w_param = self.current_mean_
        for i in range(data_pool.shape[0]):
            gradient_tf, loss = self.sess_.run([self.gradient_w_, self.loss_], {self.X_: data_pool[i: i + 1, ...],
                                                            self.y_: data_y[i: i + 1, :],
                                                            self.W_: w_param})
            gradient_tfs.append(gradient_tf[0])
            losses.append(loss)

        gradient_tf = np.concatenate(gradient_tfs, 0)

        # losses = self.sess_.run(self.losses_, {self.X_: data_pool, self.W_: self.current_mean_, self.y_: np.expand_dims(data_y, 1)})
        return gradient_tf, np.array(losses)
        

def main():
    return

if __name__ == '__main__':
    main()