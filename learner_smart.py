import copy
import numpy as np
import tensorflow as tf
from easydict import EasyDict as edict

import pdb

class LearnerS:
    def __init__(self, sess, config):
        all_types = ['RR', 'LR', 'SVM']
        self.sess_ = sess
        self.config_ = config
        self.loss_type_ = all_types[self.config_.loss_type]
        self.use_tf_ = (self.loss_type_ != 'RR')
        self.particles_ = np.random.uniform(-1, 1, size = [self.config_.particle_num, self.config_.data_dim + 1])
        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)

        self.X_ = tf.placeholder(dtype = tf.float32, shape = [None, self.config_.data_dim + 1])
        self.y_ = tf.placeholder(dtype = tf.float32, shape = [None, 1])
        self.W_ = tf.placeholder(dtype = tf.float32, shape = [None, self.config_.data_dim + 1])

        self.linear_val_ = tf.matmul(self.X_, tf.transpose(self.W_))
        if self.loss_type_ == 'RR':
            self.losses_ = tf.square(self.linear_val_ - self.y_)
            self.loss_ = 0.5 * tf.reduce_sum(self.losses_ + 0.5 * self.config_.reg_coef * tf.reduce_sum(tf.square(self.W_[:, 0: -1]), 1))
        elif self.loss_type_ == 'LR':
            self.losses_ = tf.log(1 + tf.exp(-1 * self.y_ * self.linear_val_))
            self.loss_ = tf.reduce_sum(self.losses_ + 0.5 * self.config_.reg_coef * tf.reduce_sum(tf.square(self.W_[:, 0: -1]), 1))
        elif self.loss_type_ == 'SVM':
            self.losses_ = tf.maximum(1 - self.y_ * self.linear_val_, 0)
            self.loss_ = tf.reduce_sum(self.losses_ + 0.5 * self.config_.reg_coef * tf.reduce_sum(tf.square(self.W_[:, 0: -1]), 1))

        self.gradient_w_ = tf.gradients(self.loss_, [self.W_])
        self.gradient_lv_ = tf.gradients(self.loss_, [self.linear_val_])

    def reset(self, init_ws):
        self.particles_ = copy.deepcopy(init_ws)
        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)

    def learn(self, data_pool, data_y, data_idx, gradients, step, random_prob = None):
        move_dists = []
        if not self.use_tf_:
            for i in range(self.config_.particle_num):
                gradient = np.matmul(data_pool[data_idx: data_idx + 1, :].T,
                                    (np.sum(data_pool[data_idx: data_idx + 1, :] * self.particles_[i: i + 1, :], 1, keepdims = True) - data_y[data_idx]))
                self.particles_[i, :] -= self.config_.lr * gradient[:, 0]
                move_dists.append(np.sum(np.square(gradient[:, 0])))
        else:
            gradient_tf = self.sess_.run(self.gradient_w_, {self.X_: data_pool[data_idx: data_idx + 1, :],
                                                            self.W_: self.particles_,
                                                            self.y_: data_y[data_idx] * np.ones([self.config_.particle_num, 1], dtype = np.float32)})
            self.particles_ -= self.config_.lr * gradient_tf[0]
            move_dists = np.sum(np.square(gradient_tf[0]), axis = 1)
        eliminate = 0
        #if smarter:
        gradient = gradients[data_idx: data_idx + 1, :]
        target_center = self.current_mean_ - self.config_.lr * gradient
        val_target = self.config_.lr * self.config_.lr * np.sum(np.square(gradient)) -\
                            2 * self.config_.lr * np.sum((self.current_mean_ - self.particles_) * gradient, 1)
        to_be_replaced = []
        gradients_cache = self.config_.lr * self.config_.lr * np.sum(np.square(gradients), 1)
        for i in range(self.config_.particle_num):
            if random_prob is not None:
                rd = np.random.choice(2, p = [1 - random_prob, random_prob])
                if rd == 1:
                    to_be_replaced.append(i)
                    eliminate += 1
                continue
            particle_cache = self.current_mean_ - self.particles_[i: i + 1, :]
            for j in range(data_pool.shape[0]):
                if j != data_idx:
                    val_cmp = gradients_cache[j] - 2 * self.config_.lr * np.sum(particle_cache * gradients[j, :])
                    if val_cmp < val_target[i]:
                        to_be_replaced.append(i)
                        eliminate += 1
                        break
        to_be_kept = list(set(range(0, self.config_.particle_num)) - set(to_be_replaced))
        if len(to_be_replaced) > 0:
            if len(to_be_kept) > 0 and step > 10:
                new_center = self.config_.target_ratio * target_center + self.config_.new_ratio *\
                             np.mean(self.particles_[np.array(to_be_kept), ...], axis = 0, keepdims = True)
            else:
                new_center = target_center
        for i in to_be_replaced:
            noise = np.random.normal(scale = self.config_.noise_scale, size = [1, self.config_.data_dim + 1])
            self.particles_[i: i + 1, :] = new_center + noise
        #pdb.set_trace()
        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)

        return self.current_mean_, eliminate

    def learn_sur(self, data_pool, data_y, data_idx, gradients, prev_loss, step):
        new_particle_losses = []
        if not self.use_tf_:
            move_dists = []
            for i in range(self.config_.particle_num):

                gradient = np.matmul(data_pool[data_idx: data_idx + 1, :].T,
                                    (np.sum(data_pool[data_idx: data_idx + 1, :] * self.particles_[i: i + 1, :], 1, keepdims = True) - data_y[data_idx]))
                self.particles_[i, :] -= self.config_.lr * gradient[:, 0]
                new_particle_losses.append(0.5 * np.square(np.sum(data_pool * self.particles_[i: i + 1, :], 1) - data_y))
                move_dists.append(np.sum(np.square(gradient[:, 0])))
        else:
            gradient_tf = self.sess_.run(self.gradient_w_, {self.X_: data_pool[data_idx: data_idx + 1, :],
                                                            self.W_: self.particles_,
                                                            self.y_: data_y[data_idx] * np.ones([self.config_.particle_num, 1], dtype = np.float32)})
            min_idx = np.argmin(np.sum(np.square(gradient_tf[0]), axis = 1))
            self.particles_ -= self.config_.lr * gradient_tf[0]
            for i in range(self.config_.particle_num):
                losses = self.sess_.run(self.losses_, {self.X_: data_pool,
                                                       self.W_: self.particles_[i: i + 1, :],
                                                       self.y_: np.expand_dims(data_y, 1)})
                new_particle_losses.append(losses[:, 0])
            move_dists = np.sum(np.square(gradient_tf[0]), axis = 1)
        eliminate = 0
        #if smarter:
        gradient = gradients[data_idx: data_idx + 1, :]
        target_center = self.current_mean_ - self.config_.lr * gradient
        val_target = self.config_.lr * self.config_.lr * np.sum(np.square(gradient))
        to_be_replaced = []
        gradient_cache = self.config_.lr * self.config_.lr * np.sum(np.square(gradients), 1)
        for i in range(self.config_.particle_num):
            val_target_temp = val_target - 2 * self.config_.lr * (prev_loss[data_idx] - new_particle_losses[i][data_idx])
            val_cmps = gradient_cache - 2 * self.config_.lr * (prev_loss - new_particle_losses[i])
            for j in range(data_pool.shape[0]):
                if j != data_idx and val_cmps[j] < val_target_temp:
                    to_be_replaced.append(i)
                    eliminate += 1
                    break

        to_be_kept = list(set(range(0, self.config_.particle_num)) - set(to_be_replaced))
        if len(to_be_replaced) > 0:
            if len(to_be_kept) > 0 and step > 10:
                new_center = self.config_.target_ratio * target_center + self.config_.new_ratio *\
                             np.mean(self.particles_[np.array(to_be_kept), ...], axis = 0, keepdims = True)
            else:
                new_center = target_center
        for i in to_be_replaced:
            noise = np.random.normal(scale = self.config_.noise_scale, size = [1, self.config_.data_dim + 1])
            self.particles_[i: i + 1, :] = new_center + noise
        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)

        return self.current_mean_, eliminate

    def get_grads(self, data_pool, data_y, w_param = None):
        gradients = []
        gradient_tfs = []
        losses = []
        if w_param is None:
            w_param = self.current_mean_
        for i in range(data_pool.shape[0]):
            if not self.use_tf_:
                diff = np.sum(data_pool[i: i + 1, :] * w_param, 1, keepdims = True) - data_y[i]
                gradient = np.matmul(data_pool[i: i + 1, :].T, diff)
                gradients.append(gradient)
                losses.append(0.5 * np.square(diff[0, 0]))
            else:
                gradient_tf, loss = self.sess_.run([self.gradient_w_, self.loss_], {self.X_: data_pool[i: i + 1, :],
                                                                self.y_: np.expand_dims(data_y[i: i + 1], 1),
                                                                self.W_: w_param})
                gradient_tfs.append(gradient_tf[0])
                losses.append(loss)

        if not self.use_tf_:
            return np.concatenate(gradients, 1).T, np.array(losses)
        gradient_tf = np.concatenate(gradient_tfs, 0)

        # losses = self.sess_.run(self.losses_, {self.X_: data_pool, self.W_: self.current_mean_, self.y_: np.expand_dims(data_y, 1)})
        return gradient_tf, np.array(losses)


def main():
    return

if __name__ == '__main__':
    main()
