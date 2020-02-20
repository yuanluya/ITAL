import copy
import numpy as np
from scipy.stats import t
import tensorflow as tf
from easydict import EasyDict as edict

import pdb

class LearnerSM:
    def __init__(self, sess, config):
        self.sess_ = sess
        self.config_ = config
        self.particles_ = np.random.uniform(-2, 2, size = [self.config_.particle_num,\
                                            self.config_.num_classes, self.config_.data_dim + 1])
        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)

        self.X_ = tf.placeholder(dtype = tf.float32, shape = [None, self.config_.data_dim + 1])
        self.W_ = tf.placeholder(dtype = tf.float32, shape = [None, self.config_.num_classes, self.config_.data_dim + 1])

        self.linear_val_ = tf.reduce_sum(tf.expand_dims(self.X_, 1) * self.W_, 2)
        if self.config_.task == 'classification':
            self.y_ = tf.placeholder(dtype = tf.int32, shape = [None, self.config_.num_classes])
            self.probs_ = tf.nn.softmax(self.linear_val_)
            self.losses_ = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(self.y_, self.linear_val_)
        else:
            self.y_ = tf.placeholder(dtype = tf.float32, shape = [None, self.config_.num_classes])
            self.losses_ = 0.5 * tf.reduce_sum(tf.square(self.linear_val_ - self.y_), axis = 1)
        self.loss_ = tf.reduce_sum(self.losses_ + 0.5 * self.config_.reg_coef * tf.reduce_sum(tf.square(self.W_[:, :, 0: -1]), axis = (1, 2)))

        self.gradient_w_ = tf.gradients(self.loss_, [self.W_])
        self.gradient_lv_ = tf.gradients(self.loss_, [self.linear_val_])

    def reset(self, init_ws):
        self.particles_ = copy.deepcopy(init_ws)
        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)

    def learn(self, data_pool, data_y, data_idx, gradients, step, random_prob = None):
        gradient_tf = self.sess_.run(self.gradient_w_, {self.X_: data_pool[data_idx: data_idx + 1, ...],
                                                        self.W_: self.particles_,
                                                        self.y_: data_y[data_idx: data_idx + 1, :]})
        self.particles_ -= self.config_.lr * gradient_tf[0]
        move_dists = np.sum(np.square(gradient_tf[0]), axis = (1, 2))
        eliminate = 0
        #if smarter:
        gradient = gradients[data_idx: data_idx + 1, ...]
        target_center = self.current_mean_ - self.config_.lr * gradient
        val_target = self.config_.lr * self.config_.lr * np.sum(np.square(gradient)) -\
                            2 * self.config_.lr * np.sum((self.current_mean_ - self.particles_) * gradient, axis = (1, 2))

        gradients_cache = self.config_.lr * self.config_.lr * np.sum(np.square(gradients), axis = (1, 2))
        scale = self.config_.noise_scale_min + (self.config_.noise_scale_max - self.config_.noise_scale_min) *\
                np.exp (-1 * step / self.config_.noise_scale_decay)
        #scale = np.power(0.5, int(1.0 * step / self.config_.noise_scale_decay)) * self.config_.noise_scale_max

        to_be_replaced = []
        for i in range(self.config_.particle_num):
            if random_prob is not None:
                rd = np.random.choice(2, p = [1 - random_prob, random_prob])
                if rd == 1:
                    to_be_replaced.append(i)
                continue
            particle_cache = self.current_mean_ - self.particles_[i: i + 1, ...]
            count = 0
            for j in range(data_pool.shape[0]):
                if j != data_idx:
                    val_cmp = gradients_cache[j] - 2 * self.config_.lr * np.sum(particle_cache * gradients[j: j + 1, ...])
                    if val_target[i] - val_cmp > 1e-8:
                        count += 1
                    if count == self.config_.replace_count:
                        to_be_replaced.append(i)
                        break
        #pdb.set_trace()
        to_be_kept = list(set(range(self.config_.particle_num)) - set(to_be_replaced))
        #min_idx = to_be_kept[np.argmin(np.array(move_dists)[np.array(to_be_kept)])] if len(to_be_kept) > 0 else None
        if len(to_be_replaced) > 0:
            if len(to_be_kept) > 0 and step > 10:
                new_center = self.config_.target_ratio * target_center + self.config_.new_ratio *\
                             np.mean(self.particles_[np.array(to_be_kept), ...], axis = 0, keepdims = True)
                # new_center = self.config_.target_ratio * target_center + self.config_.new_ratio *\
                #              self.particles_[min_idx: min_idx + 1, ...]
            else:
                new_center = target_center

            #scale = 1.1 * abs(new_center - self.current_mean_)
        for i in to_be_replaced:
            noise = np.random.normal(scale = scale,
                                     size = [1, self.config_.num_classes, self.config_.data_dim + 1])
                        # noise = t.rvs(df = 5, scale = scale,
                        #               size = [1, self.config_.num_classes, self.config_.data_dim + 1])
            #rd = np.random.choice(2, p = [1 - replace_ratio, replace_ratio])
            rd = np.random.choice(2, p = [0, 1])
            if rd == 0:
                self.particles_[i: i + 1, ...] += 0 #target_center + (noise if random_prob != 1 else 0)
            else:
                self.particles_[i: i + 1, ...] = new_center + (noise if random_prob != 1 else 0)
            eliminate += 1

        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)
        return self.current_mean_, eliminate

    def learn_sur(self, data_pool, data_y, data_idx, gradients, prev_loss, step):
        new_particle_losses = []
        gradient_tf = self.sess_.run(self.gradient_w_, {self.X_: data_pool[data_idx: data_idx + 1, ...],
                                                        self.W_: self.particles_,
                                                        self.y_: data_y[data_idx: data_idx + 1, :]})

        self.particles_ -= self.config_.lr * gradient_tf[0]
        move_dists = np.sum(np.square(gradient_tf[0]), axis = (1, 2))
        for i in range(self.config_.particle_num):
            losses = self.sess_.run(self.losses_, {self.X_: data_pool,
                                                    self.W_: self.particles_[i: i + 1, ...],
                                                    self.y_: data_y})
            new_particle_losses.append(losses)

        eliminate = 0
        #if smarter:
        gradient = gradients[data_idx: data_idx + 1, ...]
        target_center = self.current_mean_ - self.config_.lr * gradient
        val_target = self.config_.lr * self.config_.lr * np.sum(np.square(gradient))
        scale = self.config_.noise_scale_min + (self.config_.noise_scale_max - self.config_.noise_scale_min) *\
                np.exp (-1 * step / self.config_.noise_scale_decay)
        #scale = np.power(0.5, int(1.0 * step / self.config_.noise_scale_decay)) * self.config_.noise_scale_max
        to_be_replaced = []
        gradient_cache = self.config_.lr * self.config_.lr * np.sum(np.square(gradients), axis = (1, 2))
        for i in range(self.config_.particle_num):
            val_target_temp = val_target - 2 * self.config_.lr * (prev_loss[data_idx] - new_particle_losses[i][data_idx])
            val_cmps = gradient_cache - 2 * self.config_.lr * (prev_loss - new_particle_losses[i])
            count = 0
            for j in range(data_pool.shape[0]):
                if j != data_idx and val_target_temp - val_cmps[j] > 1e-8:
                    count += 1
                if count == self.config_.replace_count:
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
            noise = np.random.normal(scale = scale,
                                     size = [1, self.config_.num_classes, self.config_.data_dim + 1])
                        # noise = t.rvs(df = 5, scale = scale,
                        #               size = [1, self.config_.num_classes, self.config_.data_dim + 1])
            self.particles_[i: i + 1, ...] = new_center + noise

        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)

        return self.current_mean_, eliminate

    def get_grads(self, data_pool, data_y, w_param = None):
        gradients = []
        gradient_tfs = []
        gradient_lvs = []
        losses = []
        if w_param is None:
            w_param = self.current_mean_
        for i in range(data_pool.shape[0]):
            gradient_tf, gradient_lv, loss = self.sess_.run([self.gradient_w_, self.gradient_lv_, self.loss_],
                                                            {self.X_: data_pool[i: i + 1, ...],
                                                             self.y_: data_y[i: i + 1, :],
                                                             self.W_: w_param})
            gradient_tfs.append(gradient_tf[0])
            gradient_lvs.append(gradient_lv[0])
            losses.append(loss)

        gradient_tf = np.concatenate(gradient_tfs, 0)
        gradient_lv = np.concatenate(gradient_lvs, 0)

        # losses = self.sess_.run(self.losses_, {self.X_: data_pool, self.W_: self.current_mean_, self.y_: np.expand_dims(data_y, 1)})
        return gradient_tf, gradient_lv, np.array(losses)


def main():
    return

if __name__ == '__main__':
    main()
