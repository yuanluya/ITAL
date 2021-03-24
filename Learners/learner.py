import copy
import numpy as np
from scipy.stats import t
import tensorflow.compat.v1 as tf
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
            self.losses_ = tf.nn.softmax_cross_entropy_with_logits_v2(self.y_, self.linear_val_)
        else:
            self.y_ = tf.placeholder(dtype = tf.float32, shape = [None, self.config_.num_classes])
            self.losses_ = 0.5 * tf.reduce_sum(tf.square(self.linear_val_ - self.y_), axis = 1)
        self.loss_ = tf.reduce_sum(self.losses_ + 0.5 * self.config_.reg_coef * tf.reduce_sum(tf.square(self.W_[:, :, 0: -1]), axis = (1, 2)))

        self.gradient_w_ = tf.gradients(self.loss_, [self.W_])
        self.gradient_lv_ = tf.gradients(self.loss_, [self.linear_val_])
        
        self.particle_weights_ = None

    def reset(self, init_ws):
        self.particles_ = copy.deepcopy(init_ws)
        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)

    def learn_cont(self, data_pool, data_y, data_idx, gradients):
        prev_mean = copy.deepcopy(self.current_mean_)
        exp_cache_prev_func = lambda w_est: -1 * self.config_.beta *\
                                            ((self.config_.lr ** 2) * np.sum(np.square(gradients), axis = (1, 2)) -\
                                            2 * self.config_.lr * np.sum((prev_mean - w_est) * gradients, axis = (1, 2)))
        exp_cache_func = lambda vals: np.exp(vals - np.max(vals))
        teacher_sample_lle_func = lambda exps: np.log((exps / np.sum(exps))[data_idx])
        lle_gradient_func = lambda exps: -2 * self.config_.beta * self.config_.lr * gradients[data_idx: data_idx + 1, ...] +\
                                                2 * self.config_.beta * self.config_.lr * np.sum(gradients *\
                                                np.expand_dims(np.expand_dims(exps, -1), -1) / np.sum(exps), axis = 0, keepdims = True)
        
        gradient_tf = self.sess_.run(self.gradient_w_, {self.X_: data_pool[data_idx: data_idx + 1, ...],
                                                            self.W_: self.current_mean_,
                                                            self.y_: data_y[data_idx: data_idx + 1, :]})
        self.current_mean_ -= self.config_.lr * gradient_tf[0]
        lle_gradient = lle_gradient_func(exp_cache_func(exp_cache_prev_func(self.current_mean_)))
        self.current_mean_ += self.config_.lr * lle_gradient

        return self.current_mean_

    def learn(self, data_pool, data_y, data_idx, gradients):      
        gradient_tf = self.sess_.run(self.gradient_w_, {self.X_: data_pool[data_idx: data_idx + 1, ...],
                                                        self.W_: self.particles_,
                                                        self.y_: data_y[data_idx: data_idx + 1, :]})
        self.particles_ -= self.config_.lr * gradient_tf[0]
        move_dists = np.sum(np.square(gradient_tf[0]), axis = (1, 2))

        gradient = gradients[data_idx: data_idx + 1, ...]
        target_center = self.current_mean_ - self.config_.lr * gradient

        self.particles_[0: 0 + 1, ...] = target_center 
        self.current_mean_ = np.mean(self.particles_, 0, keepdims = True)
        return self.current_mean_

    def learn_sur_cont(self, data_pool, data_y, data_idx, gradients, prev_loss, step):
        exp_cache_prev_func = lambda w_est_loss: -1 * self.config_.beta *\
                                                ((self.config_.lr ** 2) * np.sum(np.square(gradients), axis = (1, 2)) -\
                                                2 * self.config_.lr * (prev_loss - w_est_loss))
        exp_cache_func = lambda vals: np.exp(vals - np.max(vals))
        teacher_sample_lle_func = lambda exps: np.log((exps / np.sum(exps))[data_idx])
        lle_gradient_func = lambda w_est_loss_grad, exps:\
                                -2 * self.config_.beta * self.config_.lr * w_est_loss_grad[data_idx: data_idx + 1, ...] +\
                                2 * self.config_.beta * self.config_.lr * np.sum(w_est_loss_grad *\
                                np.expand_dims(np.expand_dims(exps, -1), -1) / np.sum(exps), axis = 0, keepdims = True)

        current_w_losses = copy.deepcopy(prev_loss)
        current_w_losses_gradient = copy.deepcopy(gradients)
        exp_cache = exp_cache_func(exp_cache_prev_func(current_w_losses))

        self.current_mean_ -= self.config_.lr * current_w_losses_gradient[data_idx: data_idx + 1, ...]
        current_w_losses_gradient, _, current_w_losses = self.get_grads(data_pool, data_y)
        exp_cache = exp_cache_func(exp_cache_prev_func(current_w_losses))

        lle_gradient = lle_gradient_func(current_w_losses_gradient, exp_cache)
        self.current_mean_ += self.config_.lr * lle_gradient
        current_w_losses_gradient, _, _ = self.get_grads(data_pool, data_y)

        self.config_.beta *= np.power(self.config_.beta_decay, step)
        return self.current_mean_


    def get_grads(self, data_pool, data_y, w_param = None):
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

        return gradient_tf, gradient_lv, np.array(losses)


def main():
    return

if __name__ == '__main__':
    main()
