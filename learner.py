import numpy as np
import tensorflow as tf
from easydict import EasyDict as edict

import pdb

class Learner:
    def __init__(self, sess, init_w, config):
        all_types = ['RR', 'LR', 'SVM']
        self.sess_ = sess
        self.init_w_ = init_w
        self.config_ = config
        self.loss_type_ = all_types[self.config_.loss_type]
        self.w_ = tf.Variable(initial_value = self.init_w_, name = 'weight', dtype = tf.float32)
        if (self.config_.task == "classification"):
            self.y_ = tf.compat.v1.placeholder(shape = [None, self.config_.num_classes], dtype = tf.float32)
        else:
            self.y_ = tf.compat.v1.placeholder(shape = [None], dtype = tf.float32)

        self.x_ = tf.compat.v1.placeholder(shape = [None, self.config_.data_dim + 1], dtype = tf.float32)
        #self.linear_val_ = tf.reduce_sum(self.w_ * self.x_, 1)# + self.b_
        if self.config_.task == "classification":
            self.linear_val_ = tf.reduce_sum(tf.expand_dims(self.x_, 1) * self.w_, 2)
            self.probs_ = tf.nn.softmax(self.linear_val_)
            self.losses_ = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(self.y_, self.linear_val_)
            self.loss_ = tf.reduce_mean(self.losses_ + 0.5 * self.config_.reg_coef * tf.reduce_sum(tf.square(self.w_[:, 0: -1])))
            self.gradient_ = tf.gradients(self.loss_, [self.w_])
            self.diff = self.linear_val_ - self.y_
            self.gradient_manual_ = tf.matmul(tf.transpose(self.x_), self.diff)
        else:
            self.linear_val_ = tf.reduce_sum(self.w_ * self.x_, 1)
            self.loss_ = 0.5 * tf.square(self.linear_val_ - self.y_)
            self.loss_ = tf.reduce_mean(self.loss_) + 0.5 * self.config_.reg_coef * tf.reduce_sum(tf.square(self.w_[:, 0: -1]))
            self.gradient_ = tf.gradients(self.loss_, [self.w_])
            self.diff = tf.expand_dims(self.linear_val_ - self.y_, 1)
            self.gradient_manual_ = tf.matmul(tf.transpose(self.x_), self.diff)

        self.opt_ = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = self.config_.lr)
        self.train_op_ = self.opt_.minimize(self.loss_)

    def learn(self, data_point, gradients = None):
        current_w = self.sess_.run([self.w_])
        if gradients is not None:
            self.sess_.run(self.train_op_, {self.x_: data_point[0], self.y_: data_point[1].reshape((1, self.config_.num_classes))})
            [new_w] = self.sess_.run([self.w_])
            vals = []
            for i in range(gradients.shape[0]):
                val = np.sum(self.config_.lr * self.config_.lr * np.square(gradients[i, :])) -\
                      2 * self.config_.lr * np.sum((current_w - new_w) * gradients[i, :])
                vals.append(val)
            return new_w, np.argmin(vals)
        else:
            if (self.config_.task == "classification"):
                self.sess_.run(self.train_op_, {self.x_: data_point[0], self.y_: data_point[1]})
            else:
                self.sess_.run(self.train_op_, {self.x_: data_point[0], self.y_: data_point[1].flatten()})
            [w] = self.sess_.run([self.w_])
            return w

    def get_grads(self, data_pool, data_y):
        gradients = []
        losses = []
        for d in range(data_pool.shape[0]):
            gradient, loss = self.sess_.run([self.gradient_, self.loss_], {self.x_: data_pool[d: d + 1], self.y_: data_y[d: d + 1]})
            gradients.append(gradient)
            losses.append(loss)
        return np.squeeze(np.array(gradients)), np.array(losses)

def main():
    config = edict({'data_dim': 10, 'reg_coef': 0, 'lr': 1e-4})
    learner = Learner(0, config)

if __name__ == '__main__':
    main()
