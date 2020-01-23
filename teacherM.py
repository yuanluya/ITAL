import numpy as np
import tensorflow as tf
from easydict import EasyDict as edict
import copy

from sklearn.linear_model import LogisticRegression
from utils import rvs

import pdb

class TeacherM:
    def __init__(self, config):
        self.config_ = config
        self.means_ = []
        for i in range(self.config_.num_classes):
            mean = np.random.uniform(low = -1 * self.config_.data_dim / 10, high = self.config_.data_dim / 10, size = self.config_.data_dim)
            self.means_.append(mean)
        self.data_pool_ = []
        self.gt_y_ = []
        self.gt_y_label_ = []
        for i in range(self.config_.num_classes):
            data_points = np.random.normal(loc = self.means_[i], scale = 1.0,
                                           size = [self.config_.data_pool_size_class, self.config_.data_dim])
            self.data_pool_.append(data_points)
            labels = np.zeros([self.config_.data_pool_size_class, self.config_.num_classes])
            labels[:, i] = np.ones(self.config_.data_pool_size_class)
            self.gt_y_.append(labels)
            self.gt_y_label_.append(i * np.ones(self.config_.data_pool_size_class))

        self.data_pool_ = np.concatenate([np.concatenate(self.data_pool_, 0),
                                         np.ones([self.config_.data_pool_size_class * self.config_.num_classes, 1])], 1)
        self.gt_y_ = np.concatenate(self.gt_y_)
        self.gt_y_label_ = np.concatenate(self.gt_y_label_)
        self.clf_ = LogisticRegression(random_state = 0, fit_intercept = False, max_iter = 5000, solver = 'sag')
        self.clf_.fit(self.data_pool_, self.gt_y_label_)
        self.gt_w_ = self.clf_.coef_
        self.linear_vals_ = np.matmul(self.data_pool_, self.gt_w_.T)
        softmax = np.exp(self.linear_vals_)
        self.softmax_ = softmax / np.sum(softmax, 1, keepdims = True)
        self.gt_loss_ = -1 * np.sum(self.gt_y_ * np.log(self.softmax_ + 1e-6), 1)
        if self.config_.transform:
            self.t_mat_ = rvs(self.config_.data_dim)
            self.data_pool_tea_ = np.matmul(self.data_pool_[:, 0: -1], self.t_mat_.T)
            self.data_pool_tea_ = np.concatenate([self.data_pool_tea_, np.ones([self.data_pool_tea_.shape[0], 1])], 1)
            self.gt_w_tea_ = np.concatenate([np.matmul(self.gt_w_[:, 0: -1], self.t_mat_.T),
                                            self.gt_w_[:, -1:] * np.ones([self.config_.num_classes, 1])], 1)
    def choose(self, gradients, prev_w, lr):
        vals = np.sum(lr * lr * np.square(gradients), axis = (1, 2)) - 2 * lr * np.sum((prev_w - self.gt_w_) * gradients, axis = (1, 2))
        return np.argmin(vals)

    def choose_sur(self, gradients, prev_losses, lr):
        vals = np.sum(lr * lr * np.square(gradients), axis = (1, 2)) - 2 * lr * (prev_losses - self.gt_loss_)
        return np.argmin(vals)

def main():
    config = edict({'data_pool_size_class': 15, 'data_dim': 10, 'num_classes': 3, 'transform': True})
    teacher = TeacherM(config)
    print(teacher.gt_y_label_)
    pdb.set_trace()

if __name__ == '__main__':
    main()