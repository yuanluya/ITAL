import numpy as np
import tensorflow as tf
from easydict import EasyDict as edict
import copy

from utils import rvs

import pdb

class Teacher:
    def __init__(self, config):
        self.config_ = config
        if self.config_.loss_type == 0:
            self.gt_w_ = np.random.uniform(low = -1, high = 1,
                                       size = [1, self.config_.data_dim + 1])
            self.data_pool_full_ = np.random.uniform(low = -1, high = 1,
                                                size = [self.config_.data_pool_size, self.config_.data_dim])
            self.data_pool_full_ = np.concatenate([self.data_pool_full_, np.ones([self.config_.data_pool_size, 1])], 1)
            self.gt_y_full_ = np.sum(self.data_pool_full_ * self.gt_w_, 1)
            self.gt_loss_full_ = np.zeros(self.config_.data_pool_size)
        else:
            pos_mean = np.random.uniform(low = -1, high = 1, size = self.config_.data_dim)
            neg_mean = np.random.uniform(low = -1, high = 1, size = self.config_.data_dim)
            midpoint = 0.5 * (pos_mean + neg_mean)
            self.gt_w_ = np.ones(shape = [1, self.config_.data_dim + 1])
            self.gt_w_[0, 0: -1] = pos_mean - neg_mean
            self.gt_w_[0, -1] = -1 * np.sum(midpoint * (pos_mean - neg_mean))
            self.data_pool_pos_ = np.random.normal(loc = pos_mean, scale = 1.0,
                                                   size = [int(0.5 * self.config_.data_pool_size), self.config_.data_dim])
            self.data_pool_neg_ = np.random.normal(loc = neg_mean, scale = 1.0,
                                                   size = [int(0.5 * self.config_.data_pool_size), self.config_.data_dim])
            self.data_pool_full_ = np.concatenate([np.concatenate([self.data_pool_pos_, self.data_pool_neg_], 0),
                                             np.ones([self.config_.data_pool_size, 1])], 1)
            self.gt_y_full_ = np.ones(self.config_.data_pool_size)
            self.gt_y_full_[int(0.5 * self.config_.data_pool_size): ] *= -1
            if self.config_.loss_type == 1:
                self.gt_loss_full_ = np.log(1 + np.exp(-1 * self.gt_y_full_ * np.sum(self.data_pool_full_ * self.gt_w_, 1)))
            elif self.config_.loss_type == 2:
                self.gt_loss_full_ = np.maximum(0, 1 - self.gt_y_full_ * np.sum(self.data_pool_full_ * self.gt_w_, 1))
        if self.config_.transform:
            self.t_mat_ = rvs(self.config_.data_dim)
            self.data_pool_tea_full_ = np.matmul(self.data_pool_full_[:, 0: -1], self.t_mat_.T)
            self.data_pool_tea_full_ = np.concatenate([self.data_pool_tea_full_, np.ones([self.config_.data_pool_size, 1])], 1)
            self.gt_w_tea_ = np.concatenate([np.matmul(self.gt_w_[:, 0: -1], self.t_mat_.T),
                                            self.gt_w_[0, -1] * np.ones([1, 1])], 1)
            self.init_w_tea_ = np.random.uniform(low = -1, high = 1,
                                size = [1, self.config_.data_dim + 1])
            self.stu_current_w_ = copy.deepcopy(self.init_w_tea_)
    
    def choose(self, gradients, prev_w, lr):
        vals = np.sum(lr * lr * np.square(gradients), 1) - 2 * lr * np.sum((prev_w - self.gt_w_) * gradients, 1)
        return np.argmin(vals)

    def choose_sur(self, gradients, prev_losses, lr):
        vals = np.sum(lr * lr * np.square(gradients), 1) - 2 * lr * (prev_losses - self.gt_loss_)
        return np.argmin(vals)

    def choose_imit(self, gradients, lr):
        vals = np.sum(lr * lr * np.square(gradients), 1) - 2 * lr * np.sum((self.stu_current_w_ - self.gt_w_tea_) * gradients, 1)
        return np.argmin(vals)
    
    def reset(self):
        self.stu_current_w_ = self.init_w_tea_

    def update_stu_est(self, stu_gt_w, num = 10):
        for i in range(num):
            idx = np.random.randint(self.data_pool_full_.shape[0])
            self.stu_current_w_ -= self.config_.lr * (np.sum(self.data_pool_tea_full_[idx: idx + 1, :] * self.stu_current_w_) -\
                                                      np.sum(self.data_pool_full_[idx: idx + 1, :] * stu_gt_w)) * self.data_pool_tea_full_[idx: idx + 1, :]
    
    def sample(self):
        indices = np.random.choice(self.data_pool_full_.shape[0], self.config_.sample_size)
        self.data_pool_ = self.data_pool_full_[indices, :]
        self.gt_y_ = self.gt_y_full_[indices]
        self.gt_loss_ = self.gt_loss_full_[indices]
        if self.config_.transform:
            self.data_pool_tea_ = self.data_pool_tea_full_[indices, :]

def main():
    config = edict({'data_pool_size': 30, 'data_dim': 10})
    teacher = Teacher(config)
    print(teacher.gt_y_full_)

if __name__ == '__main__':
    main()