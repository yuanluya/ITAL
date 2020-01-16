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
            self.data_pool_ = np.random.uniform(low = -1, high = 1,
                                                size = [self.config_.data_pool_size, self.config_.data_dim])
            if self.config_.transform:
                self.t_mat_ = rvs(self.config_.data_dim)
                self.data_pool_tea_ = np.matmul(self.data_pool_, self.t_mat_.T)
                self.gt_w_tea_ = np.concatenate([np.matmul(self.gt_w_[:, 0: -1], self.t_mat_.T),
                                                 self.gt_w_[0, -1] * np.ones([1, 1])], 1)
                self.init_w_tea_ = np.random.uniform(low = -1, high = 1,
                                       size = [1, self.config_.data_dim + 1])
                self.stu_current_w_ = copy.deepcopy(self.init_w_tea_)
                self.data_pool_tea_ = np.concatenate([self.data_pool_tea_, np.ones([self.config_.data_pool_size, 1])], 1)
            self.data_pool_ = np.concatenate([self.data_pool_, np.ones([self.config_.data_pool_size, 1])], 1)
            self.gt_y_ = np.sum(self.data_pool_ * self.gt_w_, 1)
            self.gt_loss_ = np.zeros(self.config_.data_pool_size)
        else:
            self.gt_w_ = np.ones(shape = [1, self.config_.data_dim + 1])
            self.gt_w_[0, -1] = 0
            self.data_pool_pos_ = np.random.normal(loc = 0.5 * np.ones(self.config_.data_dim), scale = 1.0,
                                                   size = [int(0.5 * self.config_.data_pool_size), self.config_.data_dim])
            self.data_pool_neg_ = np.random.normal(loc = -0.5 * np.ones(self.config_.data_dim), scale = 1.0,
                                                   size = [int(0.5 * self.config_.data_pool_size), self.config_.data_dim])
            self.data_pool_ = np.concatenate([np.concatenate([self.data_pool_pos_, self.data_pool_neg_], 0),
                                             np.ones([self.config_.data_pool_size, 1])], 1)
            self.gt_y_ = np.ones(self.config_.data_pool_size)
            self.gt_y_[int(0.5 * self.config_.data_pool_size): ] *= -1
            if self.config_.loss_type == 1:
                self.gt_loss_ = np.log(1 + np.exp(-1 * self.gt_y_ * np.sum(self.data_pool_ * self.gt_w_, 1)))
            elif self.config_.loss_type == 2:
                self.gt_loss_ = np.maximum(0, 1 - self.gt_y_ * np.sum(self.data_pool_ * self.gt_w_, 1))
    
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

    def update_stu_est(self, data_x, stu_linear_val):
        self.stu_current_w_ -= self.config_.lr * (np.sum(data_x * self.stu_current_w_) - stu_linear_val) * data_x

def main():
    config = edict({'data_pool_size': 30, 'data_dim': 10})
    teacher = Teacher(config)
    print(teacher.gt_y_)

if __name__ == '__main__':
    main()