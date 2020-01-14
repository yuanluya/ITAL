import numpy as np
import tensorflow as tf
from easydict import EasyDict as edict

import pdb

class Teacher:
    def __init__(self, config):
        self.config_ = config
        if self.config_.loss_type == 0:
            self.gt_w_ = np.random.uniform(low = -1, high = 1,
                                       size = [1, self.config_.data_dim + 1])
            self.data_pool_ = np.random.uniform(low = -1, high = 1,
                                                size = [self.config_.data_pool_size, self.config_.data_dim])
            self.data_pool_ = np.concatenate([self.data_pool_, np.ones([self.config_.data_pool_size, 1])], 1)
            self.gt_y_ = np.sum(self.data_pool_ * self.gt_w_, 1)
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
    
    def choose(self, gradients, prev_w):
        vals = np.sum(self.config_.lr * self.config_.lr * np.square(gradients), 1) -\
               2 * self.config_.lr * np.sum((prev_w - self.gt_w_) * gradients, 1)
        return np.argmin(vals)

def main():
    config = edict({'data_pool_size': 30, 'data_dim': 10})
    teacher = Teacher(config)
    print(teacher.gt_y_)

if __name__ == '__main__':
    main()