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
            mean = np.random.uniform(low = -1, high = 1, size = self.config_.data_dim)
            self.means_.append(mean)
        if self.config_.data_x is not None:
            print('<External data and label>')
            self.data_pool_full_ = np.concatenate([self.config_.data_x, np.ones([self.config_.data_x.shape[0], 1])], 1)
            self.gt_y_full_ = self.config_.data_y

            self.data_pool_full_test_ = np.concatenate([self.config_.test_x, np.ones([self.config_.test_x.shape[0], 1])], 1)
            self.gt_y_full_test_ = self.config_.test_y
            if self.config_.task == 'classification':
                self.gt_y_label_full_ = np.argmax(self.config_.data_y, 1)
                self.gt_y_label_full_test_ = np.argmax(self.config_.test_y, 1)
        elif self.config_.task == 'classification':
            self.data_pool_full_ = []
            self.data_pool_full_test_ = []
            self.gt_y_full_ = []
            self.gt_y_full_test_ = []
            self.gt_y_label_full_ = []
            self.gt_y_label_full_test_ = []
            for i in range(self.config_.num_classes):
                data_points = np.random.normal(loc = self.means_[i], scale = 0.5,
                                               size = [self.config_.data_pool_size_class, self.config_.data_dim])
                self.data_pool_full_.append(data_points)
                labels = np.zeros([self.config_.data_pool_size_class, self.config_.num_classes])
                labels[:, i] = np.ones(self.config_.data_pool_size_class)
                self.gt_y_full_.append(labels)
                self.gt_y_label_full_.append(i * np.ones(self.config_.data_pool_size_class))
                test_points = np.random.normal(loc = self.means_[i], scale = 0.5,
                                               size = [self.config_.data_pool_size_class, self.config_.data_dim])
                self.data_pool_full_test_.append(test_points)
                self.gt_y_full_test_.append(labels)
                self.gt_y_label_full_test_.append(i * np.ones(self.config_.data_pool_size_class))

            self.data_pool_full_ = np.concatenate([np.concatenate(self.data_pool_full_, 0),
                                             np.ones([self.config_.data_pool_size_class * self.config_.num_classes, 1])], 1)
            self.gt_y_full_ = np.concatenate(self.gt_y_full_)
            self.gt_y_full_test_ = np.concatenate(self.gt_y_full_test_)
            self.gt_y_label_full_ = np.concatenate(self.gt_y_label_full_)
            self.data_pool_full_test_ = np.concatenate([np.concatenate(self.data_pool_full_test_, 0),
                                             np.ones([self.config_.data_pool_size_class * self.config_.num_classes, 1])], 1)
            self.gt_y_label_full_test_ = np.concatenate(self.gt_y_label_full_test_)
        else:
            self.gt_w_ = np.random.uniform(low = -1, high = 1,
                                       size = [self.config_.num_classes, self.config_.data_dim + 1])
            self.data_pool_full_ = np.random.uniform(low = -1, high = 1,
                                    size = [self.config_.data_pool_size_class * self.config_.num_classes, self.config_.data_dim])
            self.data_pool_full_ = np.concatenate([self.data_pool_full_, np.ones([self.data_pool_full_.shape[0], 1])], 1)
            self.gt_y_full_ = np.matmul(self.data_pool_full_, self.gt_w_.T)
            
            train_size = int(0.5 * self.data_pool_full_.shape[0])
            self.data_pool_full_test_ = self.data_pool_full_[train_size:, ...]
            self.gt_y_full_test_ = self.gt_y_full_[train_size:, ...]
            self.data_pool_full_ = self.data_pool_full_[0: train_size, ...]
            self.gt_y_full_ = self.gt_y_full_[0: train_size, ...]

        if self.config_.gt_w is None:
            if self.config_.task == 'classification':
                self.clf_ = LogisticRegression(random_state = 0, fit_intercept = False, max_iter = 5000, solver = 'sag')
                self.clf_.fit(self.data_pool_full_, self.gt_y_label_full_)
                self.gt_w_ = self.clf_.coef_
        else:
            print('<External gt weights (included bias)>')
            self.gt_w_ = self.config_.gt_w
        self.linear_vals_ = np.matmul(self.data_pool_full_, self.gt_w_.T)
        if self.config_.task == 'classification':
            softmax = np.exp(self.linear_vals_)
            self.softmax_ = softmax / np.sum(softmax, 1, keepdims = True)
            self.gt_loss_full_ = -1 * np.sum(self.gt_y_full_ * np.log(self.softmax_ + 1e-6), 1)
        else:
            self.gt_loss_full_ = np.zeros(self.data_pool_full_.shape[0])
        if self.config_.transform:
            if self.config_.data_x_tea is None:
                self.t_mat_ = rvs(self.config_.data_dim)
                self.data_pool_full_tea_ = np.matmul(self.data_pool_full_[:, 0: -1], self.t_mat_.T)
                self.data_pool_full_tea_ = np.concatenate([self.data_pool_full_tea_, np.ones([self.data_pool_full_tea_.shape[0], 1])], 1)
                self.gt_w_tea_ = np.concatenate([np.matmul(self.gt_w_[:, 0: -1], self.t_mat_.T),
                                                self.gt_w_[:, -1:] * np.ones([self.config_.num_classes, 1])], 1)
            else:
                self.data_pool_full_tea_ = np.concatenate([self.config_.data_x_tea, np.ones([self.config_.data_x_tea.shape[0], 1])], 1)
                self.gt_y_full_tea_ = self.config_.data_y_tea
                self.gt_y_label_full_tea_ = np.argmax(self.config_.data_y_tea, 1)

                self.data_pool_full_test_tea_ = np.concatenate([self.config_.test_x_tea, np.ones([self.config_.test_x_tea.shape[0], 1])], 1)
                self.gt_y_full_test_tea_ = self.config_.test_y_tea
                self.gt_y_label_full_test_tea_ = np.argmax(self.config_.test_y_tea, 1)
                self.gt_w_tea_ = self.config_.gt_w_tea
            
                self.linear_vals_tea_ = np.matmul(self.data_pool_full_tea_, self.gt_w_tea_.T)
                if self.config_.task == 'classification':
                    softmax = np.exp(self.linear_vals_tea_)
                    self.softmax_tea_ = softmax / np.sum(softmax, 1, keepdims = True)
                    self.gt_loss_full_ = -1 * np.sum(self.gt_y_full_tea_ * np.log(self.softmax_tea_ + 1e-6), 1)

            self.data_pool_tea_ = None
        self.data_pool_ = None
        self.gt_y_ = None
        self.gt_loss_ = None
        self.indices_ = []

    def choose(self, gradients, prev_w, lr, hard = True):
        vals = -1 * self.config_.beta * (np.sum(lr * lr * np.square(gradients), axis = (1, 2)) - 2 * lr * np.sum((prev_w - self.gt_w_) * gradients, axis = (1, 2)))
        if hard:
            return np.argmax(vals)
        vals -= np.max(vals)
        logits = np.exp(vals)
        selected = np.random.choice(len(gradients), 1, p = logits / np.sum(logits))[0]
        # return np.argmin(vals)
        return selected

    def choose_sur(self, gradients, prev_losses, lr, hard = True):
        vals = -1 * self.config_.beta * (np.sum(lr * lr * np.square(gradients), axis = (1, 2)) - 2 * lr * (prev_losses - self.gt_loss_))
        if hard:
            return np.argmax(vals)
        vals -= np.max(vals)
        logits = np.exp(vals)
        selected = np.random.choice(len(gradients), 1, p = logits / np.sum(logits))[0]
        # return np.argmin(vals)
        return selected

    def sample(self, step = None, save = False):
        if step is not None:
            indices = self.indices_[step]
        else:
            indices = np.random.choice(self.data_pool_full_.shape[0], self.config_.sample_size)
        self.data_pool_ = self.data_pool_full_[indices, :]
        self.gt_y_ = self.gt_y_full_[indices, :]
        self.gt_loss_ = self.gt_loss_full_[indices]
        if self.config_.transform:
            self.data_pool_tea_ = self.data_pool_full_tea_[indices, :]
        if save:
            self.indices_.append(indices)

def main():
    config = edict({'data_pool_full_size_class': 15, 'data_dim': 10, 'num_classes': 3, 'transform': True})
    teacher = TeacherM(config)
    print(teacher.gt_y_label_full_)
    pdb.set_trace()

if __name__ == '__main__':
    main()
