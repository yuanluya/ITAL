from easydict import EasyDict as edict
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from learner import Learner
from teacher import Teacher

import pdb

def main():
    tfconfig = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    tfconfig.gpu_options.allow_growth = True
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    sess = tf.Session(config = tfconfig)
    np.random.seed(425)


    lr = 3e-4
    config_T = edict({'data_pool_size': 100, 'data_dim': 10, 'lr': lr})
    config_L = edict({'data_dim': 10, 'reg_coef': 0, 'lr': lr})
    init_w = np.concatenate([np.random.uniform(-1, 1, size = [1, config_L.data_dim]), np.zeros([1, 1])], 1)
    teacher = Teacher(config_T)
    learner = Learner(sess, init_w, 0, config_L)
    init = tf.global_variables_initializer()
    # sess.run(init)

    # [w] = sess.run([learner.w_])
    # dists = [np.sum(np.square(w - teacher.gt_w_))]
    # for i in range(50000):
    #     data_idx = np.random.randint(config_T.data_pool_size)
    #     data_point = [teacher.data_pool_[data_idx: data_idx + 1], teacher.gt_y_[data_idx: data_idx + 1]]
    #     w = learner.learn(data_point)
    #     dists.append(np.sum(np.square(w - teacher.gt_w_)))
    
    # plt.plot(dists)

    sess.run(init)

    [w] = sess.run([learner.w_])
    dists = [np.sum(np.square(w - teacher.gt_w_))]
    for i in range(15000):
        gradients = learner.get_grads(teacher.data_pool_, teacher.gt_y_)
        data_idx = teacher.choose(gradients, w)
        data_point = [teacher.data_pool_[data_idx: data_idx + 1], teacher.gt_y_[data_idx: data_idx + 1]]
        w, best_idx = learner.learn(data_point, gradients)
        dists.append(np.sum(np.square(w - teacher.gt_w_)))
    plt.plot(dists)
    plt.show()
if __name__ == '__main__':
    main()
