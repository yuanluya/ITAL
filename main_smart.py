from easydict import EasyDict as edict
import tensorflow as tf
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

from learner import Learner
from learner_smart import LearnerS
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
    config_L = edict({'particle_num': 1000, 'data_dim': 10, 'reg_coef': 0, 'lr': lr})
    init_ws = np.concatenate([np.random.uniform(-1, 1, size = [config_L.particle_num, config_L.data_dim]),
                              np.zeros([config_L.particle_num, 1])], 1)
    init_w = np.mean(init_ws, 0, keepdims = True)
    teacher = Teacher(config_T)
    learner = Learner(sess, init_w, 0, config_L)
    learnerS = LearnerS(config_L)
    init = tf.global_variables_initializer()
    sess.run(init)

    [w] = sess.run([learner.w_])
    dists0 = [np.sum(np.square(w - teacher.gt_w_))]
    for _ in tqdm(range(10000)):
        data_point = [teacher.data_pool_, teacher.gt_y_]
        w = learner.learn(data_point)
        dists0.append(np.sum(np.square(w - teacher.gt_w_)))
    
    line_neg1, = plt.plot(dists0, label = 'batch')

    sess.run(init)
    [w] = sess.run([learner.w_])
    dists0 = [np.sum(np.square(w - teacher.gt_w_))]
    for _ in tqdm(range(10000)):
        data_idx = np.random.randint(config_T.data_pool_size)
        data_point = [teacher.data_pool_[data_idx: data_idx + 1], teacher.gt_y_[data_idx: data_idx + 1]]
        w = learner.learn(data_point)
        dists0.append(np.sum(np.square(w - teacher.gt_w_)))
    
    line0, = plt.plot(dists0, label = 'sgd')

    sess.run(init)
    [w] = sess.run([learner.w_])
    dists1 = [np.sum(np.square(w - teacher.gt_w_))]
    data_choices1 = []
    for i in tqdm(range(10000)):
        gradients = learner.get_grads(teacher.data_pool_, teacher.gt_y_)
        data_idx = teacher.choose(gradients, w)
        data_choices1.append(data_idx)
        data_point = [teacher.data_pool_[data_idx: data_idx + 1], teacher.gt_y_[data_idx: data_idx + 1]]
        w, best_idx = learner.learn(data_point, gradients)
        dists1.append(np.sum(np.square(w - teacher.gt_w_)))
    line1, = plt.plot(dists1, label = 'regular')
    

    learnerS.reset(init_ws)
    w = learnerS.current_mean_
    dists2 = [np.sum(np.square(w - teacher.gt_w_))]
    data_choices2 = []
    for i in tqdm(range(10000)):
        gradients = learnerS.get_grads(teacher.data_pool_, teacher.gt_y_)
        data_idx = teacher.choose(gradients, w)
        data_choices2.append(data_idx)
        data_point = [teacher.data_pool_[data_idx: data_idx + 1], teacher.gt_y_[data_idx: data_idx + 1]]
        w, _ = learnerS.learn(teacher.data_pool_, teacher.gt_y_, data_idx, gradients)
        dists2.append(np.sum(np.square(w - teacher.gt_w_)))
    line2, = plt.plot(dists2, label = 'smart')


    learnerS.reset(init_ws)
    w = learnerS.current_mean_
    dists3 = [np.sum(np.square(w - teacher.gt_w_))]
    data_choices3 = []
    eliminates = []
    for i in tqdm(range(10000)):
        gradients = learnerS.get_grads(teacher.data_pool_, teacher.gt_y_)
        data_idx = teacher.choose(gradients, w)
        data_choices3.append(data_idx)
        data_point = [teacher.data_pool_[data_idx: data_idx + 1], teacher.gt_y_[data_idx: data_idx + 1]]
        w, eliminate = learnerS.learn(teacher.data_pool_, teacher.gt_y_, data_idx, gradients, True)
        eliminates.append(eliminate)
        dists3.append(np.sum(np.square(w - teacher.gt_w_)))
    line3, = plt.plot(dists3, label = 'smarter')

    plt.legend([line_neg1, line0, line1, line2, line3],
               ['batch', 'sgd', 'machine teaching: %d' % np.unique(data_choices1).shape[0],\
                'compare: %d' % np.unique(data_choices2).shape[0], 'pragmatic: %d' % np.unique(data_choices3).shape[0]], prop={'size': 12})
    plt.show()
    pdb.set_trace()

if __name__ == '__main__':
    main()
