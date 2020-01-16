from easydict import EasyDict as edict
import tensorflow as tf
import numpy as np
import os
import sys
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
    np.random.seed(1000)

    mode_idx = 1
    modes = ['omni', 'surr', 'imit']
    mode = modes[mode_idx]

    lr = 1e-4
    lt = int(sys.argv[1])
    dd = 10 if lt != 0 else 50
    train_iter_simple = 25000
    train_iter_smart = 2500
    reg_coef = 0 if lt == 0 else 5e-5

    config_T = edict({'data_pool_size': 100, 'data_dim': dd, 'loss_type': lt, 'lr': lr, 'transform': mode == 'imit'})
    config_L = edict({'data_dim': dd, 'reg_coef': reg_coef, 'lr': 10 * lr, 'loss_type': lt})
    config_LS = edict({'particle_num': 1000, 'data_dim': dd, 'reg_coef': reg_coef, 'lr': lr, 'loss_type': lt})
    init_ws = np.concatenate([np.random.uniform(-1, 1, size = [config_LS.particle_num, dd]),
                              np.zeros([config_LS.particle_num, 1])], 1)
    init_w = np.mean(init_ws, 0, keepdims = True)
    teacher = Teacher(config_T)
    learner = Learner(sess, init_w, config_L)
    learnerS = LearnerS(sess, config_LS)
    init = tf.global_variables_initializer()
    sess.run(init)
    
    
    [w] = sess.run([learner.w_])
    dists0 = [np.sum(np.square(w - teacher.gt_w_))]
    for _ in tqdm(range(train_iter_simple)):
        data_point = [teacher.data_pool_, teacher.gt_y_]
        w = learner.learn(data_point)
        dists0.append(np.sum(np.square(w - teacher.gt_w_)))
    
    line_neg1, = plt.plot(dists0, label = 'batch')

    sess.run(init)
    [w] = sess.run([learner.w_])
    dists0 = [np.sum(np.square(w - teacher.gt_w_))]
    for _ in tqdm(range(train_iter_simple)):
        data_idx = np.random.randint(config_T.data_pool_size)
        data_point = [teacher.data_pool_[data_idx: data_idx + 1], teacher.gt_y_[data_idx: data_idx + 1]]
        w = learner.learn(data_point)
        dists0.append(np.sum(np.square(w - teacher.gt_w_)))
    
    line0, = plt.plot(dists0, label = 'sgd')

    sess.run(init)
    [w] = sess.run([learner.w_])
    dists1 = [np.sum(np.square(w - teacher.gt_w_))]
    data_choices1 = []
    for i in tqdm(range(train_iter_simple)):
        gradients, losses = learner.get_grads(teacher.data_pool_, teacher.gt_y_)
        if mode == 'surr':
            data_idx = teacher.choose_sur(gradients, losses, config_L.lr)
        elif mode == 'omni':
            data_idx = teacher.choose(gradients, w, config_L.lr)
        else:
            gradients_tea, _ = learnerS.get_grads(teacher.data_pool_tea_, teacher.gt_y_, teacher.stu_current_w_)
            data_idx = teacher.choose_imit(gradients_tea, config_L.lr)
            stu_linear_val = np.sum(teacher.data_pool_[data_idx: data_idx + 1, :] * w)
            teacher.update_stu_est(teacher.data_pool_tea_[data_idx: data_idx + 1, :], stu_linear_val)
        data_choices1.append(data_idx)
        data_point = [teacher.data_pool_[data_idx: data_idx + 1], teacher.gt_y_[data_idx: data_idx + 1]]
        #w, best_idx = learner.learn(data_point, gradients)
        w = learner.learn(data_point)
        dists1.append(np.sum(np.square(w - teacher.gt_w_)))
    line1, = plt.plot(dists1, label = 'regular')

    learnerS.reset(init_ws)
    w = learnerS.current_mean_
    dists3 = [np.sum(np.square(w - teacher.gt_w_))]
    data_choices3 = []
    eliminates = []
    for i in tqdm(range(train_iter_smart)):
        gradients, losses = learnerS.get_grads(teacher.data_pool_, teacher.gt_y_)
        if mode == 'omni':
            data_idx = teacher.choose(gradients, w, config_LS.lr)
        elif mode == 'surr':
            data_idx = teacher.choose_sur(gradients, losses, config_LS.lr)
        else:
            gradients_tea, _ = learnerS.get_grads(teacher.data_pool_tea_, teacher.gt_y_, teacher.stu_current_w_)
            data_idx = teacher.choose_imit(gradients_tea, config_L.lr)
            stu_linear_val = np.sum(teacher.data_pool_[data_idx: data_idx + 1, :] * w)
            teacher.update_stu_est(teacher.data_pool_tea_[data_idx: data_idx + 1, :], stu_linear_val)
        data_choices3.append(data_idx)
        data_point = [teacher.data_pool_[data_idx: data_idx + 1], teacher.gt_y_[data_idx: data_idx + 1]]
        if mode == 'surr':
            w, eliminate = learnerS.learn_sur(teacher.data_pool_, teacher.gt_y_, data_idx, gradients, losses)
        elif mode == 'omni' or mode == 'imit':
            w, eliminate = learnerS.learn(teacher.data_pool_, teacher.gt_y_, data_idx, gradients, True)
        eliminates.append(eliminate)
        dists3.append(np.sum(np.square(w - teacher.gt_w_)))
    line3, = plt.plot(dists3, label = 'smarter')

    learnerS.reset(init_ws)
    w = learnerS.current_mean_
    dists2 = [np.sum(np.square(w - teacher.gt_w_))]
    data_choices2 = []
    for i in tqdm(range(train_iter_smart)):
        gradients, losses = learnerS.get_grads(teacher.data_pool_, teacher.gt_y_)
        if mode == 'omni':
            data_idx = teacher.choose(gradients, w, config_LS.lr)
        elif mode == 'surr':
            data_idx = teacher.choose_sur(gradients, losses, config_LS.lr)
        else:
            gradients_tea, _ = learnerS.get_grads(teacher.data_pool_tea_, teacher.gt_y_, teacher.stu_current_w_)
            data_idx = teacher.choose_imit(gradients_tea, config_L.lr)
            stu_linear_val = np.sum(teacher.data_pool_[data_idx: data_idx + 1, :] * w)
            teacher.update_stu_est(teacher.data_pool_tea_[data_idx: data_idx + 1, :], stu_linear_val)
        data_choices2.append(data_idx)
        data_point = [teacher.data_pool_[data_idx: data_idx + 1], teacher.gt_y_[data_idx: data_idx + 1]]
        w, _ = learnerS.learn(teacher.data_pool_, teacher.gt_y_, data_idx,
                              gradients, np.mean(eliminates) / config_LS.particle_num)
        dists2.append(np.sum(np.square(w - teacher.gt_w_)))
    line2, = plt.plot(dists2, label = 'smart')

    plt.legend([line_neg1, line0, line1, line2, line3],
               ['batch', 'sgd', 'machine teaching: %d, %f' % (np.unique(data_choices1).shape[0], config_L.lr),\
                'compare: %d' % np.unique(data_choices2).shape[0],
                'pragmatic: %d, %f' % (np.unique(data_choices3).shape[0], config_LS.lr)], prop={'size': 12})
    plt.title('%s: %s' % (mode, learnerS.loss_type_))
    plt.show()
    #plt.savefig('figure_%s.png' % learnerS.loss_type_)
    pdb.set_trace()

if __name__ == '__main__':
    main()
