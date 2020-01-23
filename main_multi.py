from easydict import EasyDict as edict
import tensorflow as tf
import numpy as np
import os
import sys
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

from learner import Learner
from learnerM import LearnerSM
from teacherM import TeacherM

import pdb

def main():
    tfconfig = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    tfconfig.gpu_options.allow_growth = True
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    sess = tf.Session(config = tfconfig)
    np.random.seed(1000)

    mode_idx = 0
    modes = ['omni', 'surr', 'imit']
    mode = modes[mode_idx]

    lr = 3e-4
    num_classes = 10
    dd = int(sys.argv[1])
    dps = 4 * dd
    num_particles = dd * 500
    train_iter_smart = 5000 #2500 + 2500 * (lt == 0)
    reg_coef = 0# if lt == 0 else 5e-5

    config_T = edict({'data_pool_size_class': dps, 'data_dim': dd,'lr': lr,
                      'transform': mode == 'imit', 'num_classes': num_classes})
    config_LS = edict({'particle_num': num_particles, 'data_dim': dd, 'reg_coef': reg_coef, 'lr': lr, 'num_classes': num_classes})
    init_ws = np.concatenate([np.random.uniform(-1, 1, size = [config_LS.particle_num, config_LS.num_classes, dd]),
                              np.zeros([config_LS.particle_num, config_LS.num_classes, 1])], 2)
    init_w = np.mean(init_ws, 0, keepdims = True)
    teacher = TeacherM(config_T)
    learnerM = LearnerSM(sess, config_LS)
    init = tf.global_variables_initializer()
    sess.run(init)

    learnerM.reset(init_ws)
    w = learnerM.current_mean_
    dists3 = [np.sum(np.square(w - teacher.gt_w_))]
    data_choices3 = []
    eliminates = []
    for i in tqdm(range(train_iter_smart)):
        gradients, losses = learnerM.get_grads(teacher.data_pool_, teacher.gt_y_)
        if mode == 'omni':
            data_idx = teacher.choose(gradients, w, config_LS.lr)
        elif mode == 'surr':
            data_idx = teacher.choose_sur(gradients, losses, config_LS.lr)
        else:
            stu2tea = np.concatenate([np.matmul(w[0, :, 0: -1], teacher.t_mat_.T), w[0, :, -1:] * np.ones([config_LS.num_classes, 1])], 1)
            gradients_tea, _ = learnerM.get_grads(teacher.data_pool_tea_, teacher.gt_y_, np.expand_dims(stu2tea, 0))
            data_idx = teacher.choose_sur(gradients_tea, losses, config_LS.lr)
        data_choices3.append(data_idx)
        data_point = [teacher.data_pool_[data_idx: data_idx + 1], teacher.gt_y_[data_idx: data_idx + 1]]
        if mode == 'omni':
            w, eliminate = learnerM.learn(teacher.data_pool_, teacher.gt_y_, data_idx, gradients)
        else:
            w, eliminate = learnerM.learn_sur(teacher.data_pool_, teacher.gt_y_, data_idx, gradients, losses)
        eliminates.append(eliminate)
        dists3.append(np.sum(np.square(w - teacher.gt_w_)))
    line3, = plt.plot(dists3, label = 'smarter')

    learnerM.reset(init_ws)
    w = learnerM.current_mean_
    dists2 = [np.sum(np.square(w - teacher.gt_w_))]
    data_choices2 = []
    random_ratio = np.mean(eliminates) / config_LS.particle_num
    for i in tqdm(range(train_iter_smart)):
        gradients, losses = learnerM.get_grads(teacher.data_pool_, teacher.gt_y_)
        if mode == 'omni':
            data_idx = teacher.choose(gradients, w, config_LS.lr)
            #data_idx = np.random.randint(config_T.data_pool_size)
        elif mode == 'surr':
            data_idx = teacher.choose_sur(gradients, losses, config_LS.lr)
        else:
            stu2tea = np.concatenate([np.matmul(w[0, :, 0: -1], teacher.t_mat_.T), w[0, :, -1:] * np.ones([config_LS.num_classes, 1])], 1)
            gradients_tea, _ = learnerM.get_grads(teacher.data_pool_tea_, teacher.gt_y_, np.expand_dims(stu2tea, 0))
            data_idx = teacher.choose_sur(gradients_tea, losses, config_LS.lr)
        data_choices2.append(data_idx)
        data_point = [teacher.data_pool_[data_idx: data_idx + 1], teacher.gt_y_[data_idx: data_idx + 1]]
        w, _ = learnerM.learn(teacher.data_pool_, teacher.gt_y_, data_idx,
                              gradients, random_prob = random_ratio)
        dists2.append(np.sum(np.square(w - teacher.gt_w_)))
    line2, = plt.plot(dists2, label = 'smart')

    learnerM.reset(init_ws)
    w = learnerM.current_mean_
    dists2 = [np.sum(np.square(w - teacher.gt_w_))]
    data_choices2 = []
    random_ratio = 0
    for i in tqdm(range(train_iter_smart)):
        gradients, losses = learnerM.get_grads(teacher.data_pool_, teacher.gt_y_)
        if mode == 'omni':
            data_idx = teacher.choose(gradients, w, config_LS.lr)
            #data_idx = np.random.randint(config_T.data_pool_size)
        elif mode == 'surr':
            data_idx = teacher.choose_sur(gradients, losses, config_LS.lr)
        else:
            stu2tea = np.concatenate([np.matmul(w[0, :, 0: -1], teacher.t_mat_.T), w[0, :, -1:] * np.ones([config_LS.num_classes, 1])], 1)
            gradients_tea, _ = learnerM.get_grads(teacher.data_pool_tea_, teacher.gt_y_, np.expand_dims(stu2tea, 0))
            data_idx = teacher.choose_sur(gradients_tea, losses, config_LS.lr)
        data_choices2.append(data_idx)
        data_point = [teacher.data_pool_[data_idx: data_idx + 1], teacher.gt_y_[data_idx: data_idx + 1]]
        w, _ = learnerM.learn(teacher.data_pool_, teacher.gt_y_, data_idx,
                              gradients, random_prob = random_ratio)
        dists2.append(np.sum(np.square(w - teacher.gt_w_)))
    line0, = plt.plot(dists2, label = 'zero')

    learnerM.reset(init_ws)
    w = learnerM.current_mean_
    dists2 = [np.sum(np.square(w - teacher.gt_w_))]
    data_choices2 = []
    random_ratio = 1
    for i in tqdm(range(train_iter_smart)):
        gradients, losses = learnerM.get_grads(teacher.data_pool_, teacher.gt_y_)
        if mode == 'omni':
            data_idx = teacher.choose(gradients, w, config_LS.lr)
            #data_idx = np.random.randint(config_T.data_pool_size)
        elif mode == 'surr':
            data_idx = teacher.choose_sur(gradients, losses, config_LS.lr)
        else:
            stu2tea = np.concatenate([np.matmul(w[0, :, 0: -1], teacher.t_mat_.T), w[0, :, -1:] * np.ones([config_LS.num_classes, 1])], 1)
            gradients_tea, _ = learnerM.get_grads(teacher.data_pool_tea_, teacher.gt_y_, np.expand_dims(stu2tea, 0))
            data_idx = teacher.choose_sur(gradients_tea, losses, config_LS.lr)
        data_choices2.append(data_idx)
        data_point = [teacher.data_pool_[data_idx: data_idx + 1], teacher.gt_y_[data_idx: data_idx + 1]]
        w, _ = learnerM.learn(teacher.data_pool_, teacher.gt_y_, data_idx,
                              gradients, random_prob = random_ratio)
        dists2.append(np.sum(np.square(w - teacher.gt_w_)))
    line1, = plt.plot(dists2, label = 'one')

    plt.legend([line0, line1, line2, line3],
               ['zero', 'one', 'compare: %d' % np.unique(data_choices2).shape[0],
                'pragmatic: %d, %f' % (np.unique(data_choices3).shape[0], config_LS.lr)], prop={'size': 12})
    plt.title('%s class: %d: dim:%d_data:%d_particle:%d' % (mode, num_classes, dd, dps, num_particles))
    plt.show()
    #plt.savefig('figure_%s.png' % learnerM.loss_type_)
    pdb.set_trace()

if __name__ == '__main__':
    main()
