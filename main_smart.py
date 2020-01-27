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

    mode_idx = 0
    modes = ['omni', 'surr', 'imit']
    mode = modes[mode_idx]

    lr = 1e-4
    lt = int(sys.argv[1])
    dd = 20 if lt != 0 else 50
    dps = 2 * dd# + 150 * (lt == 2)
    num_particles = 1000
    train_iter_simple = 5000
    train_iter_smart = 5000 #2500 + 2500 * (lt == 0)
    reg_coef = 0# if lt == 0 else 5e-5

    config_T = edict({'data_pool_size': dps, 'data_dim': dd, 'loss_type': lt, 'lr': 0.1 * lr, 'transform': mode == 'imit', 'sample_size': int(0.2 * dps)})
    config_L = edict({'data_dim': dd, 'reg_coef': reg_coef, 'lr': lr, 'loss_type': lt})
    config_LS = edict({'particle_num': num_particles, 'data_dim': dd, 'reg_coef': reg_coef, 'lr': lr, 'loss_type': lt, 'noise_scale': 0.1})
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
        teacher.sample()
        data_point = [teacher.data_pool_, teacher.gt_y_]
        w = learner.learn(data_point)
        dists0.append(np.sum(np.square(w - teacher.gt_w_)))
    
    line_neg1, = plt.plot(dists0, label = 'batch')

    sess.run(init)
    [w] = sess.run([learner.w_])
    dists0 = [np.sum(np.square(w - teacher.gt_w_))]
    for _ in tqdm(range(train_iter_simple)):
        teacher.sample()
        data_idx = np.random.randint(teacher.data_pool_.shape[0])
        data_point = [teacher.data_pool_[data_idx: data_idx + 1], teacher.gt_y_[data_idx: data_idx + 1]]
        w = learner.learn(data_point)
        dists0.append(np.sum(np.square(w - teacher.gt_w_)))
    
    line0, = plt.plot(dists0, label = 'sgd')

    sess.run(init)
    [w] = sess.run([learner.w_])
    dists1 = [np.sum(np.square(w - teacher.gt_w_))]
    data_choices1 = []
    for i in tqdm(range(train_iter_simple)):
        teacher.sample()
        gradients, losses = learner.get_grads(teacher.data_pool_, teacher.gt_y_)
        if mode == 'surr':
            data_idx = teacher.choose_sur(gradients, losses, config_L.lr)
        elif mode == 'omni':
            data_idx = teacher.choose(gradients, w, config_L.lr)
        else:
            stu2tea = np.concatenate([np.matmul(w[:, 0: -1], teacher.t_mat_.T), w[0, -1] * np.ones([1, 1])], 1)
            gradients_tea, _ = learnerS.get_grads(teacher.data_pool_tea_, teacher.gt_y_, stu2tea)
            data_idx = teacher.choose_sur(gradients_tea, losses, config_L.lr)
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
        teacher.sample()
        gradients, losses = learnerS.get_grads(teacher.data_pool_, teacher.gt_y_)
        if mode == 'omni':
            data_idx = teacher.choose(gradients, w, config_LS.lr)
        elif mode == 'surr':
            data_idx = teacher.choose_sur(gradients, losses, config_LS.lr)
        else:
            stu2tea = np.concatenate([np.matmul(w[:, 0: -1], teacher.t_mat_.T), w[0, -1] * np.ones([1, 1])], 1)
            gradients_tea, _ = learnerS.get_grads(teacher.data_pool_tea_, teacher.gt_y_, stu2tea)
            data_idx = teacher.choose_sur(gradients_tea, losses, config_LS.lr)
        data_choices3.append(data_idx)
        data_point = [teacher.data_pool_[data_idx: data_idx + 1], teacher.gt_y_[data_idx: data_idx + 1]]
        if mode == 'omni':
            w, eliminate = learnerS.learn(teacher.data_pool_, teacher.gt_y_, data_idx, gradients)
        else:
            w, eliminate = learnerS.learn_sur(teacher.data_pool_, teacher.gt_y_, data_idx, gradients, losses)
        eliminates.append(eliminate)
        dists3.append(np.sum(np.square(w - teacher.gt_w_)))
    line3, = plt.plot(dists3, label = 'smarter')

    learnerS.reset(init_ws)
    w = learnerS.current_mean_
    dists3 = [np.sum(np.square(w - teacher.gt_w_))]
    data_choices3 = []
    eliminates = []
    for i in tqdm(range(train_iter_smart)):
        teacher.gt_loss_ = teacher.gt_loss_full_
        gradients, losses = learnerS.get_grads(teacher.data_pool_full_, teacher.gt_y_full_)
        if mode == 'omni':
            data_idx = teacher.choose(gradients, w, config_LS.lr)
        elif mode == 'surr':
            data_idx = teacher.choose_sur(gradients, losses, config_LS.lr)
        else:
            stu2tea = np.concatenate([np.matmul(w[:, 0: -1], teacher.t_mat_.T), w[0, -1] * np.ones([1, 1])], 1)
            gradients_tea, _ = learnerS.get_grads(teacher.data_pool_tea_full_, teacher.gt_y_full_, stu2tea)
            data_idx = teacher.choose_sur(gradients_tea, losses, config_LS.lr)
        data_choices3.append(data_idx)
        data_point = [teacher.data_pool_full_[data_idx: data_idx + 1], teacher.gt_y_full_[data_idx: data_idx + 1]]
        if mode == 'omni':
            w, eliminate = learnerS.learn(teacher.data_pool_full_, teacher.gt_y_full_, data_idx, gradients)
        else:
            w, eliminate = learnerS.learn_sur(teacher.data_pool_full_, teacher.gt_y_full_, data_idx, gradients, losses)
        eliminates.append(eliminate)
        dists3.append(np.sum(np.square(w - teacher.gt_w_)))
    line3S, = plt.plot(dists3, label = 'smarter_strong')

    learnerS.reset(init_ws)
    w = learnerS.current_mean_
    dists2 = [np.sum(np.square(w - teacher.gt_w_))]
    data_choices2 = []
    random_ratio = np.mean(eliminates) / config_LS.particle_num
    for i in tqdm(range(train_iter_smart)):
        teacher.sample()
        gradients, losses = learnerS.get_grads(teacher.data_pool_, teacher.gt_y_)
        if mode == 'omni':
            data_idx = teacher.choose(gradients, w, config_LS.lr)
            #data_idx = np.random.randint(config_T.data_pool_size)
        elif mode == 'surr':
            data_idx = teacher.choose_sur(gradients, losses, config_LS.lr)
        else:
            stu2tea = np.concatenate([np.matmul(w[:, 0: -1], teacher.t_mat_.T), w[0, -1] * np.ones([1, 1])], 1)
            gradients_tea, _ = learnerS.get_grads(teacher.data_pool_tea_, teacher.gt_y_, stu2tea)
            data_idx = teacher.choose_sur(gradients_tea, losses, config_LS.lr)
        data_choices2.append(data_idx)
        data_point = [teacher.data_pool_[data_idx: data_idx + 1], teacher.gt_y_[data_idx: data_idx + 1]]
        w, _ = learnerS.learn(teacher.data_pool_, teacher.gt_y_, data_idx,
                              gradients, random_prob = random_ratio)
        dists2.append(np.sum(np.square(w - teacher.gt_w_)))
    line2, = plt.plot(dists2, label = 'smart')

    learnerS.reset(init_ws)
    w = learnerS.current_mean_
    dists2 = [np.sum(np.square(w - teacher.gt_w_))]
    data_choices2 = []
    random_ratio = 1
    for i in tqdm(range(train_iter_smart)):
        teacher.sample()
        gradients, losses = learnerS.get_grads(teacher.data_pool_, teacher.gt_y_)
        if mode == 'omni':
            data_idx = teacher.choose(gradients, w, config_LS.lr)
            #data_idx = np.random.randint(config_T.data_pool_size)
        elif mode == 'surr':
            data_idx = teacher.choose_sur(gradients, losses, config_LS.lr)
        else:
            stu2tea = np.concatenate([np.matmul(w[:, 0: -1], teacher.t_mat_.T), w[0, -1] * np.ones([1, 1])], 1)
            gradients_tea, _ = learnerS.get_grads(teacher.data_pool_tea_, teacher.gt_y_, stu2tea)
            data_idx = teacher.choose_sur(gradients_tea, losses, config_LS.lr)
        data_choices2.append(data_idx)
        data_point = [teacher.data_pool_[data_idx: data_idx + 1], teacher.gt_y_[data_idx: data_idx + 1]]
        w, _ = learnerS.learn(teacher.data_pool_, teacher.gt_y_, data_idx,
                              gradients, random_prob = random_ratio)
        dists2.append(np.sum(np.square(w - teacher.gt_w_)))
    line2_1, = plt.plot(dists2, label = 'one')

    plt.legend([line_neg1, line0, line1, line2, line2_1, line3, line3S],
               ['batch', 'sgd', 'machine teaching: %d, %f' % (np.unique(data_choices1).shape[0], config_L.lr),\
                'compare', 'compare_1', 'pragmatic, %f' % (config_LS.lr), 'pragmatic_full'], prop={'size': 12})
    plt.title('%s: %s_dim:%d_data:%d_particle:%d' % (mode, learnerS.loss_type_, dd, dps, num_particles))
    plt.show()
    #plt.savefig('figure_%s.png' % learnerS.loss_type_)
    pdb.set_trace()

if __name__ == '__main__':
    main()
