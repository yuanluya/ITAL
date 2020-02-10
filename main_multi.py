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
    np.random.seed(400)

    mode_idx = int(sys.argv[2])
    modes = ['omni', 'surr', 'imit']
    mode = modes[mode_idx]

    lr = 1e-3
    dd = int(sys.argv[1])
    num_classes = 10 if dd == 24 or dd == 30 else 4
    dps = 3 * dd
    num_particles = 1000
    train_iter_smart = 2000 #2500 + 2500 * (lt == 0)
    reg_coef = 0# if lt == 0 else 5e-5
    
    if dd == 24:
        dx = np.load("MNIST/mnist_train_features.npy")#[:20000]
        dy = np.load("MNIST/mnist_train_labels.npy")#[:20000]
        gt_w = np.load("MNIST/mnist_tf_gt_weights.npy")
        tx = np.load("MNIST/mnist_test_features.npy")#[:20000]
        ty = np.load("MNIST/mnist_test_labels.npy")#[:20000]
        if mode == 'imit':
            dx_tea = np.load("MNIST/mnist_train_features_tea.npy")#[:20000]
            dy_tea = np.load("MNIST/mnist_train_labels_tea.npy")#[:20000]
            gt_w_tea = np.load("MNIST/mnist_tf_gt_weights_tea.npy")
            tx_tea = np.load("MNIST/mnist_test_features_tea.npy")#[:20000]
            ty_tea = np.load("MNIST/mnist_test_labels_tea.npy")#[:20000]
    else:
        dx = None
        dy = None
        gt_w = None
        tx = None
        ty = None
        dx_tea = None
        dy_tea = None
        gt_w_tea = None
        tx_tea = None
        ty_tea = None

    config_T = edict({'data_pool_size_class': dps, 'data_dim': dd,'lr': lr, 'sample_size': 20,
                      'transform': mode == 'imit', 'num_classes': num_classes,
                      'data_x': dx, 'data_y': dy, 'test_x': tx, 'test_y': ty, 'gt_w': gt_w,
                      'data_x_tea': dx_tea, 'data_y_tea': dy_tea, 'test_x_tea': tx_tea, 'test_y_tea': ty_tea, 'gt_w_tea': gt_w_tea})
    config_LS = edict({'particle_num': num_particles, 'data_dim': dd, 'reg_coef': reg_coef, 'lr': lr,
                       'num_classes': num_classes, 'noise_scale_min': 0.01, 'noise_scale_max': 0.1,
                       'noise_scale_decay': 300, 'target_ratio': 0, 'new_ratio': 1, 'replace_count': 1})
    print(config_LS, config_T)
    init_ws = np.concatenate([np.random.uniform(-1, 1, size = [config_LS.particle_num, config_LS.num_classes, dd]),
                              np.zeros([config_LS.particle_num, config_LS.num_classes, 1])], 2)
    teacher = TeacherM(config_T)
    learnerM = LearnerSM(sess, config_LS)
    init = tf.global_variables_initializer()
    sess.run(init)

    learnerM.reset(init_ws)
    w = learnerM.current_mean_
    ws = [w]
    dists3 = [np.sum(np.square(w - teacher.gt_w_))]
    dists3_ = [np.mean(np.sqrt(np.sum(np.square(learnerM.particles_ - teacher.gt_w_), axis = (1, 2))))]
    accuracies = []
    particle_hist = []
    data_choices3 = []
    eliminates = []
    for i in tqdm(range(train_iter_smart)):
        teacher.sample()
        gradients, _, losses = learnerM.get_grads(teacher.data_pool_, teacher.gt_y_)
        if mode == 'omni':
            data_idx = teacher.choose(gradients, w, config_LS.lr)
        elif mode == 'surr':
            data_idx = teacher.choose_sur(gradients, losses, config_LS.lr)
        else:
            if dd != 24:
                stu2tea = np.concatenate([np.matmul(w[0, :, 0: -1], teacher.t_mat_.T), w[0, :, -1:] * np.ones([config_LS.num_classes, 1])], 1)
                gradients_tea, _, _ = learnerM.get_grads(teacher.data_pool_tea_, teacher.gt_y_, np.expand_dims(stu2tea, 0))
            else:
                _, gradients_lv, _ = learnerM.get_grads(teacher.data_pool_, teacher.gt_y_)
                gradients_tea = []
                for i in range(teacher.data_pool_tea_.shape[0]):
                    gradients_tea.append(np.expand_dims(np.matmul(gradients_lv[i: i + 1, ...].T, teacher.data_pool_tea_[i: i + 1, ...]), 0))
                gradients_tea = np.concatenate(gradients_tea, 0)
            data_idx = teacher.choose_sur(gradients_tea, losses, config_LS.lr)
        data_choices3.append(data_idx)
        data_point = [teacher.data_pool_[data_idx: data_idx + 1], teacher.gt_y_[data_idx: data_idx + 1]]
        if mode == 'omni':
            w, eliminate = learnerM.learn(teacher.data_pool_, teacher.gt_y_, data_idx, gradients, i)
        else:
            w, eliminate = learnerM.learn_sur(teacher.data_pool_, teacher.gt_y_, data_idx, gradients, losses, i)
        particle_hist.append(copy.deepcopy(learnerM.particles_))
        eliminates.append(eliminate)
        dists3.append(np.sum(np.square(w - teacher.gt_w_)))
        dists3_.append(np.mean(np.sqrt(np.sum(np.square(learnerM.particles_ - teacher.gt_w_), axis = (1, 2)))))
        ws.append(w)
        accuracy = np.mean(np.argmax(np.matmul(teacher.data_pool_full_test_, w[0, ...].T), 1) == teacher.gt_y_label_full_test_)
        accuracies.append(accuracy)
    #line3, = plt.plot(dists3, label = 'smarter')
    line3, = plt.plot(accuracies, label = 'smarter')
    
    learned_w = copy.deepcopy(w[0, ...])
    accuracy = np.mean(np.argmax(np.matmul(teacher.data_pool_full_, learned_w.T), 1) == teacher.gt_y_label_full_)
    print('test accuracy: %f' % accuracy)
    
    # learnerM.reset(init_ws)
    # w = learnerM.current_mean_
    # dists3 = [np.sum(np.square(w - teacher.gt_w_))]
    # data_choices3 = []
    # eliminates = []
    # for i in tqdm(range(train_iter_smart)):
    #     teacher.gt_loss_ = teacher.gt_loss_full_
    #     gradients, losses = learnerM.get_grads(teacher.data_pool_full_, teacher.gt_y_full_)
    #     if mode == 'omni':
    #         data_idx = teacher.choose(gradients, w, config_LS.lr)
    #     elif mode == 'surr':
    #         data_idx = teacher.choose_sur(gradients, losses, config_LS.lr)
    #     else:
    #         stu2tea = np.concatenate([np.matmul(w[0, :, 0: -1], teacher.t_mat_.T), w[0, :, -1:] * np.ones([config_LS.num_classes, 1])], 1)
    #         gradients_tea, _ = learnerM.get_grads(teacher.data_pool_full_tea_, teacher.gt_y_full_, np.expand_dims(stu2tea, 0))
    #         data_idx = teacher.choose_sur(gradients_tea, losses, config_LS.lr)
    #     data_choices3.append(data_idx)
    #     data_point = [teacher.data_pool_full_[data_idx: data_idx + 1], teacher.gt_y_full_[data_idx: data_idx + 1]]
    #     if mode == 'omni':
    #         w, eliminate = learnerM.learn(teacher.data_pool_full_, teacher.gt_y_full_, data_idx, gradients, i)
    #     else:
    #         w, eliminate = learnerM.learn_sur(teacher.data_pool_full_, teacher.gt_y_full_, data_idx, gradients, losses, i)
    #     eliminates.append(eliminate)
    #     dists3.append(np.sum(np.square(w - teacher.gt_w_)))
    # line3S, = plt.plot(dists3, label = 'smarter_strong')

    learnerM.reset(init_ws)
    w = learnerM.current_mean_
    dists2 = [np.sum(np.square(w - teacher.gt_w_))]
    dists2_ = [np.mean(np.sqrt(np.sum(np.square(learnerM.particles_ - teacher.gt_w_), axis = (1, 2))))]
    accuracies = []
    data_choices2 = []
    random_ratio = np.mean(eliminates) / config_LS.particle_num
    for i in tqdm(range(train_iter_smart)):
        teacher.sample()
        gradients, _, losses = learnerM.get_grads(teacher.data_pool_, teacher.gt_y_)
        if mode == 'omni':
            data_idx = teacher.choose(gradients, w, config_LS.lr)
            #data_idx = np.random.randint(config_T.data_pool_size)
        elif mode == 'surr':
            data_idx = teacher.choose_sur(gradients, losses, config_LS.lr)
        else:
            if dd != 24:
                stu2tea = np.concatenate([np.matmul(w[0, :, 0: -1], teacher.t_mat_.T), w[0, :, -1:] * np.ones([config_LS.num_classes, 1])], 1)
                gradients_tea, _, _ = learnerM.get_grads(teacher.data_pool_tea_, teacher.gt_y_, np.expand_dims(stu2tea, 0))
            else:
                _, gradients_lv, _ = learnerM.get_grads(teacher.data_pool_, teacher.gt_y_)
                gradients_tea = []
                for i in range(teacher.data_pool_tea_.shape[0]):
                    gradients_tea.append(np.expand_dims(np.matmul(gradients_lv[i: i + 1, ...].T, teacher.data_pool_tea_[i: i + 1, ...]), 0))
                gradients_tea = np.concatenate(gradients_tea, 0)
            data_idx = teacher.choose_sur(gradients_tea, losses, config_LS.lr)
        data_choices2.append(data_idx)
        data_point = [teacher.data_pool_[data_idx: data_idx + 1], teacher.gt_y_[data_idx: data_idx + 1]]
        w, _ = learnerM.learn(teacher.data_pool_, teacher.gt_y_, data_idx,
                              gradients, i, random_prob = random_ratio)
        dists2.append(np.sum(np.square(w - teacher.gt_w_)))
        dists2_.append(np.mean(np.sqrt(np.sum(np.square(learnerM.particles_ - teacher.gt_w_), axis = (1, 2)))))
        accuracy = np.mean(np.argmax(np.matmul(teacher.data_pool_full_test_, w[0, ...].T), 1) == teacher.gt_y_label_full_test_)
        accuracies.append(accuracy)
    #line2, = plt.plot(dists2, label = 'smart')
    line2, = plt.plot(accuracies, label = 'smart')
    learned_w = copy.deepcopy(w[0, ...])
    accuracy = np.mean(np.argmax(np.matmul(teacher.data_pool_full_, learned_w.T), 1) == teacher.gt_y_label_full_)
    print('test accuracy: %f' % accuracy)

    learnerM.reset(init_ws)
    w = learnerM.current_mean_
    ws0 = [w]
    dists0 = [np.sum(np.square(w - teacher.gt_w_))]
    dists0_ = [np.mean(np.sqrt(np.sum(np.square(learnerM.particles_ - teacher.gt_w_), axis = (1, 2))))]
    accuracies = []
    data_choices2 = []
    particle_hist_0 = []
    random_ratio = 0
    for i in tqdm(range(train_iter_smart)):
        teacher.sample()
        gradients, _, losses = learnerM.get_grads(teacher.data_pool_, teacher.gt_y_)
        if mode == 'omni':
            data_idx = teacher.choose(gradients, w, config_LS.lr)
            #data_idx = np.random.randint(config_T.data_pool_size)
        elif mode == 'surr':
            data_idx = teacher.choose_sur(gradients, losses, config_LS.lr)
        else:
            if dd != 24:
                stu2tea = np.concatenate([np.matmul(w[0, :, 0: -1], teacher.t_mat_.T), w[0, :, -1:] * np.ones([config_LS.num_classes, 1])], 1)
                gradients_tea, _, _ = learnerM.get_grads(teacher.data_pool_tea_, teacher.gt_y_, np.expand_dims(stu2tea, 0))
            else:
                _, gradients_lv, _ = learnerM.get_grads(teacher.data_pool_, teacher.gt_y_)
                gradients_tea = []
                for i in range(teacher.data_pool_tea_.shape[0]):
                    gradients_tea.append(np.expand_dims(np.matmul(gradients_lv[i: i + 1, ...].T, teacher.data_pool_tea_[i: i + 1, ...]), 0))
                gradients_tea = np.concatenate(gradients_tea, 0)
            data_idx = teacher.choose_sur(gradients_tea, losses, config_LS.lr)
        data_choices2.append(data_idx)
        data_point = [teacher.data_pool_[data_idx: data_idx + 1], teacher.gt_y_[data_idx: data_idx + 1]]
        w, _ = learnerM.learn(teacher.data_pool_, teacher.gt_y_, data_idx,
                              gradients, i, random_prob = random_ratio)
        ws0.append(w)
        dists0.append(np.sum(np.square(w - teacher.gt_w_)))
        dists0_.append(np.mean(np.sqrt(np.sum(np.square(learnerM.particles_ - teacher.gt_w_), axis = (1, 2)))))
        particle_hist_0.append(copy.deepcopy(learnerM.particles_))
        accuracy = np.mean(np.argmax(np.matmul(teacher.data_pool_full_test_, w[0, ...].T), 1) == teacher.gt_y_label_full_test_)
        accuracies.append(accuracy)
    #line0, = plt.plot(dists0, label = 'zero')
    line0, = plt.plot(accuracies, label = 'zero')
    learned_w = copy.deepcopy(w[0, ...])
    accuracy = np.mean(np.argmax(np.matmul(teacher.data_pool_full_, learned_w.T), 1) == teacher.gt_y_label_full_)
    print('test accuracy: %f' % accuracy)

    learnerM.reset(init_ws)
    w = learnerM.current_mean_
    dists1 = [np.sum(np.square(w - teacher.gt_w_))]
    dists1_ = [np.mean(np.sqrt(np.sum(np.square(learnerM.particles_ - teacher.gt_w_), axis = (1, 2))))]
    accuracies = []
    data_choices2 = []
    random_ratio = 1
    for i in tqdm(range(train_iter_smart)):
        teacher.sample()
        gradients, _, losses = learnerM.get_grads(teacher.data_pool_, teacher.gt_y_)
        if mode == 'omni':
            data_idx = teacher.choose(gradients, w, config_LS.lr)
            #data_idx = np.random.randint(config_T.data_pool_size)
        elif mode == 'surr':
            data_idx = teacher.choose_sur(gradients, losses, config_LS.lr)
        else:
            if dd != 24:
                stu2tea = np.concatenate([np.matmul(w[0, :, 0: -1], teacher.t_mat_.T), w[0, :, -1:] * np.ones([config_LS.num_classes, 1])], 1)
                gradients_tea, _, _ = learnerM.get_grads(teacher.data_pool_tea_, teacher.gt_y_, np.expand_dims(stu2tea, 0))
            else:
                _, gradients_lv, _ = learnerM.get_grads(teacher.data_pool_, teacher.gt_y_)
                gradients_tea = []
                for i in range(teacher.data_pool_tea_.shape[0]):
                    gradients_tea.append(np.expand_dims(np.matmul(gradients_lv[i: i + 1, ...].T, teacher.data_pool_tea_[i: i + 1, ...]), 0))
                gradients_tea = np.concatenate(gradients_tea, 0)
            data_idx = teacher.choose_sur(gradients_tea, losses, config_LS.lr)
        data_choices2.append(data_idx)
        data_point = [teacher.data_pool_[data_idx: data_idx + 1], teacher.gt_y_[data_idx: data_idx + 1]]
        w, _ = learnerM.learn(teacher.data_pool_, teacher.gt_y_, data_idx,
                              gradients, i, random_prob = random_ratio)
        dists1.append(np.sum(np.square(w - teacher.gt_w_)))
        dists1_.append(np.mean(np.sqrt(np.sum(np.square(learnerM.particles_ - teacher.gt_w_), axis = (1, 2)))))
        accuracy = np.mean(np.argmax(np.matmul(teacher.data_pool_full_test_, w[0, ...].T), 1) == teacher.gt_y_label_full_test_)
        accuracies.append(accuracy)
    line1, = plt.plot(accuracies, label = 'one')
    learned_w = copy.deepcopy(w[0, ...])
    accuracy = np.mean(np.argmax(np.matmul(teacher.data_pool_full_, learned_w.T), 1) == teacher.gt_y_label_full_)
    print('test accuracy: %f' % accuracy)

    plt.legend([line0, line1, line2, line3],#, line3S],
               ['zero', 'one', 'compare',
                'pragmatic, %f' % (config_LS.lr)], prop={'size': 12})
    plt.title('%s class: %d: dim:%d_data:%d/%d/%d_particle:%d_noise: %f, %f, %d, ratio: %f, %f' %\
              (mode, num_classes, dd, config_LS.replace_count, config_T.sample_size, dps, num_particles,
               config_LS.noise_scale_min, config_LS.noise_scale_max, config_LS.noise_scale_decay,
               config_LS.target_ratio, config_LS.new_ratio))
    plt.show()

    line0, = plt.plot(dists0, label = 'zero')
    line1, = plt.plot(dists1, label = 'one')
    line2, = plt.plot(dists2, label = 'smart')
    line3, = plt.plot(dists3, label = 'smater')
    plt.legend([line0, line1, line2, line3],#, line3S],
               ['zero', 'one', 'compare',
                'pragmatic, %f' % (config_LS.lr)], prop={'size': 12})
    plt.title('%s class: %d: dim:%d_data:%d/%d/%d_particle:%d_noise: %f, %f, %d, ratio: %f, %f' %\
              (mode, num_classes, dd, config_LS.replace_count, config_T.sample_size, dps, num_particles,
               config_LS.noise_scale_min, config_LS.noise_scale_max, config_LS.noise_scale_decay,
               config_LS.target_ratio, config_LS.new_ratio))
    plt.show()

    line0, = plt.plot(dists0_, label = 'zero')
    line1, = plt.plot(dists1_, label = 'one')
    line2, = plt.plot(dists2_, label = 'smart')
    line3, = plt.plot(dists3_, label = 'smater')
    plt.legend([line0, line1, line2, line3],#, line3S],
               ['zero', 'one', 'compare',
                'pragmatic, %f' % (config_LS.lr)], prop={'size': 12})
    plt.title('%s class: %d: dim:%d_data:%d/%d/%d_particle:%d_noise: %f, %f, %d, ratio: %f, %f' %\
              (mode, num_classes, dd, config_LS.replace_count, config_T.sample_size, dps, num_particles,
               config_LS.noise_scale_min, config_LS.noise_scale_max, config_LS.noise_scale_decay,
               config_LS.target_ratio, config_LS.new_ratio))
    plt.show()
    #plt.savefig('figure_%s.png' % learnerM.loss_type_)
    pdb.set_trace()

if __name__ == '__main__':
    main()
