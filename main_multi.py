from easydict import EasyDict as edict
import tensorflow as tf
import numpy as np
from scipy.stats import multivariate_normal as mn
import os
import sys
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

from learner import Learner
from learnerM import LearnerSM
from teacherM import TeacherM

import pdb

def learn_basic(teacher, learner, train_iter, sess, init, sgd=True):
    sess.run(init)
    [w] = sess.run([learner.w_])
    dists = [np.sum(np.square(w - teacher.gt_w_))]
    dists_ = [np.sqrt(np.sum(np.square(w - teacher.gt_w_)))]
    accuracies = []
    logpdf = []
    for _ in tqdm(range(train_iter)):
        '''
        if (_ % 20 == 0):
            pdf = mn.pdf((teacher.gt_w_ - w).flatten(), mean = np.zeros(w.shape).flatten(), cov = 0.5)
            if (pdf == 0):
                print(_)
            logpdf.append(np.log(pdf))
        '''
        if teacher.config_.task == 'classification':
            logits = np.exp(np.matmul(teacher.data_pool_full_test_, w.T))
            probs = logits / np.sum(logits, axis = 1, keepdims = True)
            accuracy = np.mean(np.sum(-1 * np.log(probs) * teacher.gt_y_full_test_, 1))
        else:
            accuracy = np.mean(0.5 * np.sum(np.square(np.matmul(teacher.data_pool_full_test_, w.T) - teacher.gt_y_full_test_), axis = 1))
        accuracies.append(accuracy)
        teacher.sample()
        if (sgd):
            data_idx = np.random.randint(teacher.data_pool_.shape[0])
            data_point = [teacher.data_pool_[data_idx: data_idx + 1], teacher.gt_y_[data_idx: data_idx + 1]]
        else:
            data_point = [teacher.data_pool_, teacher.gt_y_]
        w = learner.learn(data_point)
        dists.append(np.sum(np.square(w - teacher.gt_w_)))
        dists_.append(np.sqrt(np.sum(np.square(w - teacher.gt_w_))))
    if teacher.config_.task == 'classification':
        accuracy = np.mean(np.argmax(np.matmul(teacher.data_pool_full_, w.T), 1) == teacher.gt_y_label_full_)
        print('test accuracy: %f' % accuracy)
    return dists, dists_, accuracies, logpdf

def learn(teacher, learner, mode, init_ws, train_iter, random_prob = None, plot_condition = False, prag = 0):
    learner.reset(init_ws)
    w = learner.current_mean_
    ws = [w]
    dists = [np.sum(np.square(w - teacher.gt_w_))]
    dists_ = [np.mean(np.sqrt(np.sum(np.square(learner.particles_ - teacher.gt_w_), axis = (1, 2))))]
    accuracies = []
    particle_hist = []
    data_choices = []
    eliminates = []
    logpdfs = []
    angles = []
    for i in tqdm(range(train_iter)):
        if i % 20 == 0:
            pdf = np.mean([mn.pdf((teacher.gt_w_ - p).flatten(), mean = np.zeros(p.shape).flatten(), cov = 0.5) for p in learner.particles_])
            logpdfs.append(np.log(pdf))
        if teacher.config_.task == 'classification':
            #accuracy = np.mean(np.argmax(np.matmul(teacher.data_pool_full_test_, w[0, ...].T), 1) == teacher.gt_y_label_full_test_)
            logits = np.exp(np.matmul(teacher.data_pool_full_test_, w[0, ...].T))
            probs = logits / np.sum(logits, axis = 1, keepdims = True)
            accuracy = np.mean(np.sum(-1 * np.log(probs) * teacher.gt_y_full_test_, 1))
        else:
            accuracy = np.mean(0.5 * np.sum(np.square(np.matmul(teacher.data_pool_full_test_, w[0, ...].T) - teacher.gt_y_full_test_), axis = 1))
        accuracies.append(accuracy)
        teacher.sample()
        gradients, _, losses = learner.get_grads(teacher.data_pool_, teacher.gt_y_)
        if mode == 'omni':
            data_idx = teacher.choose(gradients, w, learner.config_.lr)
        elif mode == 'surr':
            data_idx = teacher.choose_sur(gradients, losses, learner.config_.lr)
        else:
            if hasattr(teacher, 't_mat_') and teacher.t_mat_ is not None:
                stu2tea = np.concatenate([np.matmul(w[0, :, 0: -1], teacher.t_mat_.T), w[0, :, -1:] * np.ones([learner.config_.num_classes, 1])], 1)
                gradients_tea, _, _ = learner.get_grads(teacher.data_pool_tea_, teacher.gt_y_, np.expand_dims(stu2tea, 0))
            else:
                _, gradients_lv, _ = learner.get_grads(teacher.data_pool_, teacher.gt_y_)
                gradients_tea = []
                for j in range(teacher.data_pool_tea_.shape[0]):
                    gradients_tea.append(np.expand_dims(np.matmul(gradients_lv[j: j + 1, ...].T, teacher.data_pool_tea_[j: j + 1, ...]), 0))
                gradients_tea = np.concatenate(gradients_tea, 0)
            data_idx = teacher.choose_sur(gradients_tea, losses, learner.config_.lr)
        data_choices.append(data_idx)
        if mode == 'omni' or random_prob is not None:
            w, eliminate, angle = learner.learn(teacher.data_pool_, teacher.gt_y_, data_idx, gradients, i, teacher.gt_w_, random_prob = random_prob, prag = prag)
        else:
            w, eliminate, angle = learner.learn_sur(teacher.data_pool_, teacher.gt_y_, data_idx, gradients, losses, i, teacher.gt_w_, prag = prag)
            
        angles.append(angle)
        #particle_hist.append(copy.deepcopy(learner.particles_))
        eliminates.append(eliminate)
        dists.append(np.sum(np.square(w - teacher.gt_w_)))
        dists_.append(np.mean(np.sqrt(np.sum(np.square(learner.particles_ - teacher.gt_w_), axis = (1, 2)))))
        ws.append(w)

    if teacher.config_.task == 'classification':
        learned_w = copy.deepcopy(w[0, ...])
        accuracy = np.mean(np.argmax(np.matmul(teacher.data_pool_full_, learned_w.T), 1) == teacher.gt_y_label_full_)
        print('test accuracy: %f' % accuracy)
    
    if (teacher.config_.transform):
        mode = "imit"
    else:
        mode = "omni"
    if teacher.config_.data_x_tea is not None:
        teacher_dim = teacher.config_.data_x_tea.shape[1]
    else:
        teacher_dim = teacher.config_.data_dim
    if random_prob is None and plot_condition:
        num_bad = np.sum(np.array(angles) < 0)
        plt.plot(angles, 'bo', markersize=4)
        plt.plot(np.zeros(train_iter), 'r-')
        plt.title('Mode: %s, Student Dimension: %d, Teacher_Dimension: %d, Classes: %d \n bad: %d, ratio of good: %f' %
                (mode, learner.config_.data_dim, teacher_dim, learner.config_.num_classes,
                 num_bad, 1 - num_bad / train_iter))
        plt.ylabel("Projection Length")
        plt.xlabel("Iterations")
        plt.show()
    return dists, dists_, accuracies, logpdfs, eliminates

def main(argv):
    tfconfig = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    tfconfig.gpu_options.allow_growth = True
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    sess = tf.Session(config = tfconfig)
    np.random.seed(400)

    title = ''
    mode_idx = int(argv[2])
    modes = ['omni', 'surr', 'imit']
    mode = modes[mode_idx]
    title += mode
    title += '_'
    task = 'classification' if len(argv) == 6 else 'regression'
    title += task
    title += '_'
    
    lr = 1e-3
    dd = int(argv[1])
    num_classes = 10 if dd == 24 or dd == 30 else 4
    if task == 'regression':
        num_classes = 1
    title += 'num_classes'
    title += '_'
    title += str(num_classes)
    title += '_'
    if dd == 48:
        title += 'equation'
    elif dd == 24:
        title += 'mnist'
    else:
        title += 'gaussian'
    title += '_'
    title += 'particle_remove'
    
    dps = 3 * dd if task == 'classification' else 6 * dd
    num_particles = 3000
    train_iter_simple = 2000
    train_iter_smart = 2000
    reg_coef = 0

    dx = None if dd != 24 else np.load("/home/Datasets/MNIST/mnist_train_features.npy")
    dy = None if dd != 24 else np.load("/home/Datasets/MNIST/mnist_train_labels.npy")
    gt_w = None if dd != 24 else np.load("/home/Datasets/MNIST/mnist_train_features_weights.npy")
    tx = None if dd != 24 else np.load("/home/Datasets/MNIST/mnist_test_features.npy")
    ty = None if dd != 24 else np.load("/home/Datasets/MNIST/mnist_test_labels.npy")
    dx_tea = np.load("MNIST/mnist_train_features_tea.npy") if dd == 24 and mode == 'imit' else None
    dy_tea = np.load("MNIST/mnist_train_labels_tea.npy") if dd == 24 and mode == 'imit' else None
    gt_w_tea = np.load("MNIST/mnist_tf_gt_weights_tea.npy") if dd == 24 and mode == 'imit' else None
    tx_tea = np.load("MNIST/mnist_test_features_tea.npy") if dd == 24 and mode == 'imit' else None
    ty_tea = np.load("MNIST/mnist_test_labels_tea.npy") if dd == 24 and mode == 'imit' else None

    if dd == 48:
        dx = np.load("/home/Datasets/Equation/equation_train_features_cnn_3var_48_6layers.npy")[:50000]
        dy = np.load("/home/Datasets/Equation/equation_train_labels_cnn_3var_48_6layers.npy")[:50000].reshape((50000, 1))
        gt_w = np.load("/home/Datasets/Equation/equation_gt_weights_cnn_3var_48_6layers.npy")
        tx = np.load("/home/Datasets/Equation/equation_train_features_cnn_3var_48_6layers.npy")[:50000]
        ty = np.load("/home/Datasets/Equation/equation_train_labels_cnn_3var_48_6layers.npy")[:50000].reshape((50000, 1))
    
    config_T = edict({'data_pool_size_class': dps, 'data_dim': dd,'lr': lr, 'sample_size': 20,
                      'transform': mode == 'imit', 'num_classes': num_classes, 'task': task,
                      'data_x': dx, 'data_y': dy, 'test_x': tx, 'test_y': ty, 'gt_w': gt_w,
                      'data_x_tea': dx_tea, 'data_y_tea': dy_tea, 'test_x_tea': tx_tea, 'test_y_tea': ty_tea, 'gt_w_tea': gt_w_tea})
    config_LS = edict({'particle_num': num_particles, 'data_dim': dd, 'reg_coef': reg_coef, 'lr': lr, 'task': task,
                       'num_classes': num_classes, 'noise_scale_min': float(argv[3]), 'noise_scale_max': float(argv[4]),
                       'noise_scale_decay': float(argv[5]), 'target_ratio': 0, 'new_ratio': 1, 'replace_count': 1, "prob": 1})
    print(config_LS, config_T)
    config_L =  edict({'data_dim': dd, 'reg_coef': reg_coef, 'lr': lr, 'loss_type': 0, 'num_classes': num_classes, 'task': task})
    init_ws = np.concatenate([np.random.uniform(-1, 1, size = [config_LS.particle_num, config_LS.num_classes, dd]),
                              np.zeros([config_LS.particle_num, config_LS.num_classes, 1])], 2)
    init_w = np.mean(init_ws, 0)
    learner = Learner(sess, init_w, config_L)
    teacher = TeacherM(config_T)
    learnerM = LearnerSM(sess, config_LS)
    init = tf.global_variables_initializer()


    #dists_neg1_batch, dists_neg1_batch_, accuracies_neg1_batch, logpdf_neg1_batch = learn_basic(teacher, learner, train_iter_simple, sess, init, False)
    #dists_neg1_sgd, dists_neg1_sgd_, accuracies_neg1_sgd, logpdf_neg1_sgd = learn_basic(teacher, learner, train_iter_simple, sess, init, True)
    dists3, dists3_, accuracies3, logpdfs3, eliminates = learn(teacher, learnerM, mode, init_ws, train_iter_smart)
    dists4, dists4_, accuracies4, logpdfs4, eliminates4 = learn(teacher, learnerM, mode, init_ws, train_iter_smart, prag = 1)
    dists5, dists5_, accuracies5, logpdfs5, eliminates5 = learn(teacher, learnerM, mode, init_ws, train_iter_smart, prag = 2)
    
    #dists2, dists2_, accuracies2, logpdfs2, _ = learn(teacher, learnerM, mode, init_ws, train_iter_smart, np.mean(eliminates) / num_particles)
    #dists1, dists1_, accuracies1, logpdfs1, _ = learn(teacher, learnerM, mode, init_ws, train_iter_smart, 1)
    #dists0, dists0_, accuracies0, logpdfs0, _ = learn(teacher, learnerM, mode, init_ws, train_iter_smart, 0)

    
    
    fig, axs = plt.subplots(2, 2)
    #line_neg1_batch, = axs[0, 0].plot(dists_neg1_batch, label = 'batch')
    #line_neg1_sgd, = axs[0, 0].plot(dists_neg1_sgd, label = 'sgd')
    #line0, = axs[0, 0].plot(dists0, label = 'zero')
    #line1, = axs[0, 0].plot(dists1, label = 'one')
    #line2, = axs[0, 0].plot(dists2, label = 'smart')
    line3, = axs[0, 0].plot(dists3, label = 'smarter')
    line4, = axs[0, 0].plot(dists4, label = 'noise')
    line5, = axs[0, 0].plot(dists5, label = 'remove')
    axs[0, 0].set_title('mean_dist')

    #line_neg1_batch, = axs[1,1].plot(logpdf_neg1_batch, label = 'batch')
    #line_neg1_sgd, = axs[1,1].plot(logpdf_neg1_sgd, label = 'sgd')
    #line0, = axs[1, 1].plot(logpdfs0, label = 'zero')
    #line1, = axs[1, 1].plot(logpdfs1, label = 'one')
    #line2, = axs[1, 1].plot(logpdfs2, label = 'smart')
    line3, = axs[1, 1].plot(logpdfs3, label = 'smarter')
    line4, = axs[1, 1].plot(logpdfs4, label = 'noise')
    line5, = axs[1, 1].plot(logpdfs5, label = 'remove')
    axs[1, 1].set_title('log pdf per 20 iters')

    #line_neg1_batch, = axs[0, 1].plot(accuracies_neg1_batch, label = 'batch')
    #line_neg1_sgd, = axs[0, 1].plot(accuracies_neg1_sgd, label = 'sgd')
    #line0, = axs[0, 1].plot(accuracies0, label = 'zero')
    #line1, = axs[0, 1].plot(accuracies1, label = 'one')
    #line2, = axs[0, 1].plot(accuracies2, label = 'smart')
    line3, = axs[0, 1].plot(accuracies3, label = 'noise')
    line4, = axs[0, 1].plot(accuracies4, label = 'remove')
    axs[0, 1].set_title('test loss')

    #line_neg1_batch, = axs[1, 0].plot(dists_neg1_batch_, label = 'batch')
    #line_neg1_sgd, = axs[1, 0].plot(dists_neg1_sgd_, label = 'sgd')
    #line0, = axs[1, 0].plot(dists0_, label = 'zero')
    #line1, = axs[1, 0].plot(dists1_, label = 'one')
    #line2, = axs[1, 0].plot(dists2_, label = 'smart')
    line3, = axs[1, 0].plot(dists3_, label = 'noise')
    line4, = axs[1, 0].plot(dists4_, label = 'remove')
    axs[1, 0].set_title('dist mean')

    
    axs[0, 1].legend([line3, line4, line5],
               ['Pragmatic Replacement', 'noise','particle removal'], prop={'size': 6})
    fig.suptitle('%s class: %d: dim:%d_data:%d/%d/%d_particle:%d_noise: %f, %f, %d, ratio: %f, %f, lr:  %f' %\
              (mode, num_classes, dd, config_LS.replace_count, config_T.sample_size, dps, num_particles,
               config_LS.noise_scale_min, config_LS.noise_scale_max, config_LS.noise_scale_decay,
               config_LS.target_ratio, config_LS.new_ratio, config_LS.lr))
    plt.savefig('%s.png' % title)


if __name__ == '__main__':
    main(argv)
