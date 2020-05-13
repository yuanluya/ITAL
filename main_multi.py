from easydict import EasyDict as edict
from multiprocessing import Process, Manager
import numpy as np
from scipy.stats import multivariate_normal as mn
import os
import sys
import copy
import matplotlib.pyplot as plt
import sys
import time
from tqdm import tqdm

from learner import Learner
from learnerM import LearnerSM
from teacherM import TeacherM

import pdb


def learn_basic(teacher, learner, train_iter, sess, init, sgd=True):
    sess.run(init)
    [w] = sess.run([learner.w_])
    dists = [np.sqrt(np.sum(np.square(w - teacher.gt_w_)))]
    dists_ = [np.sqrt(np.sum(np.square(w - teacher.gt_w_)))]
    accuracies = []
    losses_list = []

    for _ in tqdm(range(train_iter)):

        if teacher.config_.task == 'classification':
            logits = np.exp(np.matmul(teacher.data_pool_full_test_, w.T))
            probs = logits / np.sum(logits, axis = 1, keepdims = True)
            accuracy = np.mean(np.argmax(np.matmul(teacher.data_pool_full_test_, w.T), 1) == teacher.gt_y_label_full_test_)

            loss = np.mean(np.sum(-1 * np.log(probs) * teacher.gt_y_full_test_, 1))
        else:
            accuracy = 1
            loss = np.mean(0.5 * np.sum(np.square(np.matmul(teacher.data_pool_full_test_, w.T) - teacher.gt_y_full_test_), axis = 1))
        
        accuracies.append(accuracy)
        losses_list.append(loss)
        teacher.sample()
        if (sgd):
            data_idx = np.random.randint(teacher.data_pool_.shape[0])
            data_point = [teacher.data_pool_[data_idx: data_idx + 1], teacher.gt_y_[data_idx: data_idx + 1]]
        else:
            data_point = [teacher.data_pool_, teacher.gt_y_]
        w = learner.learn(data_point)
        dists.append(np.sqrt(np.sum(np.square(w - teacher.gt_w_))))
        dists_.append(np.sqrt(np.sum(np.square(w - teacher.gt_w_))))
    if teacher.config_.task == 'classification':
        accuracy = np.mean(np.argmax(np.matmul(teacher.data_pool_full_, w.T), 1) == teacher.gt_y_label_full_)
        print('basic test accuracy: %f' % accuracy)
    return dists, dists_, accuracies, losses_list

def learn(teacher, learner, mode, init_ws, train_iter, random_prob = None, plot_condition = False):
    learner.reset(init_ws)
    if mode == 'expt':
        learner.particle_weights_ = np.ones(learner.config_.particle_num)
        eta = np.sqrt(8 * np.log(learner.config_.particle_num) / 2000)
    w = learner.current_mean_
    ws = [w]
    dists = [np.sqrt(np.sum(np.square(w - teacher.gt_w_)))]
    dists_ = [np.mean(np.sqrt(np.sum(np.square(learner.particles_ - teacher.gt_w_), axis = (1, 2))))]
    accuracies = []
    losses_list = []
    particle_hist = []
    data_choices = []
    eliminates = []

    angles = []
    for i in tqdm(range(train_iter)):
        if teacher.config_.task == 'classification':
            accuracy = np.mean(np.argmax(np.matmul(teacher.data_pool_full_test_, w[0, ...].T), 1) == teacher.gt_y_label_full_test_)

            logits = np.exp(np.matmul(teacher.data_pool_full_test_, w[0, ...].T))
            probs = logits / np.sum(logits, axis = 1, keepdims = True)
            loss = np.mean(np.sum(-1 * np.log(probs) * teacher.gt_y_full_test_, 1))
        else:
            loss = np.mean(0.5 * np.sum(np.square(np.matmul(teacher.data_pool_full_test_, w[0, ...].T) - teacher.gt_y_full_test_), axis = 1))
            accuracy = 1

        losses_list.append(loss)
        accuracies.append(accuracy)
        teacher.sample()
        gradients, _, losses = learner.get_grads(teacher.data_pool_, teacher.gt_y_)
        if mode == 'omni' or mode == 'expt':
            data_idx = teacher.choose(gradients, w, learner.config_.lr)
        elif mode == 'omni_strt':
            data_idx = teacher.choose_strt(gradients, w)
        elif mode == 'sgd_omni_cont':
            data_idx = teacher.choose(gradients, w, learner.config_.lr)
        elif mode == 'omni_cont':
            data_idx = teacher.choose(gradients, w, learner.config_.lr, hard = True)
        elif mode == 'imit' or mode == 'imit_cont' or mode == 'sgd_imit_cont':
            if hasattr(teacher, 't_mat_') and teacher.t_mat_ is not None:
                stu2tea = np.concatenate([np.matmul(w[0, :, 0: -1], teacher.t_mat_.T), w[0, :, -1:] * np.ones([learner.config_.num_classes, 1])], 1)
                gradients_tea, _, _ = learner.get_grads(teacher.data_pool_tea_, teacher.gt_y_, np.expand_dims(stu2tea, 0))
            else:
                _, gradients_lv, _ = learner.get_grads(teacher.data_pool_, teacher.gt_y_)
                gradients_tea = []
                for j in range(teacher.data_pool_tea_.shape[0]):
                    gradients_tea.append(np.expand_dims(np.matmul(gradients_lv[j: j + 1, ...].T, teacher.data_pool_tea_[j: j + 1, ...]), 0))
                gradients_tea = np.concatenate(gradients_tea, 0)
            data_idx = teacher.choose_sur(gradients_tea, losses, learner.config_.lr, hard = True)
        data_choices.append(data_idx)
        
        if mode == 'sgd_imit_cont' or mode == 'sgd_omni_cont':
            # w, eliminate, angle = learner.learn(teacher.data_pool_, teacher.gt_y_, data_idx, gradients, i, teacher.gt_w_, random_prob=random_prob, strt = True)
            w, eliminate, angle = learner.learn_cont(teacher.data_pool_, teacher.gt_y_,
                                                     data_idx, gradients, i, teacher.gt_w_,
                                                     K = -1 * learner.config_.cont_K if learner.config_.cont_K else None)
        elif mode == 'omni_cont':
            w, eliminate, angle = learner.learn_cont(teacher.data_pool_, teacher.gt_y_,
                                                     data_idx, gradients, i, teacher.gt_w_, K = learner.config_.cont_K)
        elif mode == 'imit_cont':
            w, eliminate, angle = learner.learn_sur_cont(teacher.data_pool_, teacher.gt_y_,
                                                         data_idx, gradients, losses, i, teacher.gt_w_, K = learner.config_.cont_K)
        elif mode == 'omni' or random_prob is not None:
            w, eliminate, angle = learner.learn(teacher.data_pool_, teacher.gt_y_,
                                                data_idx, gradients, i, teacher.gt_w_, random_prob = random_prob)
        else:
            w, eliminate, angle = learner.learn_sur(teacher.data_pool_, teacher.gt_y_, data_idx, gradients, losses, i, teacher.gt_w_)
        angles.append(angle)
        #particle_hist.append(copy.deepcopy(learner.particles_))
        eliminates.append(eliminate)
        dists.append(np.sqrt(np.sum(np.square(w - teacher.gt_w_))))
        dists_.append(np.mean(np.sqrt(np.sum(np.square(learner.particles_ - teacher.gt_w_), axis = (1, 2)))))
        ws.append(w)

    if teacher.config_.task == 'classification':
        learned_w = copy.deepcopy(w[0, ...])
        accuracy = np.mean(np.argmax(np.matmul(teacher.data_pool_full_, learned_w.T), 1) == teacher.gt_y_label_full_)
        print('smart test accuracy: %f' % accuracy, random_prob)

    if (teacher.config_.transform):
        mode = "imit"
    else:
        mode = "omni"
    if teacher.config_.data_x_tea is not None:
        teacher_dim = teacher.config_.data_x_tea.shape[1]
    else:
        teacher_dim = teacher.config_.data_dim

    return dists, dists_, accuracies, losses_list, eliminates

def learn_thread(teacher, learner, mode, init_ws, train_iter, random_prob, key, thread_return):
    import tensorflow as tf
    tfconfig = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config = tfconfig)
    init = tf.global_variables_initializer()

    learnerM = LearnerSM(sess, learner)
    dists, dists_, accuracies, losses, eliminate = learn(teacher, learnerM, mode, init_ws, train_iter, random_prob)

    thread_return[key] = [dists, dists_, accuracies, losses, eliminate]

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    beta = 40000
    K = 1
    np.random.seed(int(sys.argv[8]))
 
    multi_thread = True

    title = ''
    mode_idx = int(sys.argv[2])
    modes = ['omni', 'surr', 'imit']
    mode = modes[mode_idx]
    title += mode
    title += '_'
    task = 'classification' if 'regression' not in  sys.argv else 'regression'
    title += task
    title += '_'

    lr = 1e-3
    dd = int(sys.argv[1])
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
    title += sys.argv[8]

    dps = 3 * dd if task == 'classification' else 6 * dd
    num_particles = 1000
    train_iter_simple = 2000
    train_iter_smart = 2000
    reg_coef = 0

    dx = None if dd != 24 else np.load("MNIST/mnist_train_features.npy")
    dy = None if dd != 24 else np.load("MNIST/mnist_train_labels.npy")
    gt_w = None if dd != 24 else np.load("MNIST/mnist_tf_gt_weights.npy")
    tx = None if dd != 24 else np.load("MNIST/mnist_test_features.npy")
    ty = None if dd != 24 else np.load("MNIST/mnist_test_labels.npy")
    dx_tea = np.load("MNIST/mnist_train_features_tea.npy") if dd == 24 and mode == 'imit' else None
    dy_tea = np.load("MNIST/mnist_train_labels_tea.npy") if dd == 24 and mode == 'imit' else None
    gt_w_tea = np.load("MNIST/mnist_tf_gt_weights_tea.npy") if dd == 24 and mode == 'imit' else None
    tx_tea = np.load("MNIST/mnist_test_features_tea.npy") if dd == 24 and mode == 'imit' else None
    ty_tea = np.load("MNIST/mnist_test_labels_tea.npy") if dd == 24 and mode == 'imit' else None

    if dd == 48:
        dx = np.load("Equation_data/equation_train_features_cnn_3var_48_6layers.npy")[:50000]
        dy = np.load("Equation_data/equation_train_labels_cnn_3var_48_6layers.npy")[:50000].reshape((50000, 1))
        gt_w = np.load("Equation_data/equation_gt_weights_cnn_3var_48_6layers.npy")
        tx = np.load("Equation_data/equation_train_features_cnn_3var_48_6layers.npy")[:50000]
        ty = np.load("Equation_data/equation_train_labels_cnn_3var_48_6layers.npy")[:50000].reshape((50000, 1))
    
    config_T = edict({'data_pool_size_class': dps, 'data_dim': dd,'lr': lr, 'sample_size': 20,
                      'transform': mode == 'imit', 'num_classes': num_classes, 'task': task,
                      'data_x': dx, 'data_y': dy, 'test_x': tx, 'test_y': ty, 'gt_w': gt_w, 'beta': beta,
                      'data_x_tea': dx_tea, 'data_y_tea': dy_tea, 'test_x_tea': tx_tea, 'test_y_tea': ty_tea, 'gt_w_tea': gt_w_tea})
    config_LS = edict({'particle_num': num_particles, 'data_dim': dd, 'reg_coef': reg_coef, 'lr': lr, 'task': task,
                       'num_classes': num_classes, 'noise_scale_min': float(sys.argv[3]), 'noise_scale_max': float(sys.argv[4]), 'beta': beta, 'cont_K': K,
                       'noise_scale_decay': float(sys.argv[5]), 'target_ratio': 0, 'new_ratio': 1, 'replace_count': 1, "prob": 1})
    print(config_LS, config_T)
    config_L =  edict({'data_dim': dd, 'reg_coef': reg_coef, 'lr': lr, 'loss_type': 0, 'num_classes': num_classes, 'task': task})
    init_ws = np.concatenate([np.random.uniform(-1, 1, size = [config_LS.particle_num, config_LS.num_classes, dd]),
                              np.zeros([config_LS.particle_num, config_LS.num_classes, 1])], 2)
    init_w = np.mean(init_ws, 0)

    teacher = TeacherM(config_T)
    manager = Manager()

    if multi_thread:

        return_dict = manager.dict()
        jobs = []

        p = Process(target = learn_thread, args = (teacher, config_LS, mode, init_ws, train_iter_smart, 1, 1, return_dict))
        jobs.append(p)
        p.start()
    
        p = Process(target = learn_thread, args = (teacher, config_LS, "%s_cont" % mode, init_ws,
                                                   train_iter_smart, None, "%s_cont" % mode, return_dict))
        jobs.append(p)
        p.start()
        p = Process(target = learn_thread, args = (teacher, config_LS, 'sgd_%s_cont' % mode, init_ws, train_iter_smart,
                                                   None, 'sgd_%s_cont' % mode, return_dict))
        jobs.append(p)
        p.start()

        for j in jobs:
            print("joining", j)
            j.join()
            
        dists1, dists1_, accuracies1, losses1, _ = return_dict[1]
        np.save('dist1_' + title + '.npy', np.array(dists1))
        np.save('dist1__' + title + '.npy', np.array(dists1_))
        np.save('accuracies1_' + title + '.npy', np.array(accuracies1))
        np.save('losses1_' + title + '.npy', np.array(losses1))

        dists7, dists7_, accuracies7, losses7, _ = return_dict['sgd_%s_cont' % mode]
        np.save('dist7_' + title + '.npy', np.array(dists7))
        np.save('dist7__' + title + '.npy', np.array(dists7_))
        np.save('accuracies7_' + title + '.npy', np.array(accuracies7))
        np.save('losses7_' + title + '.npy', np.array(losses7))
        
        dists8, dists8_, accuracies8, losses8, _ = return_dict['%s_cont' % mode]
        np.save('dist8_' + title + '.npy', np.array(dists8))
        np.save('dist8__' + title + '.npy', np.array(dists8_))
        np.save('accuracies8_' + title + '.npy', np.array(accuracies8))
        np.save('losses8_' + title + '.npy', np.array(losses8))


    import tensorflow as tf
    tfconfig = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config = tfconfig)
    learner = Learner(sess, init_w, config_L)

    init = tf.global_variables_initializer()

    if multi_thread:
        dists_neg1_batch, dists_neg1_batch_, accuracies_neg1_batch, losses_neg1_batch = learn_basic(teacher, learner, train_iter_simple, sess, init, False)
        dists_neg1_sgd, dists_neg1_sgd_, accuracies_neg1_sgd, losses_neg1_sgd = learn_basic(teacher, learner, train_iter_simple, sess, init, True)
        np.save('distbatch_' + title + '.npy', np.array(dists_neg1_batch))
        np.save('distbatch__' + title + '.npy', np.array(dists_neg1_batch_))
        np.save('accuraciesbatch_' + title + '.npy', np.array(accuracies_neg1_batch))
        np.save('lossesbatch_' + title + '.npy', np.array(losses_neg1_batch))

        np.save('distsgd_' + title + '.npy', np.array(dists_neg1_sgd))
        np.save('distsgd__' + title + '.npy', np.array(dists_neg1_sgd_))
        np.save('accuraciessgd_' + title + '.npy', np.array(accuracies_neg1_sgd))
        np.save('lossessgd_' + title + '.npy', np.array(losses_neg1_sgd))

    else:
        learnerM = LearnerSM(sess, config_LS)

        dists3, dists3_, accuracies3, logpdfs3, eliminates = learn(teacher, learnerM, mode, init_ws, train_iter_smart)
        dists2, dists2_, accuracies2, logpdfs2, _ = learn(teacher, learnerM, mode, init_ws, train_iter_smart, np.mean(eliminates) / num_particles)
        dists1, dists1_, accuracies1, logpdfs1, _ = learn(teacher, learnerM, mode, init_ws, train_iter_smart, 1)
        dists0, dists0_, accuracies0, logpdfs0, _ = learn(teacher, learnerM, mode, init_ws, train_iter_smart, 0)
        dists4, dists4_, accuracies4, logpdfs4, _ = learn(teacher, learnerM, 'sgd_%s_cont' % mode, init_ws, train_iter_smart)
        dists5, dists5_, accuracies5, logpdfs5, _ = learn(teacher, learnerM, '%s_cont' % mode, init_ws, train_iter_smart)
        dists_neg1_batch, dists_neg1_batch_, accuracies_neg1_batch, logpdf_neg1_batch = learn_basic(teacher, learner, train_iter_simple, sess, init, False)
        dists_neg1_sgd, dists_neg1_sgd_, accuracies_neg1_sgd, logpdf_neg1_sgd = learn_basic(teacher, learner, train_iter_simple, sess, init, True)

if __name__ == '__main__':
    main()
