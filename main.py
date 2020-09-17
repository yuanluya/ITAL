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

from Learners.learner_basic import Learner
from Learners.learner import LearnerSM
from Teachers.teacher import TeacherM

import pdb

def learn_basic(teacher, learner, train_iter, sess, init, sgd=True):
    sess.run(init)
    [w] = sess.run([learner.w_])
    dists = [np.sqrt(np.sum(np.square(w - teacher.gt_w_)))]
    dists_ = [np.sqrt(np.sum(np.square(w - teacher.gt_w_)))]
    accuracies = []
    losses_list = []
    data_pool = []
    gt_y = []
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
        data_pool.append(teacher.data_pool_)
        gt_y.append(teacher.gt_y_)        
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
    return dists, dists_, accuracies, losses_list, data_pool, gt_y

def learn(teacher, learner, mode, init_ws, train_iter, random_prob = None, plot_condition = False):
    learner.reset(init_ws)

    w = learner.current_mean_
    ws = [w]
    dists = [np.sqrt(np.sum(np.square(w - teacher.gt_w_)))]
    dists_ = [np.mean(np.sqrt(np.sum(np.square(learner.particles_ - teacher.gt_w_), axis = (1, 2))))]
    accuracies = []
    losses_list = []

    data_pool = []
    gt_y = []
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
        if mode == 'omni':
            data_idx = teacher.choose(gradients, w, learner.config_.lr)
        elif mode == 'omni_cont':
            data_idx = teacher.choose(gradients, w, learner.config_.lr, hard = True)
        elif mode == 'imit' or mode == 'imit_cont':
            if hasattr(teacher, 't_mat_') and teacher.t_mat_ is not None:
                stu2tea = np.concatenate([np.matmul(w[0, :, 0: -1], teacher.t_mat_.T), w[0, :, -1:] * np.ones([learner.config_.num_classes, 1])], 1)
                gradients_tea, _, _ = learner.get_grads(teacher.data_pool_tea_, teacher.gt_y_, np.expand_dims(stu2tea, 0))
            else:
                _, gradients_lv, _ = learner.get_grads(teacher.data_pool_, teacher.gt_y_)
                gradients_tea = []
                for j in range(teacher.data_pool_tea_.shape[0]):
                    gradients_tea.append(np.expand_dims(np.matmul(gradients_lv[j: j + 1, ...].T, teacher.data_pool_tea_[j: j + 1, ...]), 0))
                gradients_tea = np.concatenate(gradients_tea, 0)
            data_idx = teacher.choose_sur(gradients_tea, losses, learner.config_.lr, hard = True)#(mode[-4: ] != 'cont'))

        data_pool.append(teacher.data_pool_)
        gt_y.append(teacher.gt_y_)            
        if mode == 'omni_cont':
            w = learner.learn_cont(teacher.data_pool_, teacher.gt_y_,
                                                     data_idx, gradients)
        elif mode == 'imit_cont':
            w = learner.learn_sur_cont(teacher.data_pool_, teacher.gt_y_,
                                                         data_idx, gradients, losses, i)
        elif mode == 'omni' or random_prob is not None:
            w = learner.learn(teacher.data_pool_, teacher.gt_y_,
                                                data_idx, gradients)
        dists.append(np.sqrt(np.sum(np.square(w - teacher.gt_w_))))
        dists_.append(np.mean(np.sqrt(np.sum(np.square(learner.particles_ - teacher.gt_w_), axis = (1, 2)))))
        ws.append(w)

    if teacher.config_.task == 'classification':
        learned_w = copy.deepcopy(w[0, ...])
        accuracy = np.mean(np.argmax(np.matmul(teacher.data_pool_full_, learned_w.T), 1) == teacher.gt_y_label_full_)
        print('smart test accuracy: %f' % accuracy, random_prob)

    return dists, dists_, accuracies, losses_list, data_pool, gt_y

def learn_thread(teacher, learner, mode, init_ws, train_iter, random_prob, key, thread_return):
    import tensorflow as tf
    tfconfig = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config = tfconfig)
    init = tf.global_variables_initializer()

    learnerM = LearnerSM(sess, learner)
    dists, dists_, accuracies, logpdfs = learn(teacher, learnerM, mode, init_ws, train_iter, random_prob)

    thread_return[key] = [dists, dists_, accuracies, logpdfs]

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    exp_folder = sys.argv[1]
    if not os.path.isdir(os.path.join('./Experiments', exp_folder)):
        print('Cannot find target folder')
        exit()
    if not os.path.exists(os.path.join('./Experiments', exp_folder, 'config.py')):
        print('Cannot find config.py in target folder')
        exit()
    exec('from Experiments.%s import config' % exp_folder, globals())
    exec('from Experiments.%s.config import config_T' % exp_folder, globals())
    exec('from Experiments.%s.config import config_LS' % exp_folder, globals())
    exec('from Experiments.%s.config import config_L' % exp_folder, globals())

    directory = sys.argv[1]
    random_seed = sys.argv[2]
    np.random.seed(int(random_seed))
    multi_thread = config.multi_thread
    mode = config.mode

    train_iter_simple = config.train_iter_simple
    train_iter_smart = config.train_iter_smart

    print(config_LS, config_T)
    init_ws = np.concatenate([np.random.uniform(-1, 1, size = [config_LS.particle_num, config_LS.num_classes, config.dd]),
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

        for j in jobs:
            print("joining", j)
            j.join()
        
        dists1, dists1_, accuracies1, losses1, data_poolIMT, gt_yIMT = return_dict[1]
        dists8, dists8_, accuracies8, losses8, data_poolITAL, gt_yITAL = return_dict['%s_cont' % mode]

    import tensorflow as tf
    tfconfig = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config = tfconfig)
    learner = Learner(sess, init_w, config_L)

    init = tf.global_variables_initializer()

    if multi_thread:
        dists_neg1_batch, dists_neg1_batch_, accuracies_neg1_batch, losses_neg1_batch, data_poolBatch, gt_yBatch \
             = learn_basic(teacher, learner, train_iter_simple, sess, init, False)
        dists_neg1_sgd, dists_neg1_sgd_, accuracies_neg1_sgd, losses_neg1_sgd, data_poolSGD, gt_ySGD \
             = learn_basic(teacher, learner, train_iter_simple, sess, init, True)
    
    else:
        learnerM = LearnerSM(sess, config_LS)

        dists1, dists1_, accuracies1, losses1, data_poolIMT, gt_yIMT = learn(teacher, learnerM, mode, init_ws, train_iter_smart, 1)
        dists8, dists8_, accuracies8, losses8, data_poolITAL, gt_yITAL = learn(teacher, learnerM, '%s_cont' % mode, init_ws, train_iter_smart)
        dists_neg1_batch, dists_neg1_batch_, accuracies_neg1_batch, losses_neg1_batch, data_poolBatch, gt_yBatch = learn_basic(teacher, learner, train_iter_simple, sess, init, False)
        dists_neg1_sgd, dists_neg1_sgd_, accuracies_neg1_sgd, losses_neg1_sgd, data_poolSGD, gt_ySGD = learn_basic(teacher, learner, train_iter_simple, sess, init, True)

    np.save('Experiments/' + directory + '/dist1_' + random_seed + '.npy', np.array(dists1))
    np.save('Experiments/' + directory + '/dist1__' + random_seed + '.npy', np.array(dists1_))
    np.save('Experiments/' + directory+ '/accuracies1_' + random_seed + '.npy', np.array(accuracies1))
    np.save('Experiments/' + directory + '/losses1_' + random_seed + '.npy', np.array(losses1))
    
    np.save('Experiments/' + directory + '/dist8_' + random_seed + '.npy', np.array(dists8))
    np.save('Experiments/' + directory + '/dist8__' + random_seed + '.npy', np.array(dists8_))
    np.save('Experiments/' + directory + '/accuracies8_' + random_seed + '.npy', np.array(accuracies8))
    np.save('Experiments/' + directory + '/losses8_' + random_seed + '.npy', np.array(losses8))    
    
    np.save('Experiments/' + directory + '/distbatch_' + random_seed + '.npy', np.array(dists_neg1_batch))
    np.save('Experiments/' + directory + '/distbatch__' + random_seed + '.npy', np.array(dists_neg1_batch_))
    np.save('Experiments/' + directory + '/accuraciesbatch_' + random_seed + '.npy', np.array(accuracies_neg1_batch))
    np.save('Experiments/' + directory + '/lossesbatch_' + random_seed + '.npy', np.array(losses_neg1_batch))

    np.save('Experiments/' + directory + '/distsgd_' + random_seed + '.npy', np.array(dists_neg1_sgd))
    np.save('Experiments/' + directory + '/distsgd__' + random_seed + '.npy', np.array(dists_neg1_sgd_))
    np.save('Experiments/' + directory + '/accuraciessgd_' + random_seed + '.npy', np.array(accuracies_neg1_sgd))
    np.save('Experiments/' + directory + '/lossessgd_' + random_seed + '.npy', np.array(losses_neg1_sgd))    

    np.save('Experiments/' + directory + '/data_poolIMT_' + random_seed + '.npy', np.array(data_poolIMT))
    np.save('Experiments/' + directory + '/data_poolITAL_' + random_seed + '.npy', np.array(data_poolITAL))
    np.save('Experiments/' + directory + '/data_poolBatch_' + random_seed + '.npy', np.array(data_poolBatch))
    np.save('Experiments/' + directory + '/data_poolSGD_' + random_seed + '.npy', np.array(data_poolSGD))    

    np.save('Experiments/' + directory + '/gt_yIMT_' + random_seed + '.npy', np.array(gt_yIMT))
    np.save('Experiments/' + directory + '/gt_yITAL_' + random_seed + '.npy', np.array(gt_yITAL))
    np.save('Experiments/' + directory + '/gt_yBatch_' + random_seed + '.npy', np.array(gt_yBatch))
    np.save('Experiments/' + directory + '/gt_ySGD_' + random_seed + '.npy', np.array(gt_ySGD))   
    
if __name__ == '__main__':
    main()
