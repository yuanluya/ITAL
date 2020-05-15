from multiprocessing import Process, Manager
import copy
import os
import sys
import numpy as np
from easydict import EasyDict as edict
from tqdm import tqdm
import matplotlib.pyplot as plt
if plt.get_backend() == 'Qt5Agg':
    from matplotlib.backends.qt_compat import QtWidgets
    qApp = QtWidgets.QApplication(sys.argv)
    plt.matplotlib.rcParams['figure.dpi'] = qApp.desktop().physicalDpiX()
import time
from OIRL.map import Map
from teacher_irl import TeacherIRL
from learner_irl import LearnerIRL

import pdb

def learn_basic(teacher, learner, train_iter, init_ws, test_set, batch = True):
    learner.reset(init_ws)

    ws = [learner.current_mean_]
    dists = [np.mean(np.max(abs(learner.current_action_prob() - teacher.action_probs_), axis = 1))]
    dists_ = [np.sum(np.square(learner.current_mean_ - teacher.stu_gt_reward_param_))]
    distsq = [np.mean(np.square(learner.q_map_ - teacher.q_map_))]
    actual_rewards = [teacher.map_.test_walk(teacher.reward_param_, learner.action_probs_, test_set[0], greedy = False)]
    for i in tqdm(range(train_iter)):
        teacher.sample()
        if not batch:
            data_idx = np.random.randint(teacher.config_.sample_size)
        else:
            data_idx = np.arange(teacher.config_.sample_size)
        _, gradients = teacher.choose(learner.current_mean_, learner.lr_)
        w, _ = learner.learn_cont(teacher.mini_batch_indices_, teacher.mini_batch_opt_acts_, data_idx,
                                         gradients, i, teacher.stu_gt_reward_param_, -1, batch = batch)
        ws.append(w)
        dists_.append(np.sum(np.square(learner.current_mean_ - teacher.stu_gt_reward_param_)))
        dists.append(np.mean(np.max(abs(learner.current_action_prob() - teacher.action_probs_), axis = 1)))
        distsq.append(np.mean(np.square(learner.q_map_ - teacher.q_map_)))
        if (i + 1) % 10 == 0:
            actual_rewards.append(teacher.map_.test_walk(teacher.reward_param_, learner.action_probs_, test_set[i + 1], greedy = False))
    learner.lr_ = learner.config_.lr
    return dists, dists_, distsq, actual_rewards, ws

def learn(teacher, learner, mode, init_ws, train_iter, test_set, random_prob = None):
    learner.reset(init_ws)

    eliminates = []
    dists_ = [np.sum(np.square(learner.current_mean_ - teacher.stu_gt_reward_param_))]
    dists = [np.mean(np.max(abs(learner.current_action_prob() - teacher.action_probs_), axis = 1))]
    distsq = [np.mean(np.square(learner.q_map_ - teacher.q_map_))]
    actual_rewards = [teacher.map_.test_walk(teacher.reward_param_, learner.action_probs_, test_set[0], greedy = False)]
    for i in tqdm(range(train_iter)):
        teacher.sample()
        if mode[0: 4] == 'omni':
            data_idx, gradients = teacher.choose(learner.current_mean_, learner.lr_, hard = (mode.find('cont') == -1))
        elif mode[0: 4] == 'imit':
            stu_rewards = np.sum(learner.map_.state_feats_ * learner.current_mean_, axis = 1, keepdims = True)
            data_idx, gradients, l_stu = teacher.choose_imit(stu_rewards, learner.lr_, hard = (mode.find('cont') == -1))
        
        if mode == 'omni' or random_prob is not None:
            eliminate, _ = learner.learn(teacher.mini_batch_indices_, teacher.mini_batch_opt_acts_, data_idx,
                                         gradients, i, teacher.stu_gt_reward_param_, random_prob)
        elif mode == 'omni_cont':
            eliminate, _ = learner.learn_cont(teacher.mini_batch_indices_, teacher.mini_batch_opt_acts_, data_idx,
                                         gradients, i, teacher.stu_gt_reward_param_, learner.config_.cont_K)
        elif mode == 'imit_sgd_cont' or mode == 'omni_sgd_cont':
            eliminate, _ = learner.learn_cont(teacher.mini_batch_indices_, teacher.mini_batch_opt_acts_, data_idx,
                                         gradients, i, teacher.stu_gt_reward_param_,
                                         -1 * learner.config_.cont_K if learner.config_.cont_K else -100)
        elif mode == 'imit':
            eliminate, _ = learner.learn_imit(teacher.mini_batch_indices_, teacher.mini_batch_opt_acts_, data_idx,
                                              l_stu, i, teacher.stu_gt_reward_param_)
        elif mode == 'imit_cont':
            eliminate, _ = learner.learn_imit_cont(teacher.mini_batch_indices_, teacher.mini_batch_opt_acts_, data_idx,
                                                   l_stu, i, teacher.stu_gt_reward_param_, learner.config_.cont_K)
        dists_.append(np.sum(np.square(learner.current_mean_ - teacher.stu_gt_reward_param_)))
        dists.append(np.mean(np.max(abs(learner.current_action_prob() - teacher.action_probs_), axis = 1)))
        distsq.append(np.mean(np.square(learner.q_map_ - teacher.q_map_)))
        eliminates.append(eliminate)
        if (i + 1) % 10 == 0:
            actual_rewards.append(teacher.map_.test_walk(teacher.reward_param_, learner.action_probs_, test_set[i + 1], greedy = False))

    learner.lr_ = learner.config_.lr
    return dists, dists_, distsq, actual_rewards, eliminates

def learn_thread(teacher, learner, mode, init_ws, train_iter, test_set, random_prob, dict_key, thread_return):
    dists, dists_, distsq, actual_rewards, eliminates = learn(teacher, learner, mode, init_ws, train_iter, test_set, random_prob)
    thread_return[dict_key] = [dists, dists_, distsq, actual_rewards, eliminates]

def learn_thread_tf(config_T, config_L, mode, train_iter, random_prob, return_key, thread_return):
    import tensorflow as tf
    tfconfig = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    tfconfig.gpu_options.allow_growth = True
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    sess = tf.Session(config = tfconfig)
    np.random.seed(int(sys.argv[9]))

    map_l = Map(sess, config_L)
    map_t = Map(sess, config_T)

    reward_type = sys.argv[2]

    gt_r_param_tea = map_l.reward_generate(3) if reward_type == 'E' else np.random.uniform(-2, 2, size = [1, config_L.shape ** 2])
    gt_r_param_stu = copy.deepcopy(gt_r_param_tea)
    if config_L.shuffle_state_feat:
        gt_r_param_stu[:, map_l.feat_idx_] = gt_r_param_tea
    assert(np.max(abs(np.sum(gt_r_param_stu * map_l.state_feats_, axis = 1) - np.sum(gt_r_param_tea * map_t.state_feats_, axis = 1))) < 1e-9)

    teacher = TeacherIRL(sess, map_t, config_T, gt_r_param_tea, gt_r_param_stu)

    init_ws = np.random.uniform(-2, 2, size = [config_L.particle_num, teacher.map_.num_states_])
    test_set = np.random.choice(teacher.map_.num_states_, size = [train_iter + 1, teacher.map_.num_states_ * 20])
    #pdb.set_trace()
    learner = LearnerIRL(sess, map_l, config_L)

    dists, dists_, distsq, actual_rewards, eliminates = learn(teacher, learner, mode, init_ws, train_iter, test_set, random_prob)
    thread_return[return_key] = [dists, dists_, distsq, actual_rewards, eliminates]

def teacher_run_tf(config_T, config_L, train_iter, thread_return):

    import tensorflow as tf
    tfconfig = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    tfconfig.gpu_options.allow_growth = True
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    sess = tf.Session(config = tfconfig)
    np.random.seed(int(sys.argv[9]))

    map_l = Map(sess, config_L)
    map_t = Map(sess, config_T)


    reward_type = sys.argv[2]
    gt_r_param_tea = map_l.reward_generate(3) if reward_type == 'E' else np.random.uniform(-2, 2, size = [1, config_T.shape ** 2])
    gt_r_param_stu = copy.deepcopy(gt_r_param_tea)
    if config_L.shuffle_state_feat:
        gt_r_param_stu[:, map_l.feat_idx_] = gt_r_param_tea
    assert(np.max(abs(np.sum(gt_r_param_stu * map_l.state_feats_, axis = 1) - np.sum(gt_r_param_tea * map_t.state_feats_, axis = 1))) < 1e-9)
    teacher = TeacherIRL(sess, map_t, config_T, gt_r_param_tea, gt_r_param_stu)

    init_ws = np.random.uniform(-2, 2, size = [config_L.particle_num, teacher.map_.num_states_])
    test_set = np.random.choice(teacher.map_.num_states_, size = [train_iter + 1, teacher.map_.num_states_ * 20])
    student = LearnerIRL(sess, map_l, config_L)

    teacher_rewards = [teacher.map_.test_walk(teacher.reward_param_, teacher.action_probs_, test_set[i + 1], greedy = False)\
                       for i in range(0, train_iter, 2)]
    thread_return.append(teacher_rewards)

def main():
    use_tf = False
    multi_thread = False

    mode_idx = int(sys.argv[1])
    modes = ['omni', 'imit']
    mode = modes[mode_idx]

    shape = int(sys.argv[3])
    lr = 1e-3
    beta = int(sys.argv[7])

    reward_type = sys.argv[2]
    approx_k = float(sys.argv[8])

    beta_select = 10000
    K = 1
    train_iter = 20

    title = ''
    title += mode
    title += '_'
        
    title += sys.argv[3]
    title += '_'
    title += 'beta'
    title += '_'
    title += sys.argv[7]
    title += '_'
    title += sys.argv[9]


    config_T = edict({'shape': shape, 'approx_type': 'gsm', 'beta': beta, 'shuffle_state_feat': False,
                      'lr': lr, 'sample_size': 20, 'use_tf': use_tf, 'approx_k': approx_k, 'beta_select': beta_select})
    config_L = edict({'shape': shape, 'approx_type': 'gsm', 'beta': beta, 'lr': lr, "prob": 1,
                      'shuffle_state_feat': mode == 'imit', 'particle_num': 1000, 'replace_count': 1,
                      'noise_scale_min': float(sys.argv[4]), 'noise_scale_max': float(sys.argv[5]), 'noise_scale_decay': float(sys.argv[6]), 'cont_K': K,
                      'target_ratio': 0, 'new_ratio': 1, 'use_tf': use_tf, 'approx_k': approx_k, 'beta_select': beta_select})

    np.set_printoptions(precision = 4)

    if not use_tf:
        import tensorflow as tf
        tfconfig = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
        tfconfig.gpu_options.allow_growth = True
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        sess = tf.Session(config = tfconfig)
        np.random.seed(int(sys.argv[9]))

        map_l = Map(sess, config_L)
        map_t = Map(sess, config_T)

        gt_r_param_tea = map_l.reward_generate(3) if reward_type == 'E' else np.random.uniform(-2, 2, size = [1, shape ** 2])
        gt_r_param_stu = copy.deepcopy(gt_r_param_tea)
        if config_L.shuffle_state_feat:
            gt_r_param_stu[:, map_l.feat_idx_] = gt_r_param_tea
        assert(np.max(abs(np.sum(gt_r_param_stu * map_l.state_feats_, axis = 1) - np.sum(gt_r_param_tea * map_t.state_feats_, axis = 1))) < 1e-9)

        teacher = TeacherIRL(sess, map_t, config_T, gt_r_param_tea, gt_r_param_stu)
        print(teacher.initial_valg_maps_[12][12])

        init_ws = np.random.uniform(-2, 2, size = [config_L.particle_num, teacher.map_.num_states_])
        #print("INIT_WS", init_ws[12][12])

        test_set = np.random.choice(teacher.map_.num_states_, size = [train_iter + 1, teacher.map_.num_states_ * 20])

    manager = Manager()
    return_list = manager.list()

    if not use_tf:
        def teacher_run(thread_return):
            teacher_rewards = [teacher.map_.test_walk(teacher.reward_param_, teacher.action_probs_, test_set[i + 1], greedy = False)\
                               for i in range(0, train_iter, 2)]
            thread_return.append(teacher_rewards)
        p = Process(target = teacher_run, args=(return_list,))
        p.start()
    else:
        p = Process(target = teacher_run_tf, args = (config_T, config_L, train_iter, return_list))
        p.start()
    time.sleep(0.5)
    #init_ws = np.load('init_ws_100.npy')

    random_probs = [None, 1, 0, 0.86]
    if multi_thread:
        return_dict = manager.dict()
        jobs = []
        for rp in random_probs:
            time.sleep(0.5)
            if use_tf:
                p = Process(target = learn_thread_tf, args=(config_T, config_L, mode, train_iter, rp, rp, return_dict))
            else:
                student = LearnerIRL(sess, map_l, config_L)
                p = Process(target = learn_thread, args = (teacher, student, mode, init_ws, train_iter, test_set, rp, rp, return_dict))
            p.start()
            jobs.append(p)
        student = LearnerIRL(sess, map_l, config_L)
        p = Process(target = learn_thread, args = (teacher, student, '%s_cont' % mode, init_ws, train_iter,
                                                   test_set, None, '%s_cont' % mode, return_dict))
        p.start()
        jobs.append(p)

        student = LearnerIRL(sess, map_l, config_L)
        p = Process(target = learn_thread, args = (teacher, student, '%s_sgd_cont' % mode, init_ws, train_iter,
                                                   test_set, None, '%s_sgd_cont' % mode, return_dict))
        p.start()
        jobs.append(p)

        for j in jobs:
            j.join()
        dists0, dists0_, distsq0, ar0, _ = return_dict[random_probs[0]]
        dists1, dists1_, distsq1, ar1, _ = return_dict[random_probs[1]]
        dists2, dists2_, distsq2, ar2, _ = return_dict[random_probs[2]]
        dists3, dists3_, distsq3, ar3, _ = return_dict[None]
        dists4, dists4_, distsq4, ar4, _ = return_dict['%s_cont' % mode]
        dists5, dists5_, distsq5, ar5, _ = return_dict['%s_sgd_cont' % mode]

    import tensorflow as tf
    tfconfig = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    tfconfig.gpu_options.allow_growth = True
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    sess = tf.Session(config = tfconfig)
    np.random.seed(400)

    map_l = Map(sess, config_L)
    map_t = Map(sess, config_T)

    gt_r_param_tea = map_l.reward_generate(3) if reward_type == 'E' else np.random.uniform(-2, 2, size = [1, shape ** 2])
    gt_r_param_stu = copy.deepcopy(gt_r_param_tea)
    if config_L.shuffle_state_feat:
        gt_r_param_stu[:, map_l.feat_idx_] = gt_r_param_tea
    assert(np.max(abs(np.sum(gt_r_param_stu * map_l.state_feats_, axis = 1) - np.sum(gt_r_param_tea * map_t.state_feats_, axis = 1))) < 1e-9)

    teacher = TeacherIRL(sess, map_t, config_T, gt_r_param_tea, gt_r_param_stu)

    init_ws = np.random.uniform(-2, 2, size = [config_L.particle_num, teacher.map_.num_states_])
    test_set = np.random.choice(teacher.map_.num_states_, size = [train_iter + 1, teacher.map_.num_states_ * 20])

    learner = LearnerIRL(sess, map_l, config_L)

    if not multi_thread:
        dists_batch, dists_batch_, distsq_batch, ar_batch, _ = learn_basic(teacher, learner, train_iter, init_ws, test_set, True)
        
        dists_sgd, dists_sgd_, distsq_sgd, ar_sgd, _ = learn_basic(teacher, learner, train_iter, init_ws, test_set, False)
        
        dists4, dists4_, distsq4, ar4, _ = learn(teacher, learner, '%s_cont' % mode, init_ws, train_iter,
                                                 test_set)
        dists5, dists5_, distsq5, ar5, _ = learn(teacher, learner, '%s_sgd_cont' % mode, init_ws, train_iter,
                                                 test_set)
        np.save('distsbatch_' + title + '.npy', np.array(dists_batch))
        np.save('distbatch__' + title + '.npy', np.array(dists_batch_))
        np.save('distsqbatch_' + title + '.npy', np.array(distsq_batch))
        np.save('arbatch_' + title + '.npy', np.array(ar_batch))

        np.save('distssgd_' + title + '.npy', np.array(dists_sgd))
        np.save('distsgd__' + title + '.npy', np.array(dists_sgd_))
        np.save('distsqsgd_' + title + '.npy', np.array(distsq_sgd))
        np.save('arsgd_' + title + '.npy', np.array(ar_sgd))

        np.save('dists4_' + title + '.npy', np.array(dists4))
        np.save('dist4__' + title + '.npy', np.array(dists4_))
        np.save('distsq4_' + title + '.npy', np.array(distsq4))
        np.save('ar4_' + title + '.npy', np.array(ar4))

        np.save('dists5_' + title + '.npy', np.array(dists5))
        np.save('dist5__' + title + '.npy', np.array(dists5_))
        np.save('distsq5_' + title + '.npy', np.array(distsq5))
        np.save('ar5_' + title + '.npy', np.array(ar5))

    return

if __name__ == '__main__':
    main()
