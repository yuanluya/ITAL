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

def learn(teacher, learner, mode, init_ws, train_iter, test_set, random_prob = None):
    np.random.seed(555)
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
        # if (i + 1) % 100 == 0:
        #     learner.lr_ /= 2
        # if i == 100 and random_prob is None:
        #     # np.save('init_ws_100.npy', learner.particles_)
        #     learner.particles_ = np.load('init_ws_100.npy')
        #     learner.current_mean_ = np.mean(learner.particles_, axis = 0, keepdims = True)
        #     np.random.seed(999)
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
    np.random.seed(400)

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
    np.random.seed(400)

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
    multi_thread = True

    mode_idx = int(sys.argv[1])
    modes = ['omni', 'imit']
    mode = modes[mode_idx]
    shape = 8
    lr = 1e-3
    beta = 1
    beta_select = 10000
    K = 20
    train_iter = 100

    reward_type = sys.argv[2]
    approx_k = 200 if reward_type == 'E' else 220

    config_T = edict({'shape': shape, 'approx_type': 'gsm', 'beta': beta, 'shuffle_state_feat': False,
                      'lr': lr, 'sample_size': 20, 'use_tf': use_tf, 'approx_k': approx_k, 'beta_select': beta_select})
    config_L = edict({'shape': shape, 'approx_type': 'gsm', 'beta': beta, 'lr': lr, "prob": 1,
                      'shuffle_state_feat': mode == 'imit', 'particle_num': 1000, 'replace_count': 1,
                      'noise_scale_min': 0.00, 'noise_scale_max': 0.2, 'noise_scale_decay': 70, 'cont_K': K,
                      'target_ratio': 0, 'new_ratio': 1, 'use_tf': use_tf, 'approx_k': approx_k, 'beta_select': beta_select})

    np.set_printoptions(precision = 4)

    if not use_tf:
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
        dists3, dists3_, distsq3, ar3, eliminates = learn(teacher, learner, mode, init_ws,
                                                          train_iter, test_set, random_prob = None)
        dists0, dists0_, distsq0, ar0, _ = learn(teacher, learner, mode, init_ws, train_iter,
                                                 test_set, random_prob = random_probs[0])
        dists1, dists1_, distsq1, ar1, _ = learn(teacher, learner, mode, init_ws, train_iter,
                                                 test_set, random_prob = random_probs[1])
        dists2, dists2_, distsq2, ar2, _ = learn(teacher, learner, mode, init_ws, train_iter,
                                                 test_set, random_prob = np.mean(eliminates) / config_L.particle_num)
        dists4, dists4_, distsq4, ar4, _ = learn(teacher, learner, '%s_cont' % mode, init_ws, train_iter,
                                                 test_set)
        dists5, dists5_, distsq5, ar5, _ = learn(teacher, learner, '%s_sgd_cont' % mode, init_ws, train_iter,
                                                 test_set)

    fig, axs = plt.subplots(2, 2, constrained_layout = True)

    line0, = axs[0, 0].plot(dists0, label = 'zero')
    line1, = axs[0, 0].plot(dists1, label = 'one')
    line2, = axs[0, 0].plot(dists2, label = 'random')
    line3, = axs[0, 0].plot(dists3, label = 'smarter')
    line4, = axs[0, 0].plot(dists4, label = 'cont')
    line5, = axs[0, 0].plot(dists5, label = 'sgd_cont')
    axs[0, 0].set_title('action prob total variance distance')

    line0, = axs[0, 1].plot(dists0_, label = 'zero')
    line1, = axs[0, 1].plot(dists1_, label = 'one')
    line2, = axs[0, 1].plot(dists2_, label = 'random')
    line3, = axs[0, 1].plot(dists3_, label = 'smarter')
    line4, = axs[0, 1].plot(dists4_, label = 'cont')
    line5, = axs[0, 1].plot(dists5_, label = 'sgd_cont')
    axs[0, 1].set_title('reward param l2 distance')
    axs[0, 1].legend([line0, line1, line2, line3, line4, line5],
                     ['zero', 'one', 'random', 'pragmatic', '%s_cont' % mode, 'sgd cont'])

    line0, = axs[1, 1].plot(distsq0, label = 'zero')
    line1, = axs[1, 1].plot(distsq1, label = 'one')
    line2, = axs[1, 1].plot(distsq2, label = 'random')
    line3, = axs[1, 1].plot(distsq3, label = 'smarter')
    line4, = axs[1, 1].plot(distsq4, label = 'cont')
    line5, = axs[1, 1].plot(distsq5, label = 'sgd_cont')
    axs[1, 1].set_title('q function l2 distance')

    line0, = axs[1, 0].plot(ar0, label = 'zero')
    line1, = axs[1, 0].plot(ar1, label = 'one')
    line2, = axs[1, 0].plot(ar2, label = 'random')
    line3, = axs[1, 0].plot(ar3, label = 'smarter')
    line4, = axs[1, 0].plot(ar4, label = 'cont')
    line5, = axs[1, 0].plot(ar5, label = 'sgd_cont')
    p.join()
    axs[1, 0].plot([np.mean(return_list[0])] * len(ar3), label = 'teacher')
    axs[1, 0].set_title('actual rewards')

    fig.suptitle('%s shape: %d, beta: %.2f, data:%d/%d/%d_particle:%d_noise: %.2f, %.2f, %d,'
                 'ratio: %.2f, %.2f, lr: %f, beta_select: %.2f, K: %d' %\
              (mode, shape, beta, config_L.replace_count, config_T.sample_size, (shape ** 2), config_L.particle_num,
               config_L.noise_scale_min, config_L.noise_scale_max, config_L.noise_scale_decay,
               config_L.target_ratio, config_L.new_ratio, lr, config_L.beta_select, config_L.cont_K if K is not None else -999))

    plt.show()
    pdb.set_trace()

    return

if __name__ == '__main__':
    main()
