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
    np.random.seed((int(sys.argv[2]) + 1)* 503)
    learner.reset(init_ws)

    ws = [learner.current_mean_]
    dists = [np.mean(np.max(abs(learner.current_action_prob() - teacher.action_probs_), axis = 1))]
    dists_ = [np.sum(np.square(learner.current_mean_ - teacher.stu_gt_reward_param_))]
    distsq = [np.mean(np.square(learner.q_map_ - teacher.q_map_))]
    actual_rewards = [teacher.map_.test_walk(teacher.reward_param_, learner.action_probs_, test_set[0], greedy = True)]
    for i in tqdm(range(train_iter)):
        teacher.sample()
        if not batch:
            data_idx = np.random.randint(teacher.config_.sample_size)
        else:
            data_idx = np.arange(teacher.config_.sample_size)
        _, gradients = teacher.choose(learner.current_mean_, learner.lr_)
        w = learner.learn_cont(teacher.mini_batch_indices_, teacher.mini_batch_opt_acts_, data_idx,
                                         gradients, i, teacher.stu_gt_reward_param_, -1, batch = batch)
        ws.append(w)
        dists_.append(np.sum(np.square(learner.current_mean_ - teacher.stu_gt_reward_param_)))
        dists.append(np.mean(np.max(abs(learner.current_action_prob() - teacher.action_probs_), axis = 1)))
        distsq.append(np.mean(np.square(learner.q_map_ - teacher.q_map_)))
        if (i + 1) % 20 == 0:
            actual_rewards.append(teacher.map_.test_walk(teacher.reward_param_, learner.action_probs_, test_set[i + 1], greedy = True))
    learner.lr_ = learner.config_.lr
    return dists, dists_, distsq, actual_rewards, ws

def learn(teacher, learner, mode, init_ws, train_iter, test_set, random_prob = None):
    np.random.seed((int(sys.argv[2]) + 1) * 503)

    learner.reset(init_ws)

    dists_ = [np.sum(np.square(learner.current_mean_ - teacher.stu_gt_reward_param_))]
    dists = [np.mean(np.max(abs(learner.current_action_prob() - teacher.action_probs_), axis = 1))]
    distsq = [np.mean(np.square(learner.q_map_ - teacher.q_map_))]
    actual_rewards = [teacher.map_.test_walk(teacher.reward_param_, learner.action_probs_, test_set[0], greedy = True)]
    ws = []
    for i in tqdm(range(train_iter)):
        teacher.sample()
        if mode[0: 4] == 'omni':
            data_idx, gradients = teacher.choose(learner.current_mean_, learner.lr_, hard = True)
        elif mode[0: 4] == 'imit':
            stu_rewards = np.sum(learner.map_.state_feats_ * learner.current_mean_, axis = 1, keepdims = True)
            data_idx, gradients, l_stu = teacher.choose_imit(stu_rewards, learner.lr_, hard = True)

        if mode == 'omni' or random_prob is not None:
            w = learner.learn(teacher.mini_batch_indices_, teacher.mini_batch_opt_acts_, data_idx,
                                         gradients, i, teacher.stu_gt_reward_param_, random_prob)
        elif mode == 'omni_cont':
            w = learner.learn_cont(teacher.mini_batch_indices_, teacher.mini_batch_opt_acts_, data_idx,
                                         gradients, i, teacher.stu_gt_reward_param_, learner.config_.cont_K)
        elif mode == 'imit':
            w = learner.learn_imit(teacher.mini_batch_indices_, teacher.mini_batch_opt_acts_, data_idx,
                                              l_stu, i, teacher.stu_gt_reward_param_)
        elif mode == 'imit_cont':
            w = learner.learn_imit_cont(teacher.mini_batch_indices_, teacher.mini_batch_opt_acts_, data_idx,
                                                   l_stu, i, teacher.stu_gt_reward_param_, learner.config_.cont_K)
        dists_.append(np.sum(np.square(learner.current_mean_ - teacher.stu_gt_reward_param_)))
        dists.append(np.mean(np.max(abs(learner.current_action_prob() - teacher.action_probs_), axis = 1)))
        distsq.append(np.mean(np.square(learner.q_map_ - teacher.q_map_)))

        ws.append(copy.deepcopy(w))
        if (i + 1) % 20 == 0:
            actual_rewards.append(teacher.map_.test_walk(teacher.reward_param_, learner.action_probs_, test_set[i + 1], greedy = True))
    learner.lr_ = learner.config_.lr
    if (mode == "omni_cont"):
        np.save('action_probs.npy', learner.action_probs_)
    return dists, dists_, distsq, actual_rewards, ws

def learn_thread(teacher, learner, mode, init_ws, train_iter, test_set, random_prob, dict_key, thread_return):
    dists, dists_, distsq, actual_rewards, ws = learn(teacher, learner, mode, init_ws, train_iter, test_set, random_prob)
    thread_return[dict_key] = [dists, dists_, distsq, actual_rewards, ws]

def learn_thread_tf(config_T, config_L, mode, train_iter, random_prob, return_key, thread_return):
    import tensorflow as tf
    tfconfig = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    tfconfig.gpu_options.allow_growth = True
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    sess = tf.Session(config = tfconfig)

    map_l = Map(sess, config_L)
    map_t = Map(sess, config_T)

    reward_type = config.reward_type

    gt_r_param_tea = map_l.reward_generate(3) if config.reward_type == 'E' else np.random.uniform(-2, 2, size = [1, config_L.shape ** 2])
    gt_r_param_stu = copy.deepcopy(gt_r_param_tea)
    if config_L.shuffle_state_feat:
        gt_r_param_stu[:, map_l.feat_idx_] = gt_r_param_tea
    assert(np.max(abs(np.sum(gt_r_param_stu * map_l.state_feats_, axis = 1) - np.sum(gt_r_param_tea * map_t.state_feats_, axis = 1))) < 1e-9)

    teacher = TeacherIRL(sess, map_t, config_T, gt_r_param_tea, gt_r_param_stu)

    init_ws = np.random.uniform(-2, 2, size = [config_L.particle_num, teacher.map_.num_states_])
    test_set = np.random.choice(teacher.map_.num_states_, size = [train_iter + 1, teacher.map_.num_states_ * 20])
    #pdb.set_trace()
    learner = LearnerIRL(sess, map_l, config_L)

    dists, dists_, distsq, actual_rewards, ws = learn(teacher, learner, mode, init_ws, train_iter, test_set, random_prob)
    thread_return[return_key] = [dists, dists_, distsq, actual_rewards, ws]

def teacher_run_tf(config_T, config_L, train_iter, thread_return):

    import tensorflow as tf
    tfconfig = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    tfconfig.gpu_options.allow_growth = True
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    sess = tf.Session(config = tfconfig)

    map_l = Map(sess, config_L)
    map_t = Map(sess, config_T)


    reward_type = config.reward_type
    gt_r_param_tea = map_l.reward_generate(3) if config.reward_type == 'E' else np.random.uniform(-2, 2, size = [1, config_T.shape ** 2])
    gt_r_param_stu = copy.deepcopy(gt_r_param_tea)
    if config_L.shuffle_state_feat:
        gt_r_param_stu[:, map_l.feat_idx_] = gt_r_param_tea
    assert(np.max(abs(np.sum(gt_r_param_stu * map_l.state_feats_, axis = 1) - np.sum(gt_r_param_tea * map_t.state_feats_, axis = 1))) < 1e-9)
    teacher = TeacherIRL(sess, map_t, config_T, gt_r_param_tea, gt_r_param_stu)

    init_ws = np.random.uniform(-2, 2, size = [config_L.particle_num, teacher.map_.num_states_])
    test_set = np.random.choice(teacher.map_.num_states_, size = [train_iter + 1, teacher.map_.num_states_ * 20])
    student = LearnerIRL(sess, map_l, config_L)

    teacher_rewards = [teacher.map_.test_walk(teacher.reward_param_, teacher.action_probs_, test_set[i + 1], greedy = True)\
                       for i in range(0, train_iter, 2)]
    thread_return.append(teacher_rewards)

def main():
    exp_folder = sys.argv[1]
    if not os.path.isdir(os.path.join('./Experiments', exp_folder)):
        print('Cannot find target folder')
        exit()
    if not os.path.exists(os.path.join('./Experiments', exp_folder, 'config.py')):
        print('Cannot find config.py in target folder')
        exit()
    exec('from Experiments.%s import config' % exp_folder, globals())
    exec('from Experiments.%s.config import config_T' % exp_folder, globals())
    exec('from Experiments.%s.config import config_L' % exp_folder, globals())

    use_tf = config.use_tf
    multi_thread = config.multi_thread
    mode = config.mode
    directory = sys.argv[1] + '/'

    train_iter = config.train_iter

    seed = int(sys.argv[2])
    np.random.seed((seed + 1) * 159)

    np.set_printoptions(precision = 4)

    if not use_tf:
        np.random.seed((seed + 1) * 157)

        import tensorflow as tf
        tfconfig = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
        tfconfig.gpu_options.allow_growth = True
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        sess = tf.Session(config = tfconfig)

        map_l = Map(sess, config_L)
        np.random.seed((seed + 1) * 163)

        map_t = Map(sess, config_T)
        np.random.seed((seed + 1) * 174)

        gt_r_param_tea = map_l.reward_generate(3) if config.reward_type == 'E' else np.random.uniform(-2, 2, size = [1, config.shape ** 2])
        gt_r_param_stu = copy.deepcopy(gt_r_param_tea)
        if config_L.shuffle_state_feat:
            #print("Shuffling with ", map_l.feat_idx_)
            gt_r_param_stu[:, map_l.feat_idx_] = gt_r_param_tea

        assert(np.max(abs(np.sum(gt_r_param_stu * map_l.state_feats_, axis = 1) - np.sum(gt_r_param_tea * map_t.state_feats_, axis = 1))) < 1e-9)

        np.random.seed((seed + 1) * 105)
        teacher = TeacherIRL(sess, map_t, config_T, gt_r_param_tea, gt_r_param_stu)
        init_ws = np.random.uniform(-2, 2, size = [config_L.particle_num, teacher.map_.num_states_])
        unshuffled_ws = copy.deepcopy(init_ws)
        if config_L.shuffle_state_feat:
            init_ws[:, map_l.feat_idx_] = unshuffled_ws

        test_set = np.random.choice(teacher.map_.num_states_, size = [train_iter + 1, teacher.map_.num_states_ * 20])

    manager = Manager()
    
    if mode == 'omni':
        '''return_list = manager.list()'''

        teacher_rewards = []
        for i in tqdm(range(0, train_iter, 20)):
            teacher_rewards.append(teacher.map_.test_walk(teacher.reward_param_, teacher.action_probs_, test_set[i + 1], greedy = True))
        teacher_reward = np.asarray([np.mean(teacher_rewards)])
        np.save('Experiments/' + directory + "teacher_rewards_%d" % (seed), teacher_rewards, allow_pickle=True)

    random_probs = [1]
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

        for j in jobs:
            j.join()

        imt = return_dict[random_probs[0]]
        prag_cont = return_dict['%s_cont' % mode]

    np.random.seed((seed + 1) * 157)

    map_l = Map(sess, config_L)
    np.random.seed((seed + 1) * 163)

    map_t = Map(sess, config_T)
    np.random.seed((seed + 1) * 174)

    gt_r_param_tea = map_l.reward_generate(3) if config.reward_type == 'E' else np.random.uniform(-2, 2, size = [1, config.shape ** 2])
    gt_r_param_stu = copy.deepcopy(gt_r_param_tea)
    if config_L.shuffle_state_feat:
        gt_r_param_stu[:, map_l.feat_idx_] = gt_r_param_tea
    assert(np.max(abs(np.sum(gt_r_param_stu * map_l.state_feats_, axis = 1) - np.sum(gt_r_param_tea * map_t.state_feats_, axis = 1))) < 1e-9)

    np.random.seed((seed + 1) * 105)

    teacher = TeacherIRL(sess, map_t, config_T, gt_r_param_tea, gt_r_param_stu)

    init_ws = np.random.uniform(-2, 2, size = [config_L.particle_num, teacher.map_.num_states_])

    test_set = np.random.choice(teacher.map_.num_states_, size = [train_iter + 1, teacher.map_.num_states_ * 20])
    learner = LearnerIRL(sess, map_l, config_L)

    batch = []
    sgd = []
    
    batch = learn_basic(teacher, learner, train_iter, init_ws, test_set, True)
    sgd = learn_basic(teacher, learner, train_iter, init_ws, test_set, False)
    #dists_batch, dists_batch_, distsq_batch, ar_batch, _ = learn_basic(teacher, learner, train_iter, init_ws, test_set, True)
    #dists_sgd, dists_sgd_, distsq_sgd, ar_sgd, _ = learn_basic(teacher, learner, train_iter, init_ws, test_set, False)

    results = [imt, prag_cont, batch, sgd]
    n = 4
    for i in range(n):
        dists = results[i][0]
        dists_ = results[i][1]
        distsq = results[i][2]
        ar = results[i][3]
        mat = results[i][4]
        np.save('Experiments/' + directory + "action_dist%d_%d" % (i, seed), dists, allow_pickle=True)
        np.save('Experiments/' + directory + "reward_dist%d_%d" % (i, seed), np.sqrt(dists_), allow_pickle=True)
        np.save('Experiments/' + directory + "q_dist%d_%d" % (i, seed), distsq, allow_pickle=True)
        np.save('Experiments/' + directory + "rewards%d_%d" % (i, seed), ar, allow_pickle=True)
        np.save('Experiments/' + directory + "matrix%d_%d" % (i, seed), mat, allow_pickle = True)

if __name__ == '__main__':
    main()
