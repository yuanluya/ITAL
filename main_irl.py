import copy
import sys
import numpy as np
from easydict import EasyDict as edict
from tqdm import tqdm

from teacher_irl import TeacherIRL
from learner_irl import LearnerIRL

import pdb

def main():
    mode_idx = int(sys.argv[1])
    modes = ['omni', 'imit']
    mode = modes[mode_idx]
    shape = 8
    config_T = edict({'shape': shape, 'approx_type': 'gsm', 'beta': 10.0, 'shuffle_state_feat': False, 'sample_size': 20})
    config_L = edict({'shape': shape, 'approx_type': 'gsm', 'beta': 10.0, 'lr': 1e-3, "prob": 1,
                      'shuffle_state_feat': mode == 'imit', 'particle_num': 500, 'replace_count': 1,
                      'noise_scale_min': 0.01, 'noise_scale_max': 0.1, 'noise_scale_decay': 1000})
    train_iter = 2000

    gt_r_param_tea = np.zeros(shape = [1, shape ** 2])
    gt_r_param_tea[0, shape * shape - 1] = 10
    teacher = TeacherIRL(config_T, gt_r_param_tea)
    learner = LearnerIRL(config_L)

    gt_r_param_stu = gt_r_param_tea[:, learner.map_.feat_idx_] if config_L.shuffle_state_feat else copy.deepcopy(gt_r_param_tea)

    init_ws = np.random.uniform(-1, 1, size = [config_L.particle_num, teacher.map_.num_states_])
    learner.reset(init_ws)

    for i in tqdm(range(train_iter)):
        teacher.sample()
        data_idx, gradients = teacher.choose(np.mean(init_ws, axis = 0, keepdims = True), config_L.lr)
        learner.learn(teacher.mini_batch_indices_, teacher.mini_batch_opt_acts_, data_idx, gradients, i, gt_r_param_stu)

    stu_rewards = np.sum(learner.map_.state_feats_ * learner.current_mean_, axis = 1)
    teacher.choose_imit(stu_rewards, config_L.lr)



    return

if __name__ == '__main__':
    main()