import copy
import sys
import numpy as np
from easydict import EasyDict as edict

from teacher_irl import TeacherIRL

def main():
    mode_idx = int(sys.argv[1])
    modes = ['omni', 'imit']
    mode = modes[mode_idx]
    shape = 8
    config_T = edict({'shape': shape, 'approx_type': 'gsm', 'beta': 10.0, 'shuffle_state_feat': False, 'sample_size': 20})
    config_L = edict({'shape': shape, 'approx_type': 'gsm', 'beta': 10.0,
                      'shuffle_state_feat': mode == 'imit', 'num_particles': 1000, 'lr': 1e-3})

    gt_r_param_tea = np.zeros(shape = [1, shape ** 2])
    gt_r_param_tea[0, shape * shape - 1] = 10
    teacher = TeacherIRL(config_T, gt_r_param_tea)

    gt_r_param_stu = gt_r_param_tea[game_map_learner.feat_idx_] if config_L.shuffle_state_feat else copy.deepcopy(gt_r_param_tea)

    init_ws = np.random.uniform(-1, 1, size = [config_L.num_particles, teacher.map_.num_states_])

    teacher.sample()
    teacher.choose(np.mean(init_ws, axis = 0, keepdims = True), config_L.lr)

    stu_rewards = np.sum(teacher.map_.state_feats_ * np.mean(init_ws, axis = 0, keepdims = True), axis = 1)
    teacher.choose_imit(stu_rewards, config_L.lr)



    return

if __name__ == '__main__':
    main()