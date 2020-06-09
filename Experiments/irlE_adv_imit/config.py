from easydict import EasyDict as edict

use_tf = False
multi_thread = True

shape = 8
lr = 1e-3
beta = 5
reward_type = 'H'
approx_k = 220

beta_select = -30000
K = 1
train_iter = 50

mode = 'imit'

noise_scale_min = 0
noise_scale_max = 0.3
noise_scale_decay = 200

config_T = edict({'shape': shape, 'approx_type': 'gsm', 'beta': beta, 'shuffle_state_feat': False,
                  'lr': lr, 'sample_size': 20, 'use_tf': use_tf, 'approx_k': approx_k, 'beta_select': beta_select})
config_L = edict({'shape': shape, 'approx_type': 'gsm', 'beta': beta, 'lr': lr, "prob": 1,
                  'shuffle_state_feat': mode == 'imit', 'particle_num': 1, 'replace_count': 1,
                  'noise_scale_min': noise_scale_min, 'noise_scale_max': noise_scale_max, 'noise_scale_decay': noise_scale_decay, 'cont_K': K,
                  'target_ratio': 0, 'new_ratio': 1, 'use_tf': use_tf, 'approx_k': approx_k, 'beta_select': beta_select})
