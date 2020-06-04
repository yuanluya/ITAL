use_tf = False
multi_thread = True

shape = 8
lr = 1e-3
beta = 5

reward_type = 'H'
approx_k = 220

beta_select = 10000
K = 1
train_iter = 21

noise_scale_min = {}
noise_scale_max = {}
noise_scale_decay = {}
noise_scale_min['omni'] = 0
noise_scale_max['omni'] = 0.3
noise_scale_decay['omni'] = 200

noise_scale_min['imit'] = 0.005
noise_scale_max['imit'] = 0.3
noise_scale_decay['imit'] = 200
