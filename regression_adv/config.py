lr = 1e-3
beta = -5000
K = 1
multi_thread = True
dd = 100
dd_ = -1
num_classes = 1
dps = 6 * dd
reg_coef = 0
num_particles = 1

train_iter_simple = 50
train_iter_smart = 50

noise_scale_min = {}
noise_scale_max = {}
noise_scale_decay = {}
noise_scale_min['omni'] = 0.1
noise_scale_max['omni'] = 0.3
noise_scale_decay['omni'] = 300

noise_scale_min['imit'] = 0.1
noise_scale_max['imit'] = 0.3
noise_scale_decay['imit'] = 200

task = 'regression'