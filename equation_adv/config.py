lr = 1e-3
beta = -5000
K = 1
multi_thread = True
dd = 45
dd_ = -1
num_classes = 1
dps = 3 * dd
reg_coef = 0
num_particles = 1

train_iter_simple = 50
train_iter_smart = 50

noise_scale_min = {}
noise_scale_max = {}
noise_scale_decay = {}
noise_scale_min['omni'] = 0
noise_scale_max['omni'] = 0.05
noise_scale_decay['omni'] = 1000

noise_scale_min['imit'] = 0
noise_scale_max['imit'] = 0.05
noise_scale_decay['imit'] = 1000

task = 'regression'
