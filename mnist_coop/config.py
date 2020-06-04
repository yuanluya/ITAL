lr = 1e-3
beta = 30001
K = 1
multi_thread = True
dd = 24
num_classes = 10
dps = 3 * dd
reg_coef = 0
num_particles = 1

train_iter_simple = 50
train_iter_smart = 50

noise_scale_min = {}
noise_scale_max = {}
noise_scale_decay = {}
noise_scale_min['omni'] = 0.01
noise_scale_max['omni'] = 0.1
noise_scale_decay['omni'] = 200

noise_scale_min['imit'] = 0.02
noise_scale_max['imit'] = 0.1
noise_scale_decay['imit'] = 1000

task = 'classification'
