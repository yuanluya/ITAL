from easydict import EasyDict as edict
import numpy as np

lr = 1e-3
beta = -30001
K = 1
multi_thread = True
dd = 24
dd_ = 30
num_classes = 10
dps = 3 * dd
reg_coef = 0
num_particles = 1
mode = 'imit'
train_iter_simple = 50
train_iter_smart = 50

noise_scale_min = 0.02
noise_scale_max = 0.1
noise_scale_decay = 1000

task = 'classification'

dx = np.load("Data/MNIST/mnist_train_features.npy") 
dy = np.load("Data/MNIST/mnist_train_labels.npy") 
gt_w = np.load("Data/MNIST/mnist_tf_gt_weights.npy")
tx = np.load("Data/MNIST/mnist_test_features.npy")
ty = np.load("Data/MNIST/mnist_test_labels.npy")

dx_tea = np.load("Data/MNIST/mnist_train_features_tea_%d.npy" % dd_)
dy_tea = np.load("Data/MNIST/mnist_train_labels_tea_%d.npy" % dd_) 
gt_w_tea = np.load("Data/MNIST/mnist_tf_gt_weights_tea_%d.npy" % dd_)
tx_tea = np.load("Data/MNIST/mnist_test_features_tea_%d.npy" % dd_)
ty_tea = np.load("Data/MNIST/mnist_test_labels_tea_%d.npy" % dd_)

config_T = edict({'data_pool_size_class': dps, 'data_dim': dd,'lr': lr, 'sample_size': 20,
                  'transform': mode == 'imit', 'num_classes': num_classes, 'task': task,
                  'data_x': dx, 'data_y': dy, 'test_x': tx, 'test_y': ty, 'gt_w': gt_w, 'beta': beta,
                  'data_x_tea': dx_tea, 'data_y_tea': dy_tea, 'test_x_tea': tx_tea, 'test_y_tea': ty_tea, 'gt_w_tea': gt_w_tea})

config_LS = edict({'particle_num': num_particles, 'data_dim': dd, 'reg_coef': reg_coef, 'lr': lr, 'task': task,
                   'num_classes': num_classes, 'noise_scale_min': noise_scale_min, 'noise_scale_max': noise_scale_max, 'beta': beta, 'cont_K': K,
                   'noise_scale_decay': noise_scale_decay, 'target_ratio': 0, 'new_ratio': 1, 'replace_count': 1, "prob": 1})


config_L =  edict({'data_dim': dd, 'reg_coef': reg_coef, 'lr': lr, 'loss_type': 0, 'num_classes': num_classes, 'task': task})

