from easydict import EasyDict as edict
import numpy as np

lr = 1e-3
beta = 1000
beta_decay = 1
K = 1
multi_thread = True
dd = 10
dd_ = 13
num_classes = 200
dps = 3 * dd
reg_coef = 0
num_particles = 1
accuracy_top_K = 5

train_iter_simple = 2000
train_iter_smart = 2000

noise_scale_min = 0.01
noise_scale_max = 0.1
noise_scale_decay = 1000

task = 'classification'
mode = 'imit'

dx_tea = np.load("Data/ImageNet_10/ImageNet_train_features16.npy")
dy_tea = np.load("Data/ImageNet_10/ImageNet_train_labels.npy")
gt_w_tea = np.load("Data/ImageNet_10/ImageNet_gt_weights16.npy")
tx_tea = np.load("Data/ImageNet_10/ImageNet_test_features16.npy")
ty_tea = np.load("Data/ImageNet_10/ImageNet_test_labels.npy")

dx = np.load("Data/ImageNet_10/ImageNet_train_features%d.npy" % dd_) 
dy = np.load("Data/ImageNet_10/ImageNet_train_labels.npy")
gt_w = np.load("Data/ImageNet_10/ImageNet_gt_weights%d.npy" % dd_)  
tx = np.load("Data/ImageNet_10/ImageNet_test_features%d.npy" % dd_)  
ty = np.load("Data/ImageNet_10/ImageNet_test_labels.npy") 

config_T = edict({'data_pool_size_class': dps, 'data_dim': dd,'lr': lr, 'sample_size': 20,
                  'transform': mode == 'imit', 'num_classes': num_classes, 'task': task,
                  'data_x': dx, 'data_y': dy, 'test_x': tx, 'test_y': ty, 'gt_w': gt_w, 'beta': beta,
                  'data_x_tea': dx_tea, 'data_y_tea': dy_tea, 'test_x_tea': tx_tea, 'test_y_tea': ty_tea, 'gt_w_tea': gt_w_tea})

config_LS = edict({'particle_num': num_particles, 'data_dim': dd, 'reg_coef': reg_coef, 'lr': lr, 'task': task, 'beta_decay': beta_decay,
                   'num_classes': num_classes, 'noise_scale_min': noise_scale_min, 'noise_scale_max': noise_scale_max, 'beta': beta, 'cont_K': K,
                   'noise_scale_decay': noise_scale_decay, 'target_ratio': 0, 'new_ratio': 1,
                   'replace_count': 1, "prob": 1, 'accuracy_top_K': accuracy_top_K})


config_L =  edict({'data_dim': dd, 'reg_coef': reg_coef, 'lr': lr, 'loss_type': 0,
                   'num_classes': num_classes, 'task': task, 'accuracy_top_K': accuracy_top_K})

