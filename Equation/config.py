from easydict import EasyDict as edict
import os

config = edict()

config.num_character = 18
config.encoding_dims = 20
config.rnn_dim = 30
config.lr = 5e-5

config.dir_path = '../../../Datasets/equation_500/'
config.ckpt_dir = 'CKPT_rnn_dim_30_lr_5e-5_encoding_dims_20_2_4_neg'

config.data_size = 500000
config.test_size = 10000

config.train_iter = 15000
