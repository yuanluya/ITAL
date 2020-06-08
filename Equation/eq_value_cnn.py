import tensorflow as tf
import numpy as np
from easydict import EasyDict as edict
import os

import pdb

class EqValue:
    def __init__(self, config, init_w, sess):
        self.config_ = config
        self.init_w_ = init_w
        self.sess_ = sess

        self.lower_eqs_idx_ = tf.placeholder(tf.int32, shape = [None, self.config_.input_dim, 1])
        self.higher_eqs_idx_ = tf.placeholder(tf.int32, shape = [None, self.config_.input_dim, 1])
        self.lower_encoding_idx_ = tf.placeholder(tf.int32, shape = [None, 1])
        self.higher_encoding_idx_ = tf.placeholder(tf.int32, shape = [None, 1])

        self.codebook_ = tf.get_variable('codebook', shape = [self.config_.num_character, self.config_.encoding_dim],
                                         dtype = tf.float32, initializer = tf.random_normal_initializer())
        self.codebook_0_ = tf.concat([self.codebook_, tf.zeros(shape = [1, self.config_.encoding_dim])], 0)
        self.lower_eqs_ = tf.expand_dims(tf.squeeze(tf.nn.embedding_lookup(self.codebook_0_, self.lower_eqs_idx_), 2), 1)
        self.higher_eqs_ = tf.expand_dims(tf.squeeze(tf.nn.embedding_lookup(self.codebook_0_, self.higher_eqs_idx_), 2), 1)

        self.layers_ = [tf.concat([self.lower_eqs_, self.higher_eqs_], axis = 0)]
        for _, (out_dim, kernel_size, stride, pool) in enumerate(self.config_.layer_info):
            out = tf.layers.conv2d(self.layers_[-1], out_dim, kernel_size = kernel_size, strides = stride, padding = 'same', 
                                      activation = tf.nn.leaky_relu, kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 5e-2))
            if pool:
                out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME')
            self.layers_.append(out)

        self.encodings_ = tf.layers.dense(tf.layers.flatten(self.layers_[-1]), units = self.config_.output_dim, activation = tf.nn.tanh)
        self.lower_encodings_0_ = tf.gather_nd(self.encodings_, self.lower_encoding_idx_)
        self.lower_encodings_ = tf.concat([self.lower_encodings_0_, tf.ones([tf.shape(self.lower_encodings_0_)[0], 1])], 1)
        self.higher_encodings_0_ = tf.gather_nd(self.encodings_, self.higher_encoding_idx_)
        self.higher_encodings_ = tf.concat([self.higher_encodings_0_, tf.ones([tf.shape(self.higher_encodings_0_)[0], 1])], 1)

        self.weight_ = tf.Variable(initial_value = self.init_w_, name = 'weight', dtype = tf.float32)
        self.lower_vals_ = tf.reduce_sum(self.lower_encodings_ * self.weight_, 1)
        self.higher_vals_ = tf.reduce_sum(self.higher_encodings_ * self.weight_, 1)
        self.diff_vals_ = tf.reduce_sum((self.higher_encodings_ - self.lower_encodings_) * self.weight_, 1)

        self.reg_loss_ = tf.add_n([tf.nn.l2_loss(v) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'bias' not in v.name])
        self.loss_ = 0.5 * tf.reduce_sum(tf.square(self.weight_)) +\
            self.config_.C * tf.reduce_sum(tf.maximum(1 - self.diff_vals_, 0)) + self.config_.reg_param * self.reg_loss_

        #learning_rate = tf.train.exponential_decay(self.config_.lr, 5000, 5000, 0.1, staircase=True)
        self.opt_ = tf.train.AdamOptimizer(learning_rate = self.config_.lr)
        self.train_op_ = self.opt_.minimize(self.loss_)

        self.loader_ = tf.train.Saver([v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
        self.saver_ = tf.train.Saver([v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)], max_to_keep = None)
        
    def save_ckpt(self, ckpt_dir, iteration):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        if not os.path.exists(os.path.join(ckpt_dir)):
            os.makedirs(os.path.join(ckpt_dir))

        self.saver_.save(self.sess_, os.path.join(ckpt_dir, 'checkpoint'), global_step = iteration+1)
        print('Saved ckpt <%d> to %s' % (iteration+1, ckpt_dir))

    def restore_ckpt(self, ckpt_dir):
        ckpt_status = tf.train.get_checkpoint_state(os.path.join(ckpt_dir))
        if ckpt_status:
            self.loader_.restore(self.sess_, ckpt_status.model_checkpoint_path)
        if ckpt_status:
            print('Load model from %s' % (ckpt_status.model_checkpoint_path))
            return True
        print('Fail to load model from Checkpoint Directory')
        return False
    
def main():
    return

if __name__ == '__main__':
    main()
