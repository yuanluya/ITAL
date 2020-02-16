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

        
        self.lower_eqs_idx_ = tf.placeholder(tf.int32, shape = [None, None, 1])
        self.higher_eqs_idx_ = tf.placeholder(tf.int32, shape = [None, None, 1])
        #self.lower_eqs_idx_ = tf.map_fn(lambda x: string_to_idx(x), self.lower_eqs_str_, dtype=tf.int32)
        #self.higher_eqs_idx_ = tf.map_fn(lambda x: string_to_idx(x), self.higher_eqs_str_, dtype=tf.int32)

        self.initial_states_ = tf.placeholder(tf.float32, shape = [self.lower_eqs_idx_.shape[0], self.config_.rnn_dim])

        self.codebook_ = tf.get_variable('codebook', shape = [self.config_.num_character, self.config_.encoding_dims],
                                         dtype = tf.float32, initializer = tf.random_normal_initializer())
        self.codebook_0_ = tf.concat([self.codebook_, tf.zeros(shape = [1, self.config_.encoding_dims])], 0)
        self.lower_eqs_ = tf.squeeze(tf.nn.embedding_lookup(self.codebook_0_, self.lower_eqs_idx_), 2)
        self.higher_eqs_ = tf.squeeze(tf.nn.embedding_lookup(self.codebook_0_, self.higher_eqs_idx_), 2)

        self.lower_encoding_idx_ = tf.placeholder(tf.int32, shape = [None, 2])
        self.higher_encoding_idx_ = tf.placeholder(tf.int32, shape = [None, 2])
 
        self.gru_1_ = tf.keras.layers.GRU(self.config_.rnn_dim, stateful = False,
                                          return_sequences = True, return_state = False, use_bias = False)
        self.bi_encoder_ = tf.keras.layers.Bidirectional(self.gru_1_)

        self.lower_eq_encodings_ = self.bi_encoder_(self.lower_eqs_)
        self.higher_eq_encodings_ = self.bi_encoder_(self.higher_eqs_)
        self.gru_2_ = tf.keras.layers.GRU(self.config_.rnn_dim, stateful = False,
                                        return_sequences = True, return_state = False)
        self.lower_eq_encodings_2_ = self.gru_2_(self.lower_eq_encodings_, self.initial_states_)
        self.higher_eq_encodings_2_ = self.gru_2_(self.higher_eq_encodings_, self.initial_states_)
        self.lower_eq_encodings_2_ = tf.gather_nd(self.lower_eq_encodings_2_, self.lower_encoding_idx_)
        self.higher_eq_encodings_2_ =  tf.gather_nd(self.higher_eq_encodings_2_, self.higher_encoding_idx_)
         
        self.weight_ = tf.Variable(initial_value = self.init_w_, name = 'weight', dtype = tf.float32)
        self.lower_vals_ = tf.reduce_sum(self.lower_eq_encodings_2_ * self.weight_, 1)
        self.higher_vals_ = tf.reduce_sum(self.higher_eq_encodings_2_* self.weight_, 1)

        self.diff_vals_ = tf.reduce_sum((self.higher_eq_encodings_2_ - self.lower_eq_encodings_2_) * self.weight_, 1)

        self.loss_ = 0.5 * tf.reduce_sum(tf.square(self.weight_)) +\
            self.config_.C * tf.reduce_sum(tf.maximum(1 - self.diff_vals_, 0))
        #learning_rate = tf.train.exponential_decay(self.config_.lr, 5000, 5000, 0.1, staircase=True)
        self.opt_ = tf.train.AdamOptimizer(learning_rate = self.config_.lr)
        self.train_op_ = self.opt_.minimize(self.loss_)

        self.loader_ = tf.train.Saver([v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
        self.saver_ = tf.train.Saver([v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)],max_to_keep=None)
        
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
