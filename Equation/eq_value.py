import tensorflow as tf
import numpy as np
from easydict import EasyDict as edict

import pdb

class EqValue:
    def __init__(self, config, init_w):
        self.config_ = config
        self.init_w_ = init_w
        self.lower_eqs_ = tf.placeholder(tf.float32, shape = [None, None, self.config_.encoding_dims])
        self.higher_eqs_ = tf.placeholder(tf.float32, shape = [None, None, self.config_.encoding_dims])
        self.initial_states_ = tf.placeholder(tf.float32, shape = [self.lower_eqs_.shape[0], self.config_.rnn_dim])

        self.gru_1_ = tf.keras.layers.GRU(self.config_.rnn_dim, stateful = False,
                                        return_sequences = True, return_state = False)
        self.bi_encoder_ = tf.keras.layers.Bidirectional(self.gru_1_)

        self.lower_eq_encodings_ = self.bi_encoder_(self.lower_eqs_)
        self.higher_eq_encodings_ = self.bi_encoder_(self.higher_eqs_)
        self.gru_2_ = tf.keras.layers.GRU(self.config_.rnn_dim, stateful = False,
                                        return_sequences = False, return_state = False)
        self.lower_eq_encodings_2_ = self.gru_2_(self.lower_eq_encodings_, self.initial_states_)
        self.higher_eq_encodings_2_ = self.gru_2_(self.higher_eq_encodings_, self.initial_states_)

        self.weight_ = tf.Variable(initial_value = self.init_w_, name = 'weight', dtype = tf.float32)
        self.lower_vals_ = tf.reduce_sum(self.lower_eq_encodings_2_ * self.weight_, 1)
        self.higher_vals_ = tf.reduce_sum(self.higher_eq_encodings_2_* self.weight_, 1)

        self.diff_vals_ = tf.reduce_sum((self.higher_eq_encodings_2_ - self.lower_eq_encodings_2_) * self.weight_, 1)

        self.loss_ = 0.5 * tf.reduce_sum(tf.square(self.weight_)) +\
                     self.config_.C * tf.reduce_sum(tf.maximum(1 - self.diff_vals_, 0))
        self.opt_ = tf.train.AdamOptimizer(learning_rate = self.config_.lr)
        self.train_op_ = self.opt_.minimize(self.loss_)
        


def main():
    config = edict({'encoding_dims': 20, 'rnn_dim': 15, 'C': 1, 'lr': 1e-4})
    init_w = np.random.uniform(size = [1, config.rnn_dim])
    eqv = EqValue(config, init_w)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    _, loss = sess.run([eqv.train_op_, eqv.loss_], {eqv.lower_eqs_: np.random.uniform(size = [32, 9, config.encoding_dims]),
                                                    eqv.higher_eqs_: np.random.uniform(size = [32, 9, config.encoding_dims]),
                                                    eqv.initial_states_: np.zeros([32, config.rnn_dim])})
    return

if __name__ == '__main__':
    main()