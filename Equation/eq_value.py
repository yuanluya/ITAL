import tensorflow as tf
import numpy as np
from easydict import EasyDict as edict
from tqdm import tqdm

import pdb

class EqValue:
    def __init__(self, config, init_w, sess):
        self.config_ = config
        self.init_w_ = init_w
        self.sess_ = sess
        self.chars_ = [str(digit) for digit in range(10)] + ['+', '-', '/', '^', '='] + ['x', 'y', 'z', 'w'] + [' '] 
        #self.eqt_num_ = self.config_.eqt_num
        #def string_to_idx(s):
            #elems = np.array(list(s))
            #index = tf.expand_dims(tf.map_fn(lambda x: self.chars_.index(x), elems, dtype=tf.int32), -1)
            #index = np.random.randint(config.num_character + 1, size = [9, 1])
            #return index

        
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

    def learn(self, lower_eqs, higher_eqs):
        lower_eqs_idx = [list(map(lambda x: self.chars_.index(x), s)) for s in lower_eqs]
        higher_eqs_idx = [list(map(lambda x: self.chars_.index(x), s)) for s in higher_eqs]
        '''
        for s in lower_eqs:
            lower_eqs_idx.append(list(map(lambda x: self.chars_.index(x), s)))
        for s in higher_eqs:
            higher_eqs_idx.append(list(map(lambda x: self.chars_.index(x), s)))
        '''
        lower_eqs_idx = np.array(lower_eqs_idx)
        higher_eqs_idx = np.array(higher_eqs_idx)
        lower_eqs_idx = np.expand_dims(lower_eqs_idx, axis=-1)
        higher_eqs_idx = np.expand_dims(higher_eqs_idx, axis=-1)

        _, w, loss = self.sess_.run([self.train_op_, self.weight_, self.loss_], {self.lower_eqs_idx_: lower_eqs_idx, self.higher_eqs_idx_: higher_eqs_idx, self.initial_states_: np.zeros([lower_eqs_idx.shape[0], self.config_.rnn_dim])})
        #[w] = self.sess_.run([self.weight_])
        return w, loss        


def main():
    eqv_config = edict({'encoding_dims': 20, 'rnn_dim': 15, 'C': 1, 'lr': 1e-4, 'num_character': 20})
    init_w = np.random.uniform(size = [1, eqv_config.rnn_dim])
    sess = tf.Session()
    eqv = EqValue(eqv_config, init_w, sess)

    train_iter = 1
    init = tf.global_variables_initializer()
    sess.run(init)

    data = np.load('../Data/equations_encoded.npy', allow_pickle=True)
    batch_size = 1
    data_size = 10
    dists0 = []
    for _ in tqdm(range(train_iter)):
        lower_equations = []
        higher_equations = []
        idx = np.random.choice(data_size, batch_size)
        hists = np.take(data, idx)
        for hist in hists:
            index = np.sort(np.random.choice(len(hist), 2))
            lower_equations.append(hist[index[0]])
            higher_equations.append(hist[index[1]])
        M0 = max(len(a) for a in lower_equations)
        M1 = max(len(a) for a in higher_equations)
        M = max(M0,M1)
        lower_equations = np.array([a + [eqv_config.num_character] * (M - len(a)) for a in lower_equations])
        higher_equations = np.array([a + [eqv_config.num_character] * (M - len(a)) for a in higher_equations])
        lower_eqs_idx = np.expand_dims(lower_equations, axis=-1)
        higher_eqs_idx = np.expand_dims(higher_equations, axis=-1)
        _, w, loss = eqv.sess_.run([eqv.train_op_, eqv.weight_, eqv.loss_], {eqv.lower_eqs_idx_: lower_eqs_idx, \
                                                    eqv.higher_eqs_idx_: higher_eqs_idx, eqv.initial_states_: np.zeros([lower_eqs_idx.shape[0], eqv.config_.rnn_dim])})


    
    return

if __name__ == '__main__':
    main()
