from easydict import EasyDict as edict
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from eq_value import EqValue
from equation import Equation

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


if __name__ == '__main__':
    main()
