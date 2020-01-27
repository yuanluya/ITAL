from easydict import EasyDict as edict
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from eq_value import EqValue
from equation import Equation

def main():
    eqv_config = edict({'encoding_dims': 20, 'rnn_dim': 15, 'C': 1, 'lr': 1e-4, 'num_character': 20})
    eqt_config = edict({'num_var': 1, 'order': 3, 'coef_range_nom': 20, 'coef_range_denom': 10})

    sess = tf.Session()
    init_w = np.random.uniform(size = [1, eqv_config.rnn_dim])
    eqv = EqValue(eqv_config, init_w, sess)
    eq = Equation(eqt_config.num_var, eqt_config.order, eqt_config.coef_range_nom, eqt_config.coef_range_denom)

    init = tf.global_variables_initializer()
    sess.run(init)
    
    dists0 = []
    for _ in tqdm(range(100)):
        equation = eq.generate()
        history = eq.simplify(equation)
        print(history[0])
        print(history[1])
        w, loss = eqv.learn([history[0]], [history[-1]])
        dists0.append(loss)
    plt.plot(range(100),dists0)
    plt.show()
if __name__ == '__main__':
    main()
