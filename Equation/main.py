from easydict import EasyDict as edict
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from eq_value import EqValue
from equation import Equation

def main():
    eqv_config = edict({'encoding_dims': 20, 'rnn_dim': 15, 'C': 1, 'lr': 1e-4, 'num_character': 20})
    eqt_config = edict({'num_var': 2, 'order': 3, 'coef_range_nom': 20, 'coef_range_denom': 10})
    train_iter = 1000
    sess = tf.Session()
    init_w = np.random.uniform(size = [1, eqv_config.rnn_dim])
    eqv = EqValue(eqv_config, init_w, sess)
    eq = Equation(eqt_config.num_var, eqt_config.order, eqt_config.coef_range_nom, eqt_config.coef_range_denom)

    init = tf.global_variables_initializer()
    sess.run(init)
    
    dists0 = []
    for _ in tqdm(range(train_iter)):
        equation = eq.generate()
        history = eq.simplify(equation)
        print(history[0])
        print(history[-1])
        l = len(history) - 1 
        for i in range(l):
            w, loss = eqv.learn([history[i]], history[i+1])
            dists0.append(loss)

    plt.plot(range(len(dists0)),dists0)
    print(dists0[-1])
    plt.show()
if __name__ == '__main__':
    main()
