import tensorflow as tf
import numpy as np
from easydict import EasyDict as edict
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import json

from equation import Equation
from eq_value import EqValue

from functools import cmp_to_key
from copy import deepcopy
import pdb

def tuple2str(seq_tuple):
    merge_seq = list(zip(*seq_tuple))
    return ' '.join([''.join(tup) for tup in merge_seq])

def encode(string):
    codebook_ = [str(digit) for digit in range(10)] + ['+', '-', '/', '^', '='] + ['x', 'y'] + [' ']
    digits = [codebook_.index(s) for s in string]
    return digits

def sort_var(seq_tuple, eq):
    seq_tuple = deepcopy(seq_tuple)
    variables = [term for term in seq_tuple[2] if term != '=' and term != '0']
    variables.sort(key = cmp_to_key(eq.cmp_), reverse = True)
    i = 0
    history= []
    while i < len(variables):
        pos = seq_tuple[2].index(variables[i])
        if eq.move(seq_tuple, pos, i)[1]:
            history.append(deepcopy(seq_tuple))
        i += 1
    return history

def main():
    test_mode_idx = int(sys.argv[1])
    test_modes = ['general', 'sort']
    test_mode = test_modes[test_mode_idx]
    
    eqv_config = edict({'encoding_dims': 20, 'rnn_dim': 30, 'C': 1, 'lr': 5e-5, 'num_character': 18, 'bacth_size': 100})
    init_w = np.random.uniform(size = [1, eqv_config.rnn_dim])
    sess = tf.Session()
    eqv = EqValue(eqv_config, init_w, sess)
    
    init = tf.global_variables_initializer()
    sess.run(init)
    ckpt_dir = 'CKPT_rnn_dim_%d_lr_5e-5_encoding_dims_%d_2_4' % (eqv_config.rnn_dim, eqv_config.encoding_dims)
    #eqv.restore_ckpt(ckpt_dir)
    #eq = Equation(2, 4, 20, 5)
    
    data = np.load('../Data/equations_encoded_2_4_20_5.npy', allow_pickle=True)
    neg_examples = np.load('../Data/neg_training_set_2_4_20_5.npy', allow_pickle=True)
    data_size = 100000
    batch_size = 100
    train_iter = 15000
    dists0 = []
    accuracy = []
    
    test_size = 10000
    accuracy_test = []
    test_sets = np.take(data, range(test_size))
    lower_tests = []
    higher_tests = []
    s_index = []
    m_index = []
    c_index = []
    o_index = []
    test_index = 0
    
    '''
    generate test set
    '''
    f = open('test_results.txt', 'w')
    lower_tests = np.load('lower_tests.npy', allow_pickle=True)
    higher_tests = np.load('higher_tests.npy', allow_pickle=True)
    test_lower_encoding_idx = np.load('test_lower_encoding_idx.npy', allow_pickle=True)
    test_higher_encoding_idx =  np.load('test_higher_encoding_idx.npy', allow_pickle=True)
    
    lower_tests_idx = np.expand_dims(lower_tests, axis=-1)
    higher_tests_idx = np.expand_dims(higher_tests, axis=-1)

    with open('operation_index_dictionary.json') as js:
        operation_index_dictionary = json.load(js)
    '''                                                                                                                                           
    training
    '''
    training_range = np.arange(data_size)[test_size:]

    for itr in tqdm(range(train_iter)):
        lower_equations = []
        higher_equations = []
        idx = np.random.choice(training_range, batch_size)
        hists = np.take(data, idx)
        idx_neg = idx - test_size
        neg_samples = np.take(neg_examples, idx_neg)
        for hist in hists:
            index = np.random.choice(len(hist)-1, 1)
            index = index[0]
            lower_equations.append(hist[index])
            higher_equations.append(hist[index+1])
        for neg_sample in neg_samples:
            index = np.random.choice(len(neg_sample), 1)
            index = index[0]
            lower_equations.append(neg_sample[index][0])
            higher_equations.append(neg_sample[index][1])
        lower_encoding_idx = []
        for i in range(len(lower_equations)):
            lower_encoding_idx.append([i, len(lower_equations[i])-1])
        higher_encoding_idx = []
        for i in range(len(higher_equations)):
            higher_encoding_idx.append([i, len(higher_equations[i])-1])
        lower_encoding_idx = np.array(lower_encoding_idx)
        higher_encoding_idx = np.array(higher_encoding_idx)

        M0 = max(len(a) for a in lower_equations)
        M1 = max(len(a) for a in higher_equations)
        M = max(M0,M1)
        lower_equations = np.array([a + [eqv_config.num_character] * (M - len(a)) for a in lower_equations])
        higher_equations = np.array([a + [eqv_config.num_character] * (M - len(a)) for a in higher_equations])
        lower_eqs_idx = np.expand_dims(lower_equations, axis=-1)
        higher_eqs_idx = np.expand_dims(higher_equations, axis=-1)

        _, w, loss, lower_vals, higher_vals = eqv.sess_.run([eqv.train_op_, eqv.weight_, eqv.loss_, eqv.lower_vals_, eqv.higher_vals_], \
                                    {eqv.lower_eqs_idx_: lower_eqs_idx, eqv.higher_eqs_idx_: higher_eqs_idx, eqv.initial_states_: np.zeros([lower_eqs_idx.shape[0], eqv.config_.rnn_dim]), \
                                     eqv.lower_encoding_idx_: lower_encoding_idx, eqv.higher_encoding_idx_: higher_encoding_idx})
        dists0.append(loss)
        accuracy_batch = np.count_nonzero(lower_vals < higher_vals)/100
        accuracy.append(accuracy_batch)
        #print(accuracy_batch)

        test_lower_vals_, test_higher_vals_, test_lower_encoding_, test_higher_encoding, weight_ = eqv.sess_.run([eqv.lower_vals_, eqv.higher_vals_, eqv.lower_eq_encodings_2_, \
                                                    eqv.higher_eq_encodings_2_, eqv.weight_], {eqv.lower_eqs_idx_: lower_tests_idx, eqv.higher_eqs_idx_:higher_tests_idx, \
                    eqv.initial_states_: np.zeros([test_size, eqv.config_.rnn_dim]), eqv.lower_encoding_idx_: test_lower_encoding_idx, eqv.higher_encoding_idx_: test_higher_encoding_idx})
     
        accuracy_test.append(np.count_nonzero(test_lower_vals_ < test_higher_vals_)/test_size)
        print(np.count_nonzero(test_lower_vals_ < test_higher_vals_)/test_size)
        index_l = ''
        for j in range(test_size):
            if test_lower_vals_[j] >= test_higher_vals_[j]:
                index_l += str(j)
                index_l += ','

        if test_mode == 'general':
            s_count = 0
            m_count = 0
            c_count = 0
            o_count = 0
            for j in range(test_size):
                if test_lower_vals_[j] >= test_higher_vals_[j]:
                    if j in s_index:
                        s_count = s_count + 1
                    elif j in m_index:
                        m_count = m_count + 1
                    elif j in c_index:
                        c_count = c_count + 1
                    else:
                        o_count = o_count + 1
            print('scale accuracy', (len(operation_index_dictionary['22'])-s_count)/len(operation_index_dictionary['22']))
            print('test_size',len(operation_index_dictionary['22']))
            print('merge accuracy', (len(operation_index_dictionary['23'])-m_count)/len(operation_index_dictionary['23']))
            print('test_size',len(operation_index_dictionary['23']))
            print('remove accuracy', (len(operation_index_dictionary['24'])-c_count)/len(operation_index_dictionary['24']))
            print('test_size',len(operation_index_dictionary['24']))
            print('sort accuracy', (len(operation_index_dictionary['25'])-o_count)/len(operation_index_dictionary['25']))
            print('test_size',len(operation_index_dictionary['25']))
        f.write(index_l)
        f.write('\n')
        if (itr + 1) % 1000 == 0:
            eqv.save_ckpt(ckpt_dir, itr)

    f.close()
    plt.figure()
    plt.plot(accuracy, label="accuracy by batch")
    plt.plot(accuracy_test, label="accuracy on test set")
    plt.xlabel("iteration")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig('accurary_batch_100_constant_learning_rate_5e-5_rnn_%d_2_4.png' % (eqv_config.rnn_dim))
    return

if __name__ == '__main__':
    main()
