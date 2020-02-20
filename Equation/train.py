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

def main():
    eqv_config = edict({'encoding_dims': 20, 'rnn_dim': 30, 'C': 1, 'lr': 5e-5, 'num_character': 18, 'bacth_size': 100})
    init_w = np.random.uniform(size = [1, eqv_config.rnn_dim])
    sess = tf.Session()
    eqv = EqValue(eqv_config, init_w, sess)
    
    init = tf.global_variables_initializer()
    sess.run(init)
    ckpt_dir = 'CKPT_rnn_dim_30_lr_5e-5_encoding_dims_20_2_4_neg'
    eqv.restore_ckpt(ckpt_dir)
    #eq = Equation(2, 4, 20, 5)
    
    data = np.load('../Data/equations_encoded_2_4_20_5.npy', allow_pickle=True)
    neg_examples = np.load('../Data/neg_training_set_2_4_20_5.npy', allow_pickle=True)
    data_size = 100000
    batch_size = 100
    neg_batch_size = 200
    train_iter = 15000
    dists0 = []
    
    test_size = 10000
    accuracy_test = []
    neg_accuracy_test = []
    test_sets = np.take(data, range(test_size))
    lower_tests = []
    higher_tests = []
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
    s_len = len(operation_index_dictionary['22'])
    m_len = len(operation_index_dictionary['23'])
    c_len = len(operation_index_dictionary['24'])
    o_len = len(operation_index_dictionary['25'])
    
    neg_lower_tests = np.load('neg_lower_tests.npy', allow_pickle=True)
    neg_higher_tests = np.load('neg_higher_tests.npy', allow_pickle=True)
    neg_test_lower_encoding_idx = np.load('neg_test_lower_encoding_idx.npy', allow_pickle=True)
    neg_test_higher_encoding_idx =  np.load('neg_test_higher_encoding_idx.npy', allow_pickle=True)

    neg_lower_tests_idx = np.expand_dims(neg_lower_tests, axis=-1)
    neg_higher_tests_idx = np.expand_dims(neg_higher_tests, axis=-1)

    with open('neg_operation_index_dictionary.json') as js:
        negative_operation_index_dictionary = json.load(js)
    negative_s_len = len(negative_operation_index_dictionary['22'])
    negative_m_len = len(negative_operation_index_dictionary['23'])
    negative_c_len = len(negative_operation_index_dictionary['24'])
    negative_o_len = len(negative_operation_index_dictionary['25'])

    '''                                                                                                                                           
    training
    '''
    training_range = np.arange(data_size)[test_size:]

    for itr in tqdm(range(train_iter)):
        accuracy = []
        lower_equations = []
        higher_equations = []
        idx = np.random.choice(training_range, batch_size)
        hists = np.take(data, idx)
        idx_neg =  np.random.choice(training_range, neg_batch_size)
        neg_samples = np.take(neg_examples, idx_neg)
        for hist in hists:
            index = np.random.choice(len(hist)-1, 1)
            index = index[0]
            lower_equations.append(hist[index][:-1])
            higher_equations.append(hist[index+1][:-1])
        for neg_sample in neg_samples:
            index = np.random.choice(len(neg_sample), 1)
            index = index[0]
            lower_equations.append(neg_sample[index][0][:-1])
            higher_equations.append(neg_sample[index][1][:-1])

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
        accuracy_batch = np.count_nonzero(lower_vals < higher_vals)/(batch_size + neg_batch_size)
        accuracy.append(accuracy_batch)
        #print(accuracy_batch)


        '''
        test 
        '''
        test_lower_vals_, test_higher_vals_ = eqv.sess_.run([eqv.lower_vals_, eqv.higher_vals_], {eqv.lower_eqs_idx_: lower_tests_idx, eqv.higher_eqs_idx_:higher_tests_idx, \
                    eqv.initial_states_: np.zeros([test_size, eqv.config_.rnn_dim]), eqv.lower_encoding_idx_: test_lower_encoding_idx, eqv.higher_encoding_idx_: test_higher_encoding_idx})
        test_accuracy = np.count_nonzero(test_lower_vals_ < test_higher_vals_)/test_size
        accuracy_test.append(test_accuracy)
        print('test accuracy', test_accuracy)

        index_l = ''
        for j in range(test_size):
            if test_lower_vals_[j] >= test_higher_vals_[j]:
                index_l += str(j)
                index_l += ','
        
        s_count = 0
        m_count = 0
        c_count = 0
        o_count = 0
        for j in range(test_size):
            if test_lower_vals_[j] >= test_higher_vals_[j]:
                s_count = s_count + (j in operation_index_dictionary['22'])
                m_count = m_count + (j in operation_index_dictionary['23'])
                c_count = c_count + (j in operation_index_dictionary['24'])
                o_count = o_count + (j in operation_index_dictionary['25'])
        print('scale accuracy', (s_len - s_count)/s_len)
        print('test_size', s_len)
        print('merge accuracy', (m_len - m_count)/m_len)
        print('test_size', m_len)
        print('remove accuracy', (c_len - c_count)/c_len)
        print('test_size', c_len)
        print('sort accuracy', (o_len - o_count)/o_len)
        print('test_size', o_len)

        neg_test_lower_vals_, neg_test_higher_vals_ = eqv.sess_.run([eqv.lower_vals_, eqv.higher_vals_], {eqv.lower_eqs_idx_: neg_lower_tests_idx, eqv.higher_eqs_idx_:neg_higher_tests_idx, \
                    eqv.initial_states_: np.zeros([test_size, eqv.config_.rnn_dim]), eqv.lower_encoding_idx_: neg_test_lower_encoding_idx, eqv.higher_encoding_idx_: neg_test_higher_encoding_idx})
        neg_test_accuracy = np.count_nonzero(neg_test_lower_vals_ < neg_test_higher_vals_)/test_size
        neg_accuracy_test.append(neg_test_accuracy)
        print('neg test accuracy', neg_test_accuracy)

        s_count = 0
        m_count = 0
        c_count = 0
        o_count = 0
        for j in range(test_size):
            if neg_test_lower_vals_[j] >= neg_test_higher_vals_[j]:
                s_count = s_count + (j in negative_operation_index_dictionary['22'])
                m_count = m_count + (j in negative_operation_index_dictionary['23'])
                c_count = c_count + (j in negative_operation_index_dictionary['24'])
                o_count = o_count + (j in negative_operation_index_dictionary['25'])
        print('scale accuracy', (negative_s_len - s_count)/negative_s_len)
        print('test_size', negative_s_len)
        print('merge accuracy', (negative_m_len - m_count)/negative_m_len)
        print('test_size', negative_m_len)
        print('remove accuracy', (negative_c_len - c_count)/negative_c_len)
        print('test_size', negative_c_len)
        print('sort accuracy', (negative_o_len - o_count)/negative_o_len)
        print('test_size', negative_o_len)


        f.write(index_l)
        f.write('\n')
        if (itr + 1) % 1000 == 0:
            eqv.save_ckpt(ckpt_dir, itr)

    f.close()
    plt.figure()
    plt.plot(accuracy, label="accuracy by batch")
    plt.plot(accuracy_test, label="accuracy on test set")
    plt.plot(neg_accuracy_test, label="accuracy on negative test set")
    plt.xlabel("iteration")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig('accurary_learning_rate_5e-5_rnn_30_2_4.png')
    return

if __name__ == '__main__':
    main()
