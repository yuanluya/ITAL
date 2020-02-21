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

from eq_value import EqValue
from config import config

from copy import deepcopy
import pdb

def main():
    test = int(sys.argv[1])

    eqv_config = edict({'encoding_dims': config.encoding_dims, 'rnn_dim': config.rnn_dim, 'C': 1, 'lr': config.lr, 'num_character': config.num_character, 'bacth_size': 100})
    init_w = np.random.uniform(size = [1, eqv_config.rnn_dim])
    sess = tf.Session()
    eqv = EqValue(eqv_config, init_w, sess)
    
    init = tf.global_variables_initializer()
    sess.run(init)
    if test:
        eqv.restore_ckpt(config.ckpt_dir)
    
    '''
    training
    '''
    if not test:
        data = np.load(config.dir_path + 'equations_encoded_2_4_20_5.npy', allow_pickle=True)[:50000]
        neg_examples = np.load(config.dir_path + 'neg_training_set_2_4_20_5.npy', allow_pickle=True)[:50000]

        accuracy = []
        batch_size = 100
        neg_batch_size = 200
        dists0 = []

        lower_equations_all = []
        higher_equations_all = []
        train_pos_idx = []
        train_neg_idx = []
        for d in data:
            for i in range(len(d)):
                if i != len(d) - 1:
                    lower_equations_all.append(d[i][:-1])
                if i != 0:
                    higher_equations_all.append(d[i][:-1])
                    train_pos_idx.append(d[i][-1])
        for d in neg_examples:
            for i in range(len(d)):
                lower_equations_all.append(d[i][0][:-1])
                higher_equations_all.append(d[i][1][:-1])
                train_neg_idx.append(d[i][0][-1])

        lower_encoding_idx_all = []
        for i in range(len(lower_equations_all)):
            lower_encoding_idx_all.append([i, len(lower_equations_all[i])-1])
        higher_encoding_idx_all = []
        for i in range(len(higher_equations_all)):
            higher_encoding_idx_all.append([i, len(higher_equations_all[i])-1])
        lower_encoding_idx_all = np.array(lower_encoding_idx_all)
        higher_encoding_idx_all = np.array(higher_encoding_idx_all)

        lower_equations_all = np.array(lower_equations_all)
        higher_equations_all = np.array(higher_equations_all)
        train_pos_idx = np.array(train_pos_idx)
        train_neg_idx = np.array(train_neg_idx)
        print("Train data composition pos: %s neg: %s" % ([np.sum(train_pos_idx == i) for i in range(22, 26)], [np.sum(train_neg_idx == i) for i in range(22, 26)]))

        train_shuffle_idx = np.arange(len(lower_equations_all))
        np.random.shuffle(train_shuffle_idx)
        lower_equations_all = lower_equations_all[train_shuffle_idx]
        higher_equations_all = higher_equations_all[train_shuffle_idx]
        lower_encoding_idx_all = lower_encoding_idx_all[train_shuffle_idx]
        higher_encoding_idx_all = higher_encoding_idx_all[train_shuffle_idx]
        
        M0 = np.max([len(a) for a in lower_equations_all])
        M1 = np.max([len(a) for a in higher_equations_all])
        M = max(M0, M1)
        
        lower_equations_all = np.array([a + [eqv_config.num_character] * (M - len(a)) for a in lower_equations_all])
        higher_equations_all = np.array([a + [eqv_config.num_character] * (M - len(a)) for a in higher_equations_all])
        lower_eqs_idx_all = np.expand_dims(lower_equations_all, axis=-1)
        higher_eqs_idx_all = np.expand_dims(higher_equations_all, axis=-1)

        for itr in tqdm(range(int(len(lower_equations_all) / batch_size))):
            lower_eqs_idx = lower_eqs_idx_all[itr * batch_size: (itr + 1) * batch_size]
            higher_eqs_idx = higher_eqs_idx_all[itr * batch_size: (itr + 1) * batch_size]
            lower_encoding_idx = lower_encoding_idx_all[itr * batch_size: (itr + 1) * batch_size]
            higher_encoding_idx = higher_encoding_idx_all[itr * batch_size: (itr + 1) * batch_size]
            
            _, w, loss, lower_vals, higher_vals = eqv.sess_.run([eqv.train_op_, eqv.weight_, eqv.loss_, eqv.lower_vals_, eqv.higher_vals_], \
                                    {eqv.lower_eqs_idx_: lower_eqs_idx, eqv.higher_eqs_idx_: higher_eqs_idx, eqv.initial_states_: np.zeros([lower_eqs_idx.shape[0], eqv.config_.rnn_dim]), \
                                     eqv.lower_encoding_idx_: lower_encoding_idx, eqv.higher_encoding_idx_: higher_encoding_idx})
            dists0.append(loss)
            accuracy_batch = np.count_nonzero(lower_vals < higher_vals)/(batch_size + neg_batch_size)
            accuracy.append(accuracy_batch)

            if (itr + 1) % int(len(lower_equations_all) / batch_size) == 0:
                eqv.save_ckpt(ckpt_dir, itr)

        plt.figure()
        plt.plot(accuracy, label="accuracy by batch")
        #plt.plot(accuracy_test, label="accuracy on test set")                                                                                                                                      
        #plt.plot(neg_accuracy_test, label="accuracy on negative test set")                                                                                                                         
        plt.xlabel("iteration")
        plt.ylabel("accuracy")
        plt.legend()
        plt.savefig('accurary_learning_rate_5e-5_rnn_30_2_4.png')

    '''
    test 
    '''
    if test:
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
        
        test_lower_vals_, test_higher_vals_ = eqv.sess_.run([eqv.lower_vals_, eqv.higher_vals_], {eqv.lower_eqs_idx_: lower_tests_idx, eqv.higher_eqs_idx_:higher_tests_idx, \
                    eqv.initial_states_: np.zeros([config.test_size, eqv.config_.rnn_dim]), eqv.lower_encoding_idx_: test_lower_encoding_idx, eqv.higher_encoding_idx_: test_higher_encoding_idx})
        test_accuracy = np.count_nonzero(test_lower_vals_ < test_higher_vals_)/config.test_size
        print('test accuracy', test_accuracy)

        s_count = 0
        m_count = 0
        c_count = 0
        o_count = 0
        for j in range(config.test_size):
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
                    eqv.initial_states_: np.zeros([config.test_size, eqv.config_.rnn_dim]), eqv.lower_encoding_idx_: neg_test_lower_encoding_idx, eqv.higher_encoding_idx_: neg_test_higher_encoding_idx})
        neg_test_accuracy = np.count_nonzero(neg_test_lower_vals_ < neg_test_higher_vals_)/config.test_size
        print('neg test accuracy', neg_test_accuracy)

        s_count = 0
        m_count = 0
        c_count = 0
        o_count = 0
        for j in range(config.test_size):
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

    return

if __name__ == '__main__':
    main()
