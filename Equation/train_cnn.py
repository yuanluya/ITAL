import tensorflow.compat.v1 as tf
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
from eq_value_cnn import EqValue

from functools import cmp_to_key
from copy import deepcopy
import pdb

def main():
    np.random.seed(1234)
    batch_size = 128
    lr = 1e-4
    codebook_ = np.array([str(digit) for digit in range(10)] + ['+', '-', '/', '^', '='] + ['x', 'y'] + [' '] + [''])
    op_type = ["scale", "merge", "remove denominators", "sort"]
    
    train_data = np.load('equations_encoded_3_4_20_5.npy', allow_pickle = True)[:490000]
    test_data = np.load('equations_encoded_3_4_20_5.npy', allow_pickle = True)[490000:]

    neg_train_data = np.load('neg_training_set_3_4_20_5.npy', allow_pickle = True)[:490000]
    neg_test_data = np.load('neg_training_set_3_4_20_5.npy', allow_pickle = True)[490000:]
    print("Train data size pos: %d neg: %d" % (len(train_data), len(neg_train_data)))
    print("Test data size pos: %d neg: %d" % (len(test_data), len(neg_test_data)))  

    lower_equations_all = []
    higher_equations_all = []
    train_pos_idx = []
    train_neg_idx = []
    for d in tqdm(train_data):
        for i in range(len(d)):
            if i != len(d) - 1:
                lower_equations_all.append(d[i][:-1])
            if i != 0:
                higher_equations_all.append(d[i][:-1])
                train_pos_idx.append(d[i][-1])
    for d in tqdm(neg_train_data):
        for i in range(len(d)):
            lower_equations_all.append(d[i][0][:-1])
            higher_equations_all.append(d[i][1][:-1])
            train_neg_idx.append(d[i][0][-1])
    lower_equations_all = np.array(lower_equations_all)
    higher_equations_all = np.array(higher_equations_all)
    train_pos_idx = np.array(train_pos_idx)
    train_neg_idx = np.array(train_neg_idx)
    print("Train data composition pos: %s neg: %s" % ([np.sum(train_pos_idx == i) for i in range(22, 26)], [np.sum(train_neg_idx == i) for i in range(22, 26)]))

    train_shuffle_idx = np.arange(len(lower_equations_all))
    np.random.shuffle(train_shuffle_idx)
    lower_equations_all = lower_equations_all[train_shuffle_idx]
    higher_equations_all = higher_equations_all[train_shuffle_idx]

    lower_equations_test = []
    higher_equations_test = []
    test_pos_idx = []
    test_neg_idx = []
    for d in tqdm(test_data):
        for i in range(len(d)):
            if i != len(d) - 1:
                lower_equations_test.append(d[i][:-1])
            if i != 0:
                higher_equations_test.append(d[i][:-1])
                test_pos_idx.append(d[i][-1])
    pos_test = len(lower_equations_test)
    for d in tqdm(neg_test_data):
        for i in range(len(d)):
            lower_equations_test.append(d[i][0][:-1])
            higher_equations_test.append(d[i][1][:-1])
            test_neg_idx.append(d[i][0][-1])
    test_pos_idx = np.array(test_pos_idx)
    test_neg_idx = np.array(test_neg_idx)
    print("Test data composition pos: %s neg: %s" % ([np.sum(test_pos_idx == i) for i in range(22, 26)], [np.sum(test_neg_idx == i) for i in range(22, 26)]))

    M0 = np.max([len(a) for a in lower_equations_all])
    M1 = np.max([len(a) for a in higher_equations_all])
    M = max(M0, M1)
    print("Max length: %d" % M)

    eqv_config = edict({'input_dim': M, 'encoding_dim': 30, 'output_dim': 40, 'C': 1, 'reg_param': 1e-5, 'batch_size': batch_size,
        'lr': lr, 'num_character': 19, 'layer_info': [(64, 5, 1, False), (64, 5, 1, True), (32, 3, 1, False), (32, 3, 1, True), (32, 3, 1, False), (32, 3, 1, True)]})
    
    init_w = np.random.uniform(size = [1, eqv_config.output_dim + 1])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    eqv = EqValue(eqv_config, init_w, sess)
    
    init = tf.global_variables_initializer()
    sess.run(init)
    ckpt_dir = 'CKPT_cnn_dim_%d_encoding_dim_%d_3_4_6layers' % (eqv_config.output_dim, eqv_config.encoding_dim)
    eqv.restore_ckpt(ckpt_dir)

    ###################### train ############################
    accuracy = []
    loss = []
    for itr in tqdm(range(int(len(lower_equations_all) / batch_size))):
        lower_equations = lower_equations_all[itr * batch_size: (itr + 1) * batch_size]
        higher_equations = higher_equations_all[itr * batch_size: (itr + 1) * batch_size]

        lower_equations = np.array([a + [eqv_config.num_character] * (M - len(a)) for a in lower_equations])
        higher_equations = np.array([a + [eqv_config.num_character] * (M - len(a)) for a in higher_equations])
        lower_eqs_idx = np.expand_dims(lower_equations, axis=-1)
        higher_eqs_idx = np.expand_dims(higher_equations, axis=-1)
        lower_encoding_idx = np.expand_dims(list(range(eqv_config.batch_size)), axis = 1)
        higher_encoding_idx = np.expand_dims(list(range(eqv_config.batch_size, 2 * eqv_config.batch_size, 1)), axis = 1)

        _, w, l, lower_vals, higher_vals, cb, lenc, henc = eqv.sess_.run([eqv.train_op_, eqv.weight_, eqv.loss_,
            eqv.lower_vals_, eqv.higher_vals_, eqv.codebook_0_, eqv.lower_encodings_, eqv.higher_encodings_], 
            {eqv.lower_eqs_idx_: lower_eqs_idx, eqv.higher_eqs_idx_: higher_eqs_idx, eqv.lower_encoding_idx_: lower_encoding_idx, eqv.higher_encoding_idx_: higher_encoding_idx})
        
        accuracy.append(np.mean(lower_vals < higher_vals))
        loss.append(l)
        
        if (itr + 1) % 1000 == 0:
            eqv.save_ckpt(ckpt_dir, itr)
            print("Iteration %d loss: %f accuracy: %f\n" % (itr, np.mean(loss), np.mean(accuracy)))
            loss = []
            accuracy = []

    ###################### test ############################
    accuracy = []
    loss = []
    accuracy_by_type = []
    train_features = np.ndarray((0, eqv_config.output_dim))
    train_labels = np.ndarray((0))
    for itr in tqdm(range(int(len(lower_equations_test) / batch_size))):
        lower_equations = lower_equations_test[itr * batch_size: (itr + 1) * batch_size]
        higher_equations = higher_equations_test[itr * batch_size: (itr + 1) * batch_size]

        lower_equations = np.array([a + [eqv_config.num_character] * (M - len(a)) for a in lower_equations])
        higher_equations = np.array([a + [eqv_config.num_character] * (M - len(a)) for a in higher_equations])
        lower_eqs_idx = np.expand_dims(lower_equations, axis=-1)
        higher_eqs_idx = np.expand_dims(higher_equations, axis=-1)
        lower_encoding_idx = np.expand_dims(list(range(eqv_config.batch_size)), axis = 1)
        higher_encoding_idx = np.expand_dims(list(range(eqv_config.batch_size, 2 * eqv_config.batch_size, 1)), axis = 1)

        w, l, lower_vals, higher_vals, lenc, henc = eqv.sess_.run([eqv.weight_, eqv.loss_, eqv.lower_vals_, eqv.higher_vals_, eqv.lower_encodings_0_, eqv.higher_encodings_0_], \
                                    {eqv.lower_eqs_idx_: lower_eqs_idx, eqv.higher_eqs_idx_: higher_eqs_idx,
                                    eqv.lower_encoding_idx_: lower_encoding_idx, eqv.higher_encoding_idx_: higher_encoding_idx})
        accuracy.append(np.mean(lower_vals < higher_vals))
        accuracy_by_type.append(lower_vals < higher_vals)
        loss.append(l)
        train_features = np.concatenate((train_features, lenc))
        train_labels = np.concatenate((train_labels, lower_vals))
        train_features = np.concatenate((train_features, henc))
        train_labels = np.concatenate((train_labels, higher_vals))
        
        # for a in range(batch_size):
        #     if (lower_vals < higher_vals)[a] == 0:
        #         str1 = codebook_[lower_equations[a]]
        #         str2 = codebook_[higher_equations[a]]
        #         print("\nOperation type: %s" % op_type[np.concatenate((test_pos_idx, test_neg_idx))[itr * batch_size + a] - 22])
        #         print(''.join(str1))
        #         print(''.join(str2))

    np.save("equation_train_features_cnn_3var_40_6layers.npy", train_features[1:])
    np.save("equation_train_labels_cnn_3var_40_6layers.npy", train_labels[1:])
    np.save("equation_gt_weights_cnn_3var_40_6layers.npy", w)

    print("Test accuracy: %f loss: %f" % (np.mean(accuracy), np.mean(loss)))
    print("Weight: %s" % w)
    accuracy_by_type_pos = np.array(accuracy_by_type).flatten()[:len(test_pos_idx)]
    accuracy_by_type_neg = np.array(accuracy_by_type).flatten()[len(test_pos_idx):]
    test_neg_idx = np.array(test_neg_idx[:len(np.array(accuracy_by_type).flatten()) - len(test_pos_idx)])
    
    for i in range(22, 26):
        print("Operation type: %s" % op_type[i - 22])
        print("\tpos accuracy: %f size: %d" % (np.mean(accuracy_by_type_pos[test_pos_idx == i]), np.sum(test_pos_idx == i)))
        print("\tneg accuracy: %f size: %d" % (np.mean(accuracy_by_type_neg[test_neg_idx == i]), np.sum(test_neg_idx == i)))


if __name__ == '__main__':
    main()
