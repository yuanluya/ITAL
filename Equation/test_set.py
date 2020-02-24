import numpy as np
from easydict import EasyDict as edict
import os

import json
from functools import cmp_to_key
from copy import deepcopy
from config import config

def convert(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError

def main():
    test_sets = np.load(config.dir_path + 'equations_encoded_2_4_20_5.npy', allow_pickle=True)[config.train_size:config.train_size+config.test_size]
    neg_test_sets = np.load(config.dir_path + 'neg_training_set_2_4_20_5.npy', allow_pickle=True)[config.train_size:config.train_size+config.test_size]
    
    lower_tests = []
    higher_tests = []
    test_index = 0
    index_pair_dictionary = {}
    operation_index_dictionary = {22:[], 23:[], 24:[], 25:[]}

    for hist in test_sets:
        index = np.random.choice(len(hist)-1, 1)
        index = index[0]
        index_pair_dictionary[test_index] = [index, index + 1]
        lower_tests.append(hist[index][:-1])
        higher_tests.append(hist[index+1][:-1])

        operation = hist[index+1][-1]
        operation_index_dictionary[operation].append(test_index)
        test_index = test_index + 1

    test_lower_encoding_idx = []
    for i in range(len(lower_tests)):
        test_lower_encoding_idx.append([i, len(lower_tests[i])-1])

    test_higher_encoding_idx = []
    for i in range(len(higher_tests)):
        test_higher_encoding_idx.append([i, len(higher_tests[i])-1])

    test_lower_encoding_idx = np.array(test_lower_encoding_idx)
    test_higher_encoding_idx = np.array(test_higher_encoding_idx)
    
    M00 = max(len(a) for a in lower_tests)
    M11 = max(len(a) for a in higher_tests)
    M = max(M00,M11)
 
    lower_tests = np.array([a + [config.num_character] * (M - len(a)) for a in lower_tests])
    higher_tests = np.array([a + [config.num_character] * (M - len(a)) for a in higher_tests])
    lower_tests_idx = np.expand_dims(lower_tests, axis=-1)
    higher_tests_idx = np.expand_dims(higher_tests, axis=-1)

    with open('index_pair_dictionary.json', 'w') as f:
        json.dump(index_pair_dictionary, f, default=convert)
    with open('operation_index_dictionary.json', 'w') as f:
        json.dump(operation_index_dictionary, f, default=convert)
        
    np.save('lower_tests.npy', lower_tests)
    np.save('higher_tests.npy', higher_tests)
    np.save('test_lower_encoding_idx.npy', test_lower_encoding_idx)
    np.save('test_higher_encoding_idx.npy', test_higher_encoding_idx)

    neg_lower_tests = []
    neg_higher_tests = []
    test_index = 0
    neg_operation_index_dictionary = {22:[], 23:[], 24:[], 25:[]}
    for neg_sample in neg_test_sets:
        index = np.random.choice(len(neg_sample), 1)
        index = index[0]
        neg_lower_tests.append(neg_sample[index][0][:-1])
        neg_higher_tests.append(neg_sample[index][1][:-1])

        operation = neg_sample[index][0][-1]
        print(operation)
        neg_operation_index_dictionary[operation].append(test_index)
        test_index = test_index + 1


    neg_test_lower_encoding_idx = []
    neg_test_higher_encoding_idx = []
    for i in range(len(neg_lower_tests)):
        neg_test_lower_encoding_idx.append([i, len(neg_lower_tests[i])-1])
    for i in range(len(neg_higher_tests)):
        neg_test_higher_encoding_idx.append([i, len(neg_higher_tests[i])-1])
    neg_test_lower_encoding_idx = np.array(neg_test_lower_encoding_idx)
    neg_test_higher_encoding_idx = np.array(neg_test_higher_encoding_idx)

    M0 = max(len(a) for a in neg_lower_tests)
    M1 = max(len(a) for a in neg_higher_tests)
    M = max(M0,M1)
    neg_lower_tests = np.array([a + [config.num_character] * (M - len(a)) for a in neg_lower_tests])
    neg_higher_tests = np.array([a + [config.num_character] * (M - len(a)) for a in neg_higher_tests])
    neg_lower_tests_idx = np.expand_dims(neg_lower_tests, axis=-1)
    neg_higher_tests_idx = np.expand_dims(neg_higher_tests, axis=-1)

    np.save('neg_lower_tests.npy', neg_lower_tests)
    np.save('neg_higher_tests.npy', neg_higher_tests)
    np.save('neg_test_lower_encoding_idx.npy', neg_test_lower_encoding_idx)
    np.save('neg_test_higher_encoding_idx.npy', neg_test_higher_encoding_idx)

    with open('neg_operation_index_dictionary.json', 'w') as f:
        json.dump(neg_operation_index_dictionary, f, default=convert)
    for i in neg_operation_index_dictionary:
        print(i,len(neg_operation_index_dictionary[i]))
if __name__ == '__main__':
    main()
