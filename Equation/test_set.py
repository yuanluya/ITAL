import numpy as np
from easydict import EasyDict as edict
import os

import json
from functools import cmp_to_key
from copy import deepcopy


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

def convert(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError

def main():
    eqv_config = edict({'encoding_dims': 20, 'rnn_dim': 30, 'C': 1, 'lr': 5e-5, 'num_character': 18, 'bacth_size': 100})
    data = np.load('../Data/equations_encoded_2_4_20_5.npy', allow_pickle=True)

    test_size = 10000
    test_sets = np.take(data, range(test_size))
    lower_tests = []
    higher_tests = []
    s_index = []
    m_index = []
    c_index = []
    o_index = []
    test_index = 0
    index_pair_dictionary = {}
    operation_index_dictionary = {22:[], 23:[], 24:[], 25:[],}
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
 
    lower_tests = np.array([a + [eqv_config.num_character] * (M - len(a)) for a in lower_tests])
    higher_tests = np.array([a + [eqv_config.num_character] * (M - len(a)) for a in higher_tests])
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
    
if __name__ == '__main__':
    main()
