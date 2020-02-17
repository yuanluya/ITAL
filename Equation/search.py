from functools import cmp_to_key
import numpy as np
from equation import Equation
from eq_value import EqValue
import tensorflow as tf

from easydict import EasyDict as edict
from copy import deepcopy
import pdb

def scale(seq_tuple, pos, scale):
    seq_tuple = deepcopy(seq_tuple)
    if seq_tuple[2][pos] == '=':
        return seq_tuple, False
    if seq_tuple[1][pos].find('/') == -1:
        return seq_tuple, False
    dash_pos = seq_tuple[1][pos].find('/')
    nominator = int(seq_tuple[1][pos][0: dash_pos])
    denominator = int(seq_tuple[1][pos][dash_pos + 1:])
    seq_tuple[1][pos] = '%d/%d' % (nominator * scale, denominator * scale)
    return seq_tuple, True

def reduction(seq_tuple, pos):
    seq_tuple = deepcopy(seq_tuple)
    if seq_tuple[0][pos] == '':
        return seq_tuple, False
    dpos = seq_tuple[1][pos].find('/')
    if dpos == -1:
        return seq_tuple, False
    nominator = int(seq_tuple[1][pos][0: dpos])
    denominator = int(seq_tuple[1][pos][dpos + 1: ])
    g = np.gcd(nominator, denominator)
    if g == 1:
        return seq_tuple, False
    if g == denominator:
        seq_tuple[1][pos] = '%d' % (nominator/g)
        return seq_tuple, True
    else:
        seq_tuple[1][pos] = '%d/%d' % (nominator/g, denominator/g)
        return seq_tuple, True

def check0(seq_tuple):
    seq_tuple = deepcopy(seq_tuple)
    if seq_tuple[2].index('=') == 0:
        seq_tuple[0].insert(0, '')
        seq_tuple[1].insert(0, '')
        seq_tuple[2].insert(0, '0')
        return seq_tuple, True
    if seq_tuple[2].index('=') == len(seq_tuple[2]) - 1:
        seq_tuple[0].append('')
        seq_tuple[1].append('')
        seq_tuple[2].append('0')
        return seq_tuple, True
    return seq_tuple, False

def move(seq_tuple, pos1, pos2):
    seq_tuple = deepcopy(seq_tuple)
    if seq_tuple[2][pos1] == '=' or pos1 == pos2:
        return seq_tuple, False
    if seq_tuple[2][pos1] == '0':
        return seq_tuple, False
    eqs_pos1 = seq_tuple[2].index('=')
    seq_tuple[0].insert(pos2, seq_tuple[0].pop(pos1))
    seq_tuple[1].insert(pos2, seq_tuple[1].pop(pos1))
    seq_tuple[2].insert(pos2, seq_tuple[2].pop(pos1))
    eqs_pos2 = seq_tuple[2].index('=')
    
    if(pos1 - eqs_pos1) * (pos2 - eqs_pos2) < 0:
        seq_tuple[0][pos2] = '-' if seq_tuple[0][pos2] == '+' else '+'
        
    return check0(seq_tuple)[0], True

def merge(seq_tuple, pos1, pos2):
    seq_tuple = deepcopy(seq_tuple)
    if seq_tuple[2][pos1] != seq_tuple[2][pos2] or pos1 == pos2:
        return seq_tuple, False
    #assert(seq_tuple[2][pos1] != '=')
    #assert(seq_tuple[2][pos2] != '=')
    if seq_tuple[2][pos1] == '=' or seq_tuple[2][pos2] == '=':
        return seq_tuple, false
    dp1 = seq_tuple[1][pos1].find('/')
    dp2 = seq_tuple[1][pos2].find('/')
    denominator = 1
    if dp1 != -1 and dp2 != -1:
        denominator1 = int(seq_tuple[1][pos1][dp1 + 1:])
        try:
            denominator2 = int(seq_tuple[1][pos2][dp2 + 1:])
        except ValueError:
            print('the error equation is', seq_tuple)
            print(pos1, pos2)
            exit()
        if denominator1 != denominator2:
            return seq_tuple, False
        denominator = denominator1
        denominator = np.lcm(denominator1, denominator2)
    elif dp1 == -1 and dp2 == -1:
        denominator = 1
    elif dp1 == -1:
        try:
            denominator = int(seq_tuple[1][pos2][dp2 + 1:])
        except ValueError:
            print('the error equation is', seq_tuple)
            print(pos1, pos2)
            exit()
    elif dp2 == -1:
        try:
            denominator = int(seq_tuple[1][pos1][dp1 + 1:])
        except ValueError:
            print('the error equation is', seq_tuple)
            print(pos1, pos2)
            exit()
    small_pos = min(pos1, pos2)
    big_pos = max(pos1, pos2)
    if small_pos == pos1:
        dps = dp1
        dpb = dp2
    else:
        dps = dp2
        dpb = dp1
    eqs_pos = seq_tuple[2].index('=')
    nominators = int(seq_tuple[1][small_pos][0: dps]) if dps != -1 else int(seq_tuple[1][small_pos][0:])
    nominatorb = int(seq_tuple[1][big_pos][0: dpb]) if dpb != -1 else int(seq_tuple[1][big_pos][0:])
    if denominator != 1 and dps == -1:
        nominators *= denominator
    elif denominator != 1 and dpb == -1:
        nominatorb *= denominator

    signs = 1 if seq_tuple[0][small_pos] == '+' else -1
    signb = 1 if ((small_pos - eqs_pos) * (big_pos - eqs_pos) > 0 and seq_tuple[0][big_pos] == '+') or\
        ((small_pos - eqs_pos) * (big_pos - eqs_pos) < 0 and seq_tuple[0][big_pos] == '-') else -1
    new_nominator = signs * nominators + signb * nominatorb
    if new_nominator != 0:
        seq_tuple[0][small_pos] = '+' if new_nominator > 0 else '-'
        if denominator != 1:
            seq_tuple[1][small_pos] = '%d/%d' % (abs(new_nominator), denominator)
            seq_tuple = reduction(seq_tuple, small_pos)[0]
        else:
            seq_tuple[1][small_pos] = '%d' % abs(new_nominator)
        seq_tuple[0].pop(big_pos)
        seq_tuple[1].pop(big_pos)
        seq_tuple[2].pop(big_pos)
    else:
        seq_tuple[0].pop(big_pos)
        seq_tuple[1].pop(big_pos)
        seq_tuple[2].pop(big_pos)
        seq_tuple[0].pop(small_pos)
        seq_tuple[1].pop(small_pos)
        seq_tuple[2].pop(small_pos)

    return check0(seq_tuple)[0], True

def constant_multiply(seq_tuple):
    seq_tuple = deepcopy(seq_tuple)
    denoms = []
    noms = []
    for idx, term in enumerate(seq_tuple[2]):
        if term != '=' and term != '0':
            dpos = seq_tuple[1][idx].find('/')
            if dpos == -1:
                denoms.append(1)
            else:
                denoms.append(int(seq_tuple[1][idx][dpos + 1:]))
            if dpos != -1:
                noms.append(int(seq_tuple[1][idx][0: dpos]))
            else:
                noms.append(int(seq_tuple[1][idx][0:]))
    i = 0
    lcm = np.lcm.reduce(denoms)
    if lcm == 1:
        return seq_tuple, False
    for idx in range(len(seq_tuple[2])):
        if seq_tuple[2][idx] != '=' and seq_tuple[2][idx] != '0':
            seq_tuple[1][idx] = '%d' % (noms[i] * lcm / denoms[i])
            i += 1
    return seq_tuple, True

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


def next_states(seq_tuple):
    length = len(seq_tuple[0])
    states = []
    count = 0
    for pos in range(length):
        for s in range(10):
            if s > 1:
                seq_t, valid = scale(seq_tuple, pos, s)
                if valid:
                    states.extend([seq_t[:]])
                    count += 1
                    #print('call scale', pos)
                    #print(equation_str(seq_t))
        seq_t, valid = reduction(seq_tuple, pos)
        if valid:
            states.extend([seq_t[:]])
            count += 1
            #print('call reduction', pos)
            #print(equation_str(seq_t))
        for pos2 in range(length):
            seq_t, valid = move(seq_tuple, pos, pos2)
            if valid:
                count += 1
                states.extend([seq_t[:]])
                #print('call move', pos, pos2)
                #print(equation_str(seq_t))
        for pos2 in range(length)[pos+1:]:
            seq_t, valid = merge(seq_tuple, pos, pos2)
            if valid:
                count += 1
                states.extend([seq_t[:]])
                #print('call merge', pos, pos2)
                #print(equation_str(seq_t))
    seq_t, valid = constant_multiply(seq_tuple)
    if valid:
        count += 1
        states.extend([seq_t[:]])
        #print('call constant_multiply')                                                                                                                                       
        #print(equation_str(seq_t))

    return states
        
def equation_str(seq_tuple):
    equation = ''
    for i in range(len(seq_tuple[0])):
        for j in range(3):
            equation += seq_tuple[j][i]
    return equation

def tuple2str(seq_tuple):
    merge_seq = list(zip(*seq_tuple))
    return ' '.join([''.join(tup) for tup in merge_seq])
    
def encode(string):
    codebook_ = [str(digit) for digit in range(10)] + ['+', '-', '/', '^', '='] + ['x', 'y'] + [' ']
    digits = [codebook_.index(s) for s in string]
    return digits


def greedy_search(seq_tuple, eqv):
    states = next_states(seq_tuple)
    states = states[:] + [seq_tuple]
    encoded_states = [encode(tuple2str(tup)) for tup in states]
    #print(len(encoded_states))

    encoding_idx = []
    for i in range(len(encoded_states)):
        encoding_idx.append([i, len(encoded_states[i])-1])
    encoding_idx = np.array(encoding_idx)
    
    M = max(len(s) for s in encoded_states)
    equations = np.array([s + [eqv.config_.num_character] * (M - len(s)) for s in encoded_states])
    eqs_idx = np.expand_dims(equations, axis=-1)
    
    states_vals_ = eqv.sess_.run([eqv.lower_vals_], {eqv.lower_eqs_idx_: eqs_idx, eqv.initial_states_: np.zeros([len(encoded_states), eqv.config_.rnn_dim]), eqv.lower_encoding_idx_: encoding_idx})
    states_vals_ = states_vals_[0]
    max_val = np.amax(states_vals_)
    print('equation is', tuple2str(seq_tuple))
    print('current state value is', states_vals_[-1])
    print('max state value is', max_val)
    print()
    if states_vals_[-1] == max_val:
        return seq_tuple
    index = np.where(states_vals_ == max_val)[0]
    next_index = np.random.choice(index, 1)[0]
    return greedy_search(states[next_index], eqv)

def beam_search_(current_states, width, eqv):
    states = deepcopy(current_states)
    for s in current_states:
        print('current equation', tuple2str(s))
        for ss in next_states(s):
            if ss not in states:
                states.append(ss)

    encoded_states = [encode(tuple2str(tup)) for tup in states]
    encoding_idx = []
    for i in range(len(encoded_states)):
        encoding_idx.append([i, len(encoded_states[i])-1])
    encoding_idx = np.array(encoding_idx)
    
    M = max(len(s) for s in encoded_states)
    equations = np.array([s + [eqv.config_.num_character] * (M - len(s)) for s in encoded_states])
    eqs_idx = np.expand_dims(equations, axis=-1)

    states_vals_ = eqv.sess_.run([eqv.lower_vals_], {eqv.lower_eqs_idx_: eqs_idx, eqv.initial_states_: np.zeros([len(encoded_states), eqv.config_.rnn_dim]), eqv.lower_encoding_idx_: encoding_idx})
    states_vals_ = states_vals_[0]
    max_val = np.amax(states_vals_)
    
    print('current state values', states_vals_[0:width])
    print('max state value is', max_val)
    print()
    
    for i in range(width):
        if states_vals_[i] == max_val:
            return current_states[i]

    indices = np.argsort(states_vals_)
    index = len(indices) - 1
    next_states_ = []
    for i in range(width):
        next_states_.append(states[indices[index-i]])
    return beam_search_(next_states_, width ,eqv)

def beam_search(seq_tuple, width, eqv):
    current_states = []
    for i in range(width):
        current_states.append(deepcopy(seq_tuple))
    return beam_search_(current_states, width, eqv)

def main():
    eqv_config = edict({'encoding_dims': 20, 'rnn_dim': 30, 'C': 1, 'lr': 5e-5, 'num_character': 18, 'batch_size': 100})
    init_w = np.random.uniform(size = [1, eqv_config.rnn_dim])
    sess = tf.Session()
    eqv = EqValue(eqv_config, init_w, sess)
    init = tf.global_variables_initializer()                                                                                                                                        
    sess.run(init)   

    ckpt_dir = 'CKPT_rnn_dim_30_lr_5e-5_encoding_dims_20_2_4'
    eqv.restore_ckpt(ckpt_dir)
    width = 6
    eq = Equation(2, 4, 20, 5)
    c = 0
    for i in range(1000):
        equation = eq.generate()
        #print(equation_str(beam_search(equation, 6, eqv)))
        #print(tuple2str(equation))
        greed_search_equation = tuple2str(greedy_search(equation,eqv))
        history = eq.simplify(equation)
        #print('rule based simpification', history[-1][:-1])
        #print(greed_search_equation==history[-1][:-1])
        if greed_search_equation==history[-1][:-1]:
            c = c + 1
    print(c/1000)
    '''
    encoded_equation = encode(history[-1][:-1]) 
    encoded_equation_e = encode(history[-1][:-1]) + [eqv_config.num_character] * 10
    eqs_idx = np.expand_dims(np.array([encoded_equation]), axis=-1)
    eqs_idx_e = np.expand_dims(np.array([encoded_equation_e]), axis=-1)

    encoding_idx = np.array([[0, len(encoded_equation) - 1]]) 
    
    states_vals_, encoding1 = eqv.sess_.run([eqv.lower_vals_, eqv.lower_eq_encodings_], {eqv.lower_eqs_idx_: eqs_idx, eqv.initial_states_: np.zeros([1, eqv.config_.rnn_dim]), eqv.lower_encoding_idx_: encoding_idx})

    states_vals_e, encoding1 = eqv.sess_.run([eqv.lower_vals_, eqv.lower_eq_encodings_], {eqv.lower_eqs_idx_: eqs_idx_e, eqv.initial_states_: np.zeros([1, eqv.config_.rnn_dim]), eqv.lower_encoding_idx_: encoding_idx})
    
    #for e in encoding1[0]:
        #print(e)
    print(states_vals_)
    print(states_vals_e)
    '''
    '''
    for i in range(100):
        equation = eq.generate()
        #equation = [['-', '', '-', '-'], ['2/10', '', '34/2', '17'], ['x^3', '=', 'x^1z^1w^1', 'x^1z^1w^1']]
        #print('equation is')
        #print(equation_str(equation))
        #print(equation)
    
        states, count = next_states(equation)
        count_all += count
    print(count_all/100)
    '''
    '''
    equation = eq.generate()
    print('equation is')                                                                                                                                                                   
    print(equation_str(equation))
    states, count = next_states(equation)
    print('next states are')
    for state in states:
        print(equation_str(state))
        print()
    '''
    
if __name__ == '__main__':
    main()
