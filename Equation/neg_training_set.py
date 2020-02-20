import numpy as np
from easydict import EasyDict as edict
import os

import json
from functools import cmp_to_key
from copy import deepcopy

from equation import Equation
import search
from search import next_states

def str_tuple(string):
    string = string[:-1]
    terms = string.split()
    sign_list = []
    coef_list = []
    var_list = []
    for t in terms:
        if t == '=':
            sign_list.append('')
            coef_list.append('')
            var_list.append('=')
        elif t == '0':
            sign_list.append('')
            coef_list.append('')
            var_list.append('0')
        else:
            xpos = t.find('x')
            ypos = t.find('y')
            if xpos == -1 and ypos == -1:
                var_list.append('')
                vpos = len(t)
            elif xpos == -1:
                vpos = ypos
                var_list.append(t[vpos:])
            else:
                vpos = xpos
                var_list.append(t[vpos:])
            sign_list.append(t[0])
            coef_list.append(t[1:vpos])
    return [sign_list, coef_list, var_list]

def main():
    eq = Equation(2, 4, 20, 5)
    file_name = '../Data/equations_2_4_20_5.txt'
    f = open(file_name, 'r')
    seq_encodes = []
    lines = f.readlines()
    c = 0
    
    for l in lines:
        print(c)
        c += 1
        seq_encode = []
        raw_eqs = l[0: -1].split(';')[0: -1]
        h_len = len(raw_eqs)

        for i in range(h_len-1):
            j = 0
            equation = str_tuple(raw_eqs[i])
            next_states_list = [eq.tuple2str(t[0]) + t[1] for  t in next_states(equation)]
            while True:
                index = np.random.choice(len(next_states_list), 1)
                temp = next_states_list[index[0]]
                if temp != raw_eqs[i+1]:
                    seq_encode.append([eq.encode(temp), eq.encode(raw_eqs[i])])
                    #print(temp)
                    break

        for i in range(int(h_len/2), h_len-1):
            equation = str_tuple(raw_eqs[i])
            next_states_list = [eq.tuple2str(t[0]) + t[1] for  t in next_states(equation)]
            while True:
                index = np.random.choice(len(next_states_list), 1)
                temp = next_states_list[index[0]]
                if temp != raw_eqs[i+1]:
                    seq_encode.append([eq.encode(temp), eq.encode(raw_eqs[i])])
                    break
                
        equation = str_tuple(raw_eqs[h_len-1])
        next_states_list = [eq.tuple2str(t[0]) + t[1] for t in next_states(equation)]
        temp = next_states_list[np.random.choice(len(next_states_list), 1)[0]]
        seq_encode.append([eq.encode(temp), eq.encode(raw_eqs[h_len-1])])
        temp = next_states_list[np.random.choice(len(next_states_list), 1)[0]]
        seq_encode.append([eq.encode(temp), eq.encode(raw_eqs[h_len-1])])
        #print(seq_encode)    
        seq_encodes.append(seq_encode)
    seq_encodes = np.array(seq_encodes)
    np.save('../Data/neg_training_set_2_4_20_5.npy', seq_encodes)
            
if __name__ == '__main__':
    main()
