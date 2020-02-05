import numpy as np
from equation import Equation
from copy import deepcopy

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
        denominator2 = int(seq_tuple[1][pos2][dp2 + 1:])
        if denominator1 != denominator2:
            return seq_tuple, False
        denominator = denominator1
    elif dp1 == -1:
        denominator = int(seq_tuple[1][pos2][dp2 + 1:])
    elif dp2 == -1:
        denominator = int(seq_tuple[1][pos1][dp1 + 1:])

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

def next_states(seq_tuple):
    length = len(seq_tuple[0])
    states = []
    
    for pos in range(length):
        for s in range(10):
            if s > 1:
                seq_t, valid = scale(seq_tuple, pos, s)
                if valid:
                    states.extend([seq_t[:]])
                    #print('call scale', pos)
                    #print(equation_str(seq_tuple))
        seq_t, valid = reduction(seq_tuple, pos)
        if valid:
            states.extend([seq_t[:]])
            #print('call reduction', pos)
            #print(equation_str(seq_tuple))
        for pos2 in range(length):
            seq_t, valid = move(seq_tuple, pos, pos2)
            if valid:
                states.extend([seq_t[:]])
                print('call move', pos)
                print(equation_str(seq_tuple))
        for pos2 in range(length)[pos+1:]:
            seq_t, valid = merge(seq_tuple, pos, pos2)
            if valid:
                states.extend([seq_t[:]])
                #print('call merge', pos)
                #print(equation_str(seq_tuple))
    seq_t, valid = constant_multiply(seq_tuple)
    if valid:
        states.extend([seq_t[:]])
        print('call constant_multiply')                                                                                                                                       
        print(equation_str(seq_tuple))
    return states
        
def equation_str(seq_tuple):
    equation = ''
    for i in range(len(seq_tuple[0])):
        for j in range(3):
            equation += seq_tuple[j][i]
    return equation
    
def main():
    eq = Equation(4, 3, 20, 5)
    '''
    for i in range(100):
        equation = eq.generate()
        #equation = [['-', '', '-', '-'], ['2/10', '', '34/2', '17'], ['x^3', '=', 'x^1z^1w^1', 'x^1z^1w^1']]
        print('equation is')
        print(equation_str(equation))
        #print(equation)
    
        states = next_states(equation)
    '''
    equation = eq.generate()
    print('equation is')                                                                                                                                                                   
    print(equation_str(equation))
    states = next_states(equation)
    print('next states are')
    for state in states:
        print(equation_str(state))
        print()
    
if __name__ == '__main__':
    main()
