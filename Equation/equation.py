from functools import cmp_to_key
from collections import defaultdict
import numpy as np
from scipy.special import comb
import copy

import pdb
class Equation:
    def __init__(self, num_var, order, coef_range_nom, coef_range_denom,
                 length_mean = 4, length_low = 1, length_high = 7):
        assert(num_var <= 4)
        assert(order <= 9 and order > 0)
        self.num_var_ = num_var
        self.order_ = order
        self.coef_range_nom_ = coef_range_nom
        self.coef_range_denom_ = coef_range_denom
        self.length_mean_ = length_mean
        self.length_low_ = length_low
        self.length_high_ = length_high
        all_var_names = ['x', 'y', 'z', 'w']
        self.all_var_names_ = all_var_names[0: self.num_var_]
        self.all_var_names_.append('1')
        all_variables_stack = [[]]
        self.all_variables_ = []
        self.all_coefs_ = []

        #sort by the exp-degree of variables
        def cmp_term(term1, term2):
            if term1 == term2:
                return 0
            deg1 = term1[2::3]
            deg2 = term2[2::3]
            var1 = term1[0::3]
            var2 = term2[0::3]
            i = 0
            while True:
                if i == len(var1):
                    return -1
                elif i == len(var2):
                    return 1
                vidx1 = self.all_var_names_.index(var1[i])
                vidx2 = self.all_var_names_.index(var2[i])
                if vidx1 != vidx2:
                    return vidx2 - vidx1
                d1 = int(deg1[i])
                d2 = int(deg2[i])
                if d1 != d2:
                    return d1 - d2
                i += 1
            
        self.cmp_ = cmp_term

        while len(all_variables_stack) > 0:
            current = all_variables_stack.pop(0)
            for v in self.all_var_names_:
                current_cpy = copy.deepcopy(current)
                current_cpy.append(v)
                name = ''
                if len(current_cpy) == self.order_:
                    for vn in self.all_var_names_[0: -1]:
                        c = current_cpy.count(vn)
                        if c != 0:
                            name += '%s^%d' % (vn, c)
                    if name == '':
                        name = '1'
                    self.all_variables_.append(name)
                else:
                    all_variables_stack.append(current_cpy)
        self.all_variables_ = list(set(self.all_variables_))
        assert(len(self.all_variables_) == int(comb(self.num_var_ + self.order_, self.num_var_)))

        for i in range(1, self.coef_range_nom_ + 1):
            for j in range(1, self.coef_range_denom_ + 1):
                g = np.gcd(i, j)
                ii = i / g
                jj = j / g
                if jj == 1:
                    self.all_coefs_.append('%d' % ii)
                else:
                    self.all_coefs_.append('%d/%d' % (ii, jj))
        self.all_coefs_ = list(set(self.all_coefs_))
        self.codebook_ = [str(digit) for digit in range(10)] + ['+', '-', '/', '^', '='] + self.all_var_names_[0: -1] + [' ']

    def generate(self):
        left_length = 0
        right_length = 0
        while left_length < self.length_low_ or left_length > self.length_high_:
            left_length = np.random.poisson(self.length_mean_)
        while right_length < self.length_low_ or right_length > self.length_high_:
            right_length = np.random.poisson(self.length_mean_)
        sign_sequence = []
        coef_sequence = []
        var_sequence = []
        for i in range(left_length):
            rd = np.random.uniform()
            sign = '+' if rd <= 0.5 else '-'
            coef = self.all_coefs_[np.random.randint(len(self.all_coefs_))]
            var = self.all_variables_[np.random.randint(len(self.all_variables_))]
            sign_sequence.append(sign)
            coef_sequence.append(coef)
            var_sequence.append(var if var != '1' else '')
        sign_sequence.append('')
        coef_sequence.append('')
        var_sequence.append('=')
        for i in range(right_length):
            rd = np.random.uniform()
            sign = '+' if rd <= 0.5 else '-'
            coef = self.all_coefs_[np.random.randint(len(self.all_coefs_))]
            var = self.all_variables_[np.random.randint(len(self.all_variables_))]
            sign_sequence.append(sign)
            coef_sequence.append(coef)
            var_sequence.append(var if var != '1' else '')
        return [sign_sequence, coef_sequence, var_sequence]
    
    def tuple2str(self, seq_tuple):
        merge_seq = list(zip(*seq_tuple))
        return ' '.join([''.join(tup) for tup in merge_seq])
    
    def encode(self, string):
        digits = [self.codebook_.index(s) for s in string]
        return digits
    
    def scale(self, seq_tuple, pos, scale):
        if pos < 0 or pos >= len(seq_tuple[0]):
            return seq_tuple, False
        if seq_tuple[2][pos] == '=':
            return seq_tuple, False
        if seq_tuple[1][pos].find('/') == -1:
            return seq_tuple, False
        dash_pos = seq_tuple[1][pos].find('/')
        nominator = int(seq_tuple[1][pos][0: dash_pos])
        denominator = int(seq_tuple[1][pos][dash_pos + 1:])
        seq_tuple[1][pos] = '%d/%d' % (nominator * scale, denominator * scale)
        return seq_tuple, True

    def reduction(self, seq_tuple, pos):
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
        

    def check0(self, seq_tuple):
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

    def swap(self, seq_tuple, pos1, pos2):
        if pos1 < 0 or pos1 > len(seq_tuple) or\
           pos2 < 0 or pos2 > len(seq_tuple) or\
           pos1 == pos2:
            return seq_tuple, False
        eqs_pos = seq_tuple[2].index('=')
        loc = (pos1 - eqs_pos) * (pos2 - eqs_pos)
        if loc == 0:
            return seq_tuple, False
        elif loc > 0:
            temp_sign = seq_tuple[0][pos1]
            temp_coef = seq_tuple[1][pos1]
            temp_var = seq_tuple[2][pos1]
            seq_tuple[0][pos1] = seq_tuple[0][pos2]
            seq_tuple[1][pos1] = seq_tuple[1][pos2]
            seq_tuple[2][pos1] = seq_tuple[2][pos2]
            seq_tuple[0][pos2] = temp_sign
            seq_tuple[1][pos2] = temp_coef
            seq_tuple[2][pos2] = temp_var
        else:
            temp_sign = seq_tuple[0][pos1]
            temp_coef = seq_tuple[1][pos1]
            temp_var = seq_tuple[2][pos1]
            seq_tuple[0][pos1] = '-' if seq_tuple[0][pos2] == '+' else '+'
            seq_tuple[1][pos1] = seq_tuple[1][pos2]
            seq_tuple[2][pos1] = seq_tuple[2][pos2]
            seq_tuple[0][pos2] = '-' if temp_sign == '+' else '+'
            seq_tuple[1][pos2] = temp_coef
            seq_tuple[2][pos2] = temp_var
        
        return seq_tuple, True

    # move elements at pos1, so that it becomes pos2 in the resulting sequence
    def move(self, seq_tuple, pos1, pos2):
        if seq_tuple[2][pos1] == '=' or pos1 == pos2:
            return seq_tuple, False
        eqs_pos1 = seq_tuple[2].index('=')
        seq_tuple[0].insert(pos2, seq_tuple[0].pop(pos1))
        seq_tuple[1].insert(pos2, seq_tuple[1].pop(pos1))
        seq_tuple[2].insert(pos2, seq_tuple[2].pop(pos1))
        eqs_pos2 = seq_tuple[2].index('=')

        if(pos1 - eqs_pos1) * (pos2 - eqs_pos2) < 0:
            seq_tuple[0][pos2] = '-' if seq_tuple[0][pos2] == '+' else '+'

        return self.check0(seq_tuple)[0], True
    
    def merge(self, seq_tuple, pos1, pos2):
        if seq_tuple[2][pos1] != seq_tuple[2][pos2] or pos1 == pos2:
            return seq_tuple
        assert(seq_tuple[2][pos1] != '=')
        assert(seq_tuple[2][pos2] != '=')
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
                seq_tuple = self.reduction(seq_tuple, small_pos)[0]
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

        return self.check0(seq_tuple)[0], True
    
    def merge_helper(self, seq_tuple, same_denom = True):
        history = []
        repeat_vars = []
        term2pos = defaultdict(list)
        for idx, term in enumerate(seq_tuple[2]):
            if term != '=' and term != '0':
                term2pos[term].append(idx)
        term_repeat = [term for term in term2pos if len(term2pos[term]) > 1]
        term_repeat.sort(key = cmp_to_key(self.cmp_), reverse = True)
        while len(term_repeat) > 0:
            current_term = term_repeat.pop(0)
            poses = [idx for idx, term in enumerate(seq_tuple[2]) if term == current_term]
            denoms = []
            for p in poses:
                denom = 1 if seq_tuple[1][p].find('/') == -1 else int(seq_tuple[1][p][seq_tuple[1][p].find('/') + 1: ])
                denoms.append(denom)
            if not same_denom:
                lcm = np.lcm(denoms[-1], denoms[-2])
                f1 = lcm / denoms[-1]
                f2 = lcm / denoms[-2]
                if f1 != 1:
                    if self.scale(seq_tuple, poses[-1], f1)[1]:
                        history.append(self.tuple2str(seq_tuple))
                if f2 != 1:
                    if self.scale(seq_tuple, poses[-2], f2)[1]:
                        history.append(self.tuple2str(seq_tuple))
                if self.merge(seq_tuple, poses[-1], poses[-2])[1]:
                    history.append(self.tuple2str(seq_tuple))
                return history
            else:
                Ud, Udc = np.unique(denoms, return_counts = True)
                biggest_idx = 0
                biggest_idx_denominator = None
                for idx, d in enumerate(Ud):
                    last_occurance = len(denoms) - 1 - denoms[::-1].index(d)
                    if Udc[idx] > 1 and last_occurance > biggest_idx:
                        biggest_idx = last_occurance
                        biggest_idx_denominator = Ud[idx]
                if biggest_idx_denominator is not None:
                    for i in range(biggest_idx - 1, -1, -1):
                        if denoms[i] == biggest_idx_denominator:
                            if self.merge(seq_tuple, poses[biggest_idx], poses[i])[1]:
                                history.append(self.tuple2str(seq_tuple))
                            return history

        return history
        
    def simplify(self, seq_tuple):
        history = [self.tuple2str(seq_tuple)]
        # 1. merge terms with same denominator
        while True:
            new_hist = self.merge_helper(seq_tuple, True)
            if len(new_hist) == 0:
                break
            history += new_hist
        # 2. merge terms with different denominator
        while True:
            new_hist = self.merge_helper(seq_tuple, False)
            if len(new_hist) == 0:
                break
            history += new_hist
        # 3. remove denominators
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
        for idx in range(len(seq_tuple[2])):
            if seq_tuple[2][idx] != '=' and seq_tuple[2][idx] != '0':
                seq_tuple[1][idx] = '%d' % (noms[i] * lcm / denoms[i])
                i += 1
        history.append(self.tuple2str(seq_tuple))
        # 4. sort by descending exp-degress
        variables = [term for term in seq_tuple[2] if term != '=' and term != '0']
        variables.sort(key = cmp_to_key(self.cmp_), reverse = True)
        i = 0
        while i < len(variables):
            pos = seq_tuple[2].index(variables[i])
            if self.move(seq_tuple, pos, i)[1]:
                history.append(self.tuple2str(seq_tuple))
            i += 1
        return history


def main():
    data_size = 100000
    file_name = '../Data/sorted_equations.txt'
    eq = Equation(4, 9, 20, 10)
    f = open(file_name, 'w')
    
    for i in range(data_size):
        equation = eq.generate()
        history = eq.simplify(equation)
        for h in history:
            f.write(h)
            f.write(';')
        f.write('\n')
    f.close()
    f = open(file_name, 'r')
    seq_encodes = []
    lines = f.readlines()
    for l in lines:
        seq_encode = []
        raw_eqs = l[0: -1].split(';')[0: -1]
        seq_encode = [eq.encode(raw_eq) for raw_eq in raw_eqs]
        seq_encodes.append(seq_encode)
    seq_encodes = np.array(seq_encodes)
    #seq_encodes = np.expand_dims(seq_encodes, axis=-1)
    np.save('../Data/equations_encoded.npy', seq_encodes)
    
if __name__ == '__main__':
    main()

