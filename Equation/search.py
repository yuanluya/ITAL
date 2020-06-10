from functools import cmp_to_key
import numpy as np
from equation import Equation
from eq_value_cnn import EqValue
import tensorflow as tf

from easydict import EasyDict as edict
from copy import deepcopy
import pdb
from tqdm import tqdm
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

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
        if seq_tuple[2].index('=') == len(seq_tuple[2]) - 1:
            seq_tuple[0].append('')
            seq_tuple[1].append('')
            seq_tuple[2].append('0')
        return seq_tuple, True
    if seq_tuple[2].index('=') == len(seq_tuple[2]) - 1:
        seq_tuple[0].append('')
        seq_tuple[1].append('')
        seq_tuple[2].append('0')
        return seq_tuple, True
    if '0' in seq_tuple[2] and seq_tuple[2].index('0') == len(seq_tuple[2]) - 1 and seq_tuple[2].index('=') != len(seq_tuple[2]) - 2:
        seq_tuple[0].pop()
        seq_tuple[1].pop()
        seq_tuple[2].pop()
        return seq_tuple, True
    if '0' in seq_tuple[2] and seq_tuple[2].index('0') == 0 and seq_tuple[2].index('=') != 1:
        seq_tuple[0].pop(0)
        seq_tuple[1].pop(0)
        seq_tuple[2].pop(0)
        return seq_tuple, True
    if '0' in seq_tuple[2] and seq_tuple[2].index('0') != 0 and seq_tuple[2].index('0') != len(seq_tuple[2]) - 1:
        seq_tuple[0].pop(seq_tuple[2].index('0'))
        seq_tuple[1].pop(seq_tuple[2].index('0'))
        seq_tuple[2].pop(seq_tuple[2].index('0'))
        return seq_tuple, True
    return seq_tuple, False

def move(seq_tuple, pos1, pos2):
    seq_tuple = deepcopy(seq_tuple)
    eqs_pos = seq_tuple[2].index('=')
    if pos2 > eqs_pos:
        return seq_tuple, False
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
    if seq_tuple[2][pos1] == '=' or seq_tuple[2][pos2] == '=':
        return seq_tuple, False
    if seq_tuple[2][pos1] == '0' or seq_tuple[2][pos2] == '0':
        return seq_tuple, False
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

def cancel_gcd(seq_tuple):
    seq_tuple = deepcopy(seq_tuple)
    int_coefs = []
    for n in seq_tuple[1]:
        if '/' in n:
            int_coefs.append(int(n[:n.index('/')]))
        elif n != '':
            int_coefs.append(int(n))
    g = np.gcd.reduce(int_coefs)
    if g == 1:
        return seq_tuple, False
    for i in range(len(seq_tuple[1])):
        if '/' in seq_tuple[1][i]:
            seq_tuple[1][i] = '%d%s'  % (int(seq_tuple[1][i][:seq_tuple[1][i].index('/')]) / g, seq_tuple[1][i][seq_tuple[1][i].index('/'):])
        elif seq_tuple[1][i] != '':
            seq_tuple[1][i] = '%d'  % (int(seq_tuple[1][i]) / g)
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

        seq_t, valid = reduction(seq_tuple, pos)
        if valid:
            states.extend([seq_t[:]])
            count += 1

        for pos2 in range(length):
            seq_t, valid = move(seq_tuple, pos, pos2)
            if valid:
                count += 1
                states.extend([seq_t[:]])

        for pos2 in range(length)[pos+1:]:
            seq_t, valid = merge(seq_tuple, pos, pos2)
            if valid:
                count += 1
                states.extend([seq_t[:]])

    seq_t, valid = constant_multiply(seq_tuple)
    if valid:
        count += 1
        states.extend([seq_t[:]])

    seq_t, valid = cancel_gcd(seq_tuple)
    if valid:
        count += 1
        states.extend([seq_t[:]])

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
    codebook_ = [str(digit) for digit in range(10)] + ['+', '-', '/', '^', '='] + ['x', 'y', 'z'] + [' ']
    digits = [codebook_.index(s) for s in string]
    return digits

def beam_search_(current_states, width, eqv, M, w, maxd):
    maxd += 1
    states = deepcopy(current_states)
    for s in current_states:
        for ss in next_states(s):
            if ss not in states:
                states.append(ss)

    encoded_states = [encode(tuple2str(tup)) for tup in states]
    equations = np.array([s + [eqv.config_.num_character] * (M - len(s)) for s in encoded_states])
    eqs_idx = np.expand_dims(equations, axis=-1)
    encoding_idx = np.expand_dims(list(range(len(encoded_states))), axis = 1)


    states_vals = eqv.sess_.run([eqv.lower_encodings_], {eqv.lower_eqs_idx_: eqs_idx, eqv.higher_eqs_idx_: eqs_idx,
        eqv.lower_encoding_idx_: encoding_idx, eqv.higher_encoding_idx_: encoding_idx})
    states_vals = np.sum(states_vals[0] * w, axis = 1)

    max_val = np.max(states_vals)
    
    if states_vals[0] == max_val or maxd > 30:
        results = []
        results.append(current_states[0])
        return current_states[0], [max_val], results, maxd

    indices = np.argsort(-1 * states_vals)
    
    nexts = []
    strs = []
    vals = []
    rs = []
    vals_ = []
    for i in range(min(width, len(indices))):
        s, v, r, _ = beam_search_([states[indices[i]]], width, eqv, M, w, maxd)
        nexts.append(s)
        strs.append(tuple2str(s))
        vals.append(v)
        vals_.append(v[0])
        rs.append(r)

    chosen = np.argmax(vals_)
    rlist = rs[chosen]
    results = []
    vs = []
    for r in rlist:
        results.append(r)
    for v in vals[chosen]:
        vs.append(v)
    results.append(states[indices[chosen]])
    vs.append(max_val)
    return nexts[chosen], vs, results, maxd

def beam_search(seq_tuple, width, eqv, M, w):
    current_states = []
    current_states.append(deepcopy(seq_tuple))
    return beam_search_(current_states, width, eqv, M, w, 0)

def main():
    np.random.seed(1234)

    M = 173 #148
    eqv_config = edict({'input_dim': M, 'encoding_dim': 30, 'output_dim': 45, 'C': 1, 'reg_param': 1e-5, 'batch_size': 128,'lr': 1e-4, 
        'num_character': 19,
        'layer_info': [(64, 5, 1, False), (64, 5, 1, True), (32, 3, 1, False), (32, 3, 1, True),(32, 3, 1, False), (32, 3, 1, True)]})
    
    init_w = np.random.uniform(size = [1, eqv_config.output_dim + 1])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    np.random.seed(1234)

    eqv = EqValue(eqv_config, init_w, sess)
    init = tf.global_variables_initializer()
    sess.run(init)
    ckpt_dir = 'CKPT_cnn_dim_%d_encoding_dim_%d_3_4_6layers' % (eqv_config.output_dim, eqv_config.encoding_dim)
    eqv.restore_ckpt(ckpt_dir)
    
    eq = Equation(3, 4, 20, 5)
    
    temp = []
    c = 0
    w = np.load("equation_gt_weights_cnn_3var_45_6layers.npy")
    for width in [1]:
        for i in tqdm(range(1000)):
            equation = eq.generate()
            greedy_search_equation, v, rs, _  = beam_search(equation, width, eqv, M, w)
            greedy_search_equation = tuple2str(greedy_search_equation)
            history = eq.simplify(equation)
            if greedy_search_equation == history[-1][:-1]:
                c = c + 1
        temp.append(c/1000)
        c = 0
        print("ground truth %d %s" % (width, temp))

    np.save("gt_search_acc.npy" % (savename), np.mean(temp))
    
    for width in [1]:
        for rd in range(20):
            savename = "regression_1_45"
            for m in ["omni_", "imit2_", "imit3_"]:
                for s in ["batch_", "sgd_", "IMT_", "ITAL_"]:
                    acc = []
                    for d in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 750, 1000]:
                        w = np.load("%s%s%s_0_w.npy" % (s, m, savename))[d][0]                        
                        np.random.seed(rd)
                        for i in tqdm(range(1000)):
                            equation = eq.generate()
                            greedy_search_equation, v, rs, _ = beam_search(equation, width, eqv, M, w)
                            greedy_search_equation = tuple2str(greedy_search_equation)
                            history = eq.simplify(equation)
                            if greedy_search_equation == history[-1][:-1]:
                                c = c + 1
                        acc.append(c / 1000)
                        c = 0
                        
                    # print("%s %s %s %s" % (m, s, savename, acc))
                    np.save("%s%s%s_w_%d_curve_%d.npy" % (s, m, savename, width, rd), acc)
    
def plot():
    import seaborn as sns
    import csv, os
    import pandas as pd
    import matplotlib.pyplot as plt
    if plt.get_backend() == 'Qt5Agg':
        from matplotlib.backends.qt_compat import QtWidgets
        qApp = QtWidgets.QApplication(sys.argv)
        plt.matplotlib.rcParams['figure.dpi'] = qApp.desktop().physicalDpiX()

    palette = {"Omniscient ITAL":sns.xkcd_rgb["red"],"Imitate Dim-20 ITAL":sns.xkcd_rgb["burnt orange"], "Imitate Dim-30 ITAL":sns.xkcd_rgb["orange"], \
                "Batch":sns.xkcd_rgb["blue"], "SGD":sns.xkcd_rgb["purple"], \
                'Omniscient IMT': sns.xkcd_rgb['green'], 'Imitate Dim-20 IMT': sns.xkcd_rgb['dark green'], 'Imitate Dim-30 IMT': sns.xkcd_rgb['olive green'],\
                "Imitate CNN-9 ITAL":sns.xkcd_rgb["burnt orange"], "Imitate CNN-12 ITAL":sns.xkcd_rgb["orange"], \
                'Omniscient IMT': sns.xkcd_rgb['green'], 'Imitate CNN-9 IMT': sns.xkcd_rgb['dark green'], 'Imitate CNN-12 IMT': sns.xkcd_rgb['olive green'],\
                "Imitate ITAL":sns.xkcd_rgb["orange"], 'Imitate IMT': sns.xkcd_rgb['dark green'], \
                'Imitate Dim-50 ITAL':sns.xkcd_rgb["orange"], 'Imitate Dim-40 ITAL':sns.xkcd_rgb["burnt orange"], \
                'Imitate Dim-50 IMT': sns.xkcd_rgb['olive green'], 'Imitate Dim-40 IMT': sns.xkcd_rgb['dark green'],'Teacher Rewards Truth':sns.xkcd_rgb['grey']}
    dash = {"Omniscient ITAL": 'solid',"Imitate ITAL": 'solid',"Batch":'solid', "SGD": 'solid', \
            'Omniscient IMT': 'solid', 'Imitate Dim-40 IMT': 'solid', 'Imitate Dim-40 ITAL':'solid','Imitate Dim-50 IMT':'solid','Imitate IMT': 'solid','Imitate Dim-50 ITAL':'solid',"Teacher Rewards Truth":'dashed'}

    names = ["Batch", "SGD", "Omniscient IMT", "Omniscient ITAL", 'Imitate Dim-40 IMT', 'Imitate Dim-40 ITAL','Imitate Dim-50 IMT', 'Imitate Dim-50 ITAL']
    savename = "regression_1_45"
    
    with open('w_curve.csv', mode='w') as csv_file:
        fieldnames = ['Method', 'Iteration', 'Accuracy']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        idx = 0
        for m in ["omni_"]:
            x = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 750, 1000])
            for s in ["batch_", "sgd_", "IMT_", "ITAL_"]:
                for rd in range(20):
                    l = np.load("%s%s%s_w_1_curve_%d.npy" % (s, m, savename, rd))

                    for i in range(len(l)):
                        writer.writerow({'Method': names[idx], 'Iteration': x[i],  'Accuracy': l[i]})
                    
                idx+=1


        for m in ["imit2_", "imit3_"]:
            x = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 750, 1000])
            for s in ["IMT_", "ITAL_"]:
                for rd in range(20):
                    l = np.load("%s%s%s_w_1_curve_%d.npy" % (s, m, savename, rd))

                    for i in range(len(l)):
                        writer.writerow({'Method': names[idx], 'Iteration': x[i],  'Accuracy': l[i]})
                    
                idx+=1


        for i in range(len(x)):
            writer.writerow({'Method': 'Teacher Rewards Truth', 'Iteration': x[i], 'Accuracy': np.load("gt_search_acc.npy")})


    paper_rc = {'lines.linewidth': 2.5}
    sns.set(style="darkgrid")
    sns.set(font_scale=1.95, rc = paper_rc)

    plt.figure() 
    f, axes = plt.subplots(1, 1, constrained_layout = True, figsize=(10, 6)) 
    df = pd.read_csv('w_curve.csv')
    plt1 = sns.lineplot(x="Iteration", y="Accuracy",
                 hue="Method",data=df, ax=axes, palette=palette)
    # axes.axhline(np.load("gt_search_acc.npy"), color=sns.xkcd_rgb['grey'], linestyle='-')
    
    plt1.legend_.remove()

    axes.set_xlabel('Training Iteration', fontweight="bold", size=29)
    axes.set_title('Simplification Accuracy', fontweight="bold", size=29)
    axes.set_ylabel('')
    plt1.lines[8].set_linestyle('dashed')
    # plt.show()
    plt.savefig('%s_w.pdf' % (savename), dpi=300)

if __name__ == '__main__':
    main()
    # plot()
