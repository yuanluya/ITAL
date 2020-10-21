import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from easydict import EasyDict as edict
import multiprocessing
import time
import glob, os
import argparse
import pdb
def get_path(data_cate, type_ = 'd'):
    #add dist or dist_ or ... as argyment
    #note data_cate in different main functions are different
    titles = []
    methods = edict()

    if type_ != 'irl':
        lines = ['1', '8', 'batch', 'sgd']
    else:
        lines = ['0', '1', '2', '3']
    if data_cate == 'teacher_rewards':
        lines = ['']
    for l in lines:
            '''
            titles.append('dist'+l+title)
            titles.append('dist'+l+'_'+title)
            titles.append('logpdfs'+l+title)
            titles.append('accuracies'+l+title)
            '''
            if data_cate == 'dist_':
                t = 'dist'+l+'_'
            else:
                t = data_cate+l
            titles.append(t)
            methods[t] = l

    return titles, methods

def save_csv(data_cate, setting_name, mode, random_seeds, type_ = 'd'):
    methods_code = {'0': 'No Rep', '1': 'IMT', '2': 'Rand Rep', '3': 'prag', '4': 'Prag (Strt Lin)', '5': 'IMT (Strt Lin)', \
                    '7': 'cont_sgd', '8': 'ITAL', '6': 'Expert', 'batch': 'Batch', 'sgd': 'SGD', 'm2': 'ITAL 2', 'm5': 'ITAL 5',
                    'm10': 'ITAL 10', 'm15': 'ITAL 15'}
    if type_ == 'irl':
        methods_code = {'0': 'IMT', '1': 'ITAL', '2': 'Batch', '3': 'SGD', '': '', 'm2': 'ITAL 2', 'm5': 'ITAL 5',
                    'm10': 'ITAL 10', 'm15': 'ITAL 15'}

    titles, methods = get_path(data_cate, type_)
    data = []
    method = []
    iterations = []
    for t in titles:
        for s in random_seeds:
            filename = 'Experiments/' + setting_name + '/' + t + '_' + str(s) + '.npy'
            d = np.load(filename, allow_pickle = True)
            #d = d[:int(len(d)/4)]
            length = len(d)
            print(filename)
            data.append(d)
            method += [methods_code[methods[t]] for j in range(length)]
            iterations += [j for j in range(length)]

            if t[-1] == '8' or (type_ == 'irl' and t[-1] == '1'):
                for mini_size in [2, 5, 10, 15]:
                    filename = 'Experiments/' + setting_name + '/' + t + '_' + str(s) + '_' + str(mini_size) + '.npy'
                    d = np.load(filename, allow_pickle = True)
                    #d = d[:int(len(d)/4)]
                    length = len(d)
                    print(filename)
                    data.append(d)
                    method += [methods_code['m' + str(mini_size)] for j in range(length)]
                    iterations += [j for j in range(length)]

    data = np.array(data).flatten()

    #save_name = setting_name + '_csv/' + mode + '_' + setting_name + '_csv/' + data_cate + '_' + mode + '_' + setting_name + '.csv'
    save_name = 'Experiments/' + setting_name + '/' + data_cate + '_' + setting_name + '.csv'

    df = pd.DataFrame({'method': method,
                       'iteration': iterations,
                       'data': data})
    df.to_csv(save_name)
    print('saved file to %s \n' % save_name)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def collect_data(setting_name, mode, random_seeds, arguments, type_):
    child_processes = []

    cpu_cnt = int(multiprocessing.cpu_count()/10) + 1
    #cpu_cnt = 1
    random_seed = list(chunks(random_seeds, cpu_cnt))
    for ss in random_seed:
        child_processes = []
        for s in ss:
            arguments_ = arguments[:] + [str(s)]

            p = subprocess.Popen(arguments_)
            child_processes.append(p)
        for cp in child_processes:
            cp.wait()
        #subprocess.call(arguments_)
        #subprocess.call(arguments.append(str(s)))
        p.wait()

    if type_ == 'd' or type_ == 'c':
        save_csv('dist', setting_name, mode, random_seeds, type_)
        save_csv('dist_', setting_name, mode, random_seeds, type_)
        save_csv('losses', setting_name, mode, random_seeds, type_)
        save_csv('accuracies', setting_name, mode, random_seeds, type_)
    else:
        save_csv('action_dist', setting_name, mode, random_seeds, type_)
        save_csv('reward_dist', setting_name, mode, random_seeds, type_)
        save_csv('q_dist', setting_name, mode, random_seeds, type_)
        save_csv('rewards', setting_name, mode, random_seeds, type_)
        if mode == 'imit':
            save_csv('teacher_rewards', setting_name, mode, random_seeds, type_)
    print('collected data\n')


def plot(setting_name, imit_dim=None):
    main_settings = {'regression_coop', 'regression_adv', 'class10_coop', 'class10_adv'}
    classification = {'class10_coop', 'class10_adv', 'mnist'}
    irl_settings = {'irlH_coop', 'irlH_adv', 'irlE_coop', 'irlE_adv'}

    paper_rc = {'lines.linewidth': 2.5}
    sns.set(style="darkgrid")
    sns.set(font_scale=1.95, rc = paper_rc)

    directory = 'Experiments/'

    omni_path = directory + setting_name + '_omni/'
    imit_path = directory + setting_name + '_imit/'

    # palette = {"Omniscient ITAL":sns.xkcd_rgb["red"],"Imitate Dim-20 ITAL":sns.xkcd_rgb["burnt orange"], "Imitate Dim-30 ITAL":sns.xkcd_rgb["orange"], \
    #             "Batch":sns.xkcd_rgb["blue"], "SGD":sns.xkcd_rgb["purple"], \
    #             'Omniscient IMT': sns.xkcd_rgb['green'], 'Imitate Dim-20 IMT': sns.xkcd_rgb['dark green'], 'Imitate Dim-30 IMT': sns.xkcd_rgb['olive green'],\
    #             "Imitate CNN-9 ITAL":sns.xkcd_rgb["burnt orange"], "Imitate CNN-12 ITAL":sns.xkcd_rgb["orange"], \
    #             'Omniscient IMT': sns.xkcd_rgb['green'], 'Imitate CNN-9 IMT': sns.xkcd_rgb['dark green'], 'Imitate CNN-12 IMT': sns.xkcd_rgb['olive green'],\
    #             "Imitate ITAL":sns.xkcd_rgb["burnt orange"], 'Imitate IMT': sns.xkcd_rgb['dark green'], \
    #             'Imitate Dim-50 ITAL':sns.xkcd_rgb["orange"], 'Imitate Dim-40 ITAL':sns.xkcd_rgb["burnt orange"], \
    #             'Imitate Dim-50 IMT': sns.xkcd_rgb['olive green'], 'Imitate Dim-40 IMT': sns.xkcd_rgb['dark green']}

    palette = {"ITAL":sns.xkcd_rgb["red"],"ITAL 2":sns.xkcd_rgb["bright yellow"], "ITAL 5":sns.xkcd_rgb["golden yellow"],  "ITAL 10":sns.xkcd_rgb["orange"],  "ITAL 15":sns.xkcd_rgb["burnt orange"], \
               "Dim-20 ITAL":sns.xkcd_rgb["red"],"Dim-20 ITAL 2":sns.xkcd_rgb["bright yellow"], "Dim-20 ITAL 5":sns.xkcd_rgb["golden yellow"],  "Dim-20 ITAL 10":sns.xkcd_rgb["orange"],  "Dim-20 ITAL 15":sns.xkcd_rgb["burnt orange"], \
               "Dim-30 ITAL":sns.xkcd_rgb["red"],"Dim-30 ITAL 2":sns.xkcd_rgb["bright yellow"], "Dim-30 ITAL 5":sns.xkcd_rgb["golden yellow"],  "Dim-30 ITAL 10":sns.xkcd_rgb["orange"],  "Dim-30 ITAL 15":sns.xkcd_rgb["burnt orange"], \
               "Batch":sns.xkcd_rgb["blue"], "SGD":sns.xkcd_rgb["purple"], \
                'IMT': sns.xkcd_rgb['green'], 'Imitate Dim-20 IMT': sns.xkcd_rgb['dark green'], 'Imitate Dim-30 IMT': sns.xkcd_rgb['olive green'],\
                "Imitate CNN-9 ITAL":sns.xkcd_rgb["burnt orange"], "Imitate CNN-12 ITAL":sns.xkcd_rgb["orange"], \
                'Omniscient IMT': sns.xkcd_rgb['green'], 'Imitate CNN-9 IMT': sns.xkcd_rgb['dark green'], 'Imitate CNN-12 IMT': sns.xkcd_rgb['olive green'],\
                "Imitate ITAL":sns.xkcd_rgb["burnt orange"], 'Imitate IMT': sns.xkcd_rgb['dark green'], \
                'Imitate Dim-50 ITAL':sns.xkcd_rgb["orange"], 'Imitate Dim-40 ITAL':sns.xkcd_rgb["burnt orange"], \
                'Imitate Dim-50 IMT': sns.xkcd_rgb['olive green'], 'Imitate Dim-40 IMT': sns.xkcd_rgb['dark green']}

    dash = {"Omniscient ITAL": '',"Imitate ITAL": (5, 5),"Batch":'', "SGD": '', \
        'Omniscient IMT': '', 'Imitate IMT': (5, 5), "ITAL":'', 'IMT': '', \
            "ITAL":'',"ITAL 2":'', "ITAL 5":'',  "ITAL 10":'',  "ITAL 15":'', \
            "Dim-30 ITAL":(5, 5),"Dim-30 ITAL 2":(5, 5), "Dim-30 ITAL 5":(5, 5),  "Dim-30 ITAL 10":(5, 5),  "Dim-30 ITAL 15":(5, 5)}
        

    display_methods = [ 'Batch', 'SGD', 'IMT', 'ITAL', 'ITAL 2', 'ITAL 5', 'ITAL 10', 'ITAL 15']

    plt.figure()

    if setting_name in main_settings:
        f, axes = plt.subplots(1, 1, constrained_layout = True, figsize=(10, 6))
        results0_imit = pd.read_csv(imit_path + '%s.csv' % ('dist_'+setting_name+'_imit'))

        if setting_name in classification:
            results1_imit = pd.read_csv(imit_path + '%s.csv' % ('accuracies_'+setting_name+'_imit'))
        else:
            results1_imit = pd.read_csv(imit_path + '%s.csv' % ('losses_'+setting_name+'_imit'))

        df1 = results0_imit
        df2 = results1_imit

        plt1 = sns.lineplot(x="iteration", y="data",
                            hue="method", data=df1, ax=axes, palette=palette, ci=68)

        plt1.legend_.remove()
        plt1.set(xlabel='', ylabel='')
        axes.set_title('L2 Distance', fontweight="bold", size=29)
        plt.savefig('Experiments/' + setting_name + '_l2.pdf', dpi=300)

        plt.figure()
        f, axes = plt.subplots(1, 1, constrained_layout = True, figsize=(10, 6))
        plt2 = sns.lineplot(x="iteration", y="data",
                            hue="method", data=df2, ax=axes, palette=palette, ci=68)


        axes.set_title('Square Loss', fontweight="bold", size=29)
        if setting_name in classification:
            if setting_name == 'class4':
                axes.set_title('4-Class Classification Accuracy', fontweight="bold", size=29)
            else:
                axes.set_title('10-Class Classification Accuracy', fontweight="bold", size=29)

        plt2.legend_.remove()
        plt2.set(xlabel='', ylabel='')
        if setting_name in classification:
            plt.savefig('Experiments/' + setting_name + '_accuracy.pdf', dpi=300)
        else:
            plt.savefig('Experiments/' + setting_name + '_squareLoss.pdf', dpi=300)

    elif setting_name == 'equation_coop' or setting_name == 'equation_adv':
        df0 = pd.read_csv(imit_path[:-1] + '_' + imit_dim + '/'  + '%s.csv' % ('dist' + '_'+setting_name+'_imit_' + imit_dim))
        df1 = pd.read_csv(imit_path[:-1] + '_' + imit_dim + '/'  + '%s.csv' % ('losses'+'_'+setting_name+'_imit_' + imit_dim))

        f, axes = plt.subplots(1, 1, constrained_layout = True, figsize=(10, 6))
        plt1 = sns.lineplot(x="iteration", y="data",
                            hue="method",data=df0, ax=axes, palette=palette, ci=68)
        plt1.legend_.remove()
        axes.set_xlabel('Training Iteration', fontweight="bold", size=29)
        axes.set_ylabel('')
        axes.set_title('L2 Distance', fontweight="bold", size=29)
        plt.savefig('Experiments/' + setting_name +imit_dim + '_l2.pdf', dpi=300)

        plt.figure()
        f, axes = plt.subplots(1, 1, constrained_layout = True, figsize=(10, 6))
        plt2 = sns.lineplot(x="iteration", y="data",
                            hue="method",data=df1, ax=axes, palette=palette, ci=68)
        plt2.legend_.remove()
        axes.set_title('Square Loss', fontweight="bold", size=29)
        axes.set_xlabel('Training Iteration', fontweight="bold", size=29)
        axes.set_ylabel('')
        plt.savefig('Experiments/' + setting_name + imit_dim + '_squareLoss.pdf', dpi=300)

    elif setting_name == 'mnist_coop' or setting_name == 'mnist_adv' :
        df0 = pd.read_csv(imit_path[:-1] + '_' + imit_dim + '/'  + '%s.csv' % ('dist' + '_'+setting_name+'_imit_' + imit_dim))
        df1 = pd.read_csv(imit_path[:-1] + '_' + imit_dim + '/'  + '%s.csv' % ('accuracies'+ '_'+setting_name+'_imit_' + imit_dim ))

        f, axes = plt.subplots(1, 1, constrained_layout = True, figsize=(10, 6))
        plt1 = sns.lineplot(x="iteration", y="data",
                            hue="method",data=df0, ax=axes, palette=palette, ci=68)
        plt1.legend_.remove()
        plt1.set(xlabel='')
        axes.set_ylabel('')
        axes.set_title('L2 Distance', fontweight="bold", size=29)
        plt.savefig('Experiments/' + setting_name +imit_dim + '_l2.pdf', dpi=300)

        plt.figure()
        f, axes = plt.subplots(1, 1, constrained_layout = True, figsize=(10, 6))
        plt2 = sns.lineplot(x="iteration", y="data",
                            hue="method",data=df1, ax=axes, palette=palette, ci=68)
        plt2.legend_.remove()
        plt2.set(xlabel='')
        
        axes.set_title('10-Class Classification Accuracy', fontweight="bold", size=29)
        axes.set_ylabel('')
        plt.savefig('Experiments/' + setting_name + imit_dim + '_accuracy.pdf', dpi=300)

    elif setting_name == 'cifar_coop' or setting_name == 'cifar_adv': #to be modified to add more settings
        df0 = pd.read_csv(imit_path[:-1] + '_' + imit_dim + '/'  + '%s.csv' % ('dist' + '_'+setting_name+'_imit_' + imit_dim))
        df1 = pd.read_csv(imit_path[:-1] + '_' + imit_dim + '/'  + '%s.csv' % ('accuracies'+'_'+setting_name+'_imit_' + imit_dim))

        f, axes = plt.subplots(1, 1, constrained_layout = True, figsize=(10, 6))
        plt1 = sns.lineplot(x="iteration", y="data",
                            hue="method",data=df0, ax=axes, palette=palette, ci=68)
        plt1.legend_.remove()
        plt1.set(xlabel='')
        axes.set_ylabel('')
        axes.set_title('L2 Distance', fontweight="bold", size=29)
        plt.savefig('Experiments/' + setting_name +imit_dim + '_l2.pdf', dpi=300)

        plt.figure()
        f, axes = plt.subplots(1, 1, constrained_layout = True, figsize=(10, 6))
        plt2 = sns.lineplot(x="iteration", y="data",
                            hue="method",data=df1, ax=axes, palette=palette, ci=68)
        plt2.legend_.remove()
        plt2.set(xlabel='')
        
        axes.set_title('10-Class Classification Accuracy', fontweight="bold", size=29)
        axes.set_ylabel('')
        plt.savefig('Experiments/' + setting_name + imit_dim + '_accuracy.pdf', dpi=300)


    elif setting_name in irl_settings: #to be modified to add more settings
        f, axes = plt.subplots(1, 1, constrained_layout = True, figsize=(10, 6))

        df1 = pd.read_csv(imit_path + '%s.csv' % ('reward_dist_'+setting_name+'_imit'))
        df2 = pd.read_csv(imit_path + '%s.csv' % ('action_dist_'+setting_name+'_imit'))

        plt1 = sns.lineplot(x="iteration", y="data",
                            hue="method", data=df1, ax=axes, palette=palette, ci=68)

        plt1.legend_.remove()
        plt1.set(ylabel='')
        axes.set_xlabel('Training Iteration', fontweight="bold", size=29)
        axes.set_title('L2 Distance', fontweight="bold", size=29)
        plt.savefig('Experiments/' + setting_name + '_l2.pdf', dpi=300)

        plt.figure()
        f, axes = plt.subplots(1, 1, constrained_layout = True, figsize=(10, 6))
        plt2 = sns.lineplot(x="iteration", y="data",
                            hue="method", data=df2, ax=axes, palette=palette, ci=68)

        axes.set_title('Total Policy Variance', fontweight="bold", size=29)
        plt2.legend_.remove()
        plt2.set(ylabel='')
        axes.set_xlabel('Training Iteration', fontweight="bold", size=29)
        plt.savefig('Experiments/' + setting_name + '_accuracy.pdf', dpi=300)

def plot_supp(setting_name, imit_dim=None):
    classification = {'class10_coop', 'class10_adv'}
    irl_settings = {'irlH_coop', 'irlH_adv', 'irlE_coop', 'irlE_adv'}

    paper_rc = {'lines.linewidth': 2.5}
    sns.set(style="darkgrid")
    sns.set(font_scale=1.95, rc = paper_rc)

    directory = 'Experiments/'

    omni_path = directory + setting_name + '_omni/'
    imit_path = directory + setting_name + '_imit/'

    palette = {"ITAL":sns.xkcd_rgb["red"],"ITAL 2":sns.xkcd_rgb["bright yellow"], "ITAL 5":sns.xkcd_rgb["golden yellow"],  "ITAL 10":sns.xkcd_rgb["orange"],  "ITAL 15":sns.xkcd_rgb["burnt orange"], \
               "Dim-20 ITAL":sns.xkcd_rgb["red"],"Dim-20 ITAL 2":sns.xkcd_rgb["bright yellow"], "Dim-20 ITAL 5":sns.xkcd_rgb["golden yellow"],  "Dim-20 ITAL 10":sns.xkcd_rgb["orange"],  "Dim-20 ITAL 15":sns.xkcd_rgb["burnt orange"], \
            "Dim-30 ITAL":sns.xkcd_rgb["red"],"Dim-30 ITAL 2":sns.xkcd_rgb["bright yellow"], "Dim-30 ITAL 5":sns.xkcd_rgb["golden yellow"],  "Dim-30 ITAL 10":sns.xkcd_rgb["orange"],  "Dim-30 ITAL 15":sns.xkcd_rgb["burnt orange"], \
               "Batch":sns.xkcd_rgb["blue"], "SGD":sns.xkcd_rgb["purple"], \
               'IMT': sns.xkcd_rgb['green'], 'Imitate Dim-20 IMT': sns.xkcd_rgb['dark green'], 'Imitate Dim-30 IMT': sns.xkcd_rgb['olive green'],\
               "Imitate CNN-9 ITAL":sns.xkcd_rgb["burnt orange"], "Imitate CNN-12 ITAL":sns.xkcd_rgb["orange"], \
               'Omniscient IMT': sns.xkcd_rgb['green'], 'Imitate CNN-9 IMT': sns.xkcd_rgb['dark green'], 'Imitate CNN-12 IMT': sns.xkcd_rgb['olive green'],\
               "Imitate ITAL":sns.xkcd_rgb["burnt orange"], 'Imitate IMT': sns.xkcd_rgb['dark green'], \
               'Imitate Dim-50 ITAL':sns.xkcd_rgb["orange"], 'Imitate Dim-40 ITAL':sns.xkcd_rgb["burnt orange"], \
               'Imitate Dim-50 IMT': sns.xkcd_rgb['olive green'], 'Imitate Dim-40 IMT': sns.xkcd_rgb['dark green'], 'Teacher Rewards': sns.xkcd_rgb["grey"]}


    display_methods = [ 'Batch', 'SGD', 'IMT', 'ITAL', 'ITAL 2', 'ITAL 5', 'ITAL 10', 'ITAL 15']
    dash = {"Omniscient ITAL": '',"Imitate ITAL": '',"Batch":'', "SGD": '', \
            'Omniscient IMT': '', 'Imitate IMT': '', 'Teacher Rewards': (5, 5)}    
    plt.figure()
    f, axes = plt.subplots(1, 1, constrained_layout = True, figsize=(10, 6))

    if setting_name in classification:
        df1 = pd.read_csv(imit_path + '%s.csv' % ('losses_'+setting_name+'_imit'))

        plt1 = sns.lineplot(x="iteration", y="data",
                 hue="method", data=df1, ax=axes, palette=palette, ci=68)

        axes.set_title('Cross Entropy Loss', fontweight="bold", size=29)

        plt1.legend_.remove()
        plt1.set(xlabel='', ylabel='')
        plt.savefig('Experiments/' + setting_name + '_crossEntropy.pdf', dpi=300)

    elif setting_name == 'mnist_coop' or setting_name == 'mnist_adv' :
        df0 = pd.read_csv(imit_path[:-1] + '_' + imit_dim + '/' + '%s.csv' % ('losses'+ '_'+setting_name+'_imit_' + imit_dim ))
        
        plt1 = sns.lineplot(x="iteration", y="data",
                            hue="method",data=df0, ax=axes, palette=palette, ci=68)
        plt1.legend_.remove()
        
        axes.set_title('Cross Entropy Loss', fontweight="bold", size=29)
        axes.set_ylabel('')
        axes.set_xlabel('')
        plt.savefig('Experiments/' + setting_name + imit_dim + '_crossEntropy.pdf', dpi=300)

    elif setting_name == 'cifar_coop' or setting_name == 'cifar_adv':
        df0 = pd.read_csv(imit_path[:-1] + '_' + imit_dim + '/' + '%s.csv' % ('losses'+ '_'+setting_name+'_imit_' + imit_dim ))
        
        plt1 = sns.lineplot(x="iteration", y="data",
                            hue="method",data=df0, ax=axes, palette=palette, ci=68)
        plt1.legend_.remove()
        
        axes.set_title('Cross Entropy Loss', fontweight="bold", size=29)
        axes.set_ylabel('')
        axes.set_xlabel('Training Iteration', fontweight="bold", size=29)
        plt.savefig('Experiments/' + setting_name + imit_dim + '_crossEntropy.pdf', dpi=300)
        
    elif setting_name == 'equation_coop' or setting_name == 'equation_adv':
        df0 = pd.read_csv('Experiments/%s.csv' % ('search_acc_'+setting_name))
        plt1 = sns.lineplot(x="iteration", y="data",
                 hue="method",data=df0, ax=axes, palette=palette)
        # axes.axhline(np.load(omni_path + 'gt_search_acc.npy'), color=sns.xkcd_rgb['grey'], linestyle='-')
        # plt1.lines[8].set_linestyle('dashed')
        plt1.legend_.remove()

        axes.set_title('Simplification Accuracy', fontweight="bold", size=29)
        axes.set_xlabel('Training Iteration', fontweight="bold", size=29)
        axes.set_ylabel('')

    elif setting_name in irl_settings:
        df0 = pd.read_csv(imit_path + '%s.csv' % ('rewards_'+setting_name+'_imit'))

        teacher_rewards = pd.read_csv(imit_path + "%s.csv" % ('teacher_rewards_'+setting_name+'_imit'))
        teacher_rewards['method'] = "Teacher Rewards"

        df0 = pd.concat([df0, teacher_rewards])
        plt1 = sns.lineplot(x="iteration", y="data",
                 hue="method", data=df0, ax=axes, palette=palette, ci=68)

        plt1.legend_.remove()
        plt1.set(ylabel='')

        plt1.lines[9-1].set_linestyle("dashed")
        # plt1.set_xticks([0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100])
        # plt1.set_xticklabels([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
        axes.set_title('Actual Rewards', fontweight="bold", size=29)
        axes.set_xlabel('Training Iteration', fontweight="bold", size=29)
        plt.savefig('Experiments/' + setting_name + '_rewards.pdf', dpi=300)


def remove_npy(dir):
    print(dir)
    return
    for f in glob.glob(dir + "/*.npy"):
        os.remove(f)

def CollectDataAndPlot(setting_name, seed_range):
    type_ = 'irl'
    random_seeds = [j for j in range(seed_range)]
    irl_settings = {'irlH_coop', 'irlH_adv', 'irlE_coop', 'irlE_adv'}

    if setting_name not in irl_settings:
        type_ = 'c'
    if setting_name == 'regression_coop' or setting_name == 'regression_adv'\
        or setting_name == 'class10_coop' or setting_name == 'class10_adv':
        #omni_setting = setting_name + '_omni'
        imit_setting = setting_name + '_imit'
        arguments = ['python3', 'main.py', imit_setting]
        collect_data(imit_setting, 'imit', random_seeds, arguments, type_)
        remove_npy('Experiments/' + imit_setting)

        plot(setting_name)
        plot_supp(setting_name)
    elif setting_name == 'equation_coop' or setting_name == 'equation_adv':
 
        # omni_setting = setting_name + '_omni'
        imit_setting1 = setting_name + '_imit_40'
        imit_setting2 = setting_name + '_imit_50'

        arguments = ['python3', 'main.py', imit_setting1]
        collect_data(imit_setting1, 'imit', random_seeds, arguments, type_)
        remove_npy('Experiments/' + imit_setting1)

        arguments = ['python3', 'main.py', imit_setting2]
        collect_data(imit_setting1, 'imit', random_seeds, type_)
        remove_npy('Experiments/' + imit_setting2)
        

        for imit_dim in ['40', '50']:
            plot(setting_name, imit_dim)
            plot_supp(setting_name,imit_dim)        
        from Equation import search
        search.main()
        remove_npy('Experiments')

    elif setting_name == 'mnist_coop' or setting_name == 'mnist_adv':
        #omni_setting = setting_name + '_omni'
        imit_setting1 = setting_name + '_imit_20'
        imit_setting2 = setting_name + '_imit_30'

        arguments = ['python3', 'main.py', imit_setting1]
        collect_data(imit_setting1, 'imit', random_seeds, arguments, type_)
        remove_npy('Experiments/' + imit_setting1)
        
        arguments = ['python3', 'main.py', imit_setting2]
        collect_data(imit_setting2, 'imit', random_seeds, arguments, type_)
        remove_npy('Experiments/' + imit_setting2)
        
        for imit_dim in ['20', '30']:
            plot(setting_name, imit_dim)
            plot_supp(setting_name,imit_dim)

    elif setting_name == 'cifar_coop' or setting_name == 'cifar_adv':
        # omni_setting = setting_name + '_omni'
        imit_setting1 = setting_name + '_imit_9'
        imit_setting2 = setting_name + '_imit_12'

        arguments = ['python3', 'main.py', imit_setting1]
        collect_data(imit_setting1, 'imit', random_seeds, arguments, type_)
        remove_npy('Experiments/' + imit_setting1)

        arguments = ['python3', 'main.py', imit_setting2]
        collect_data(imit_setting2, 'imit', random_seeds, arguments, type_)
        remove_npy('Experiments/' + imit_setting2)
        
        for imit_dim in ['9', '12']:
            plot(setting_name, imit_dim)
            plot_supp(setting_name,imit_dim)

    elif setting_name in irl_settings:
        #omni_setting = setting_name + '_omni'
        imit_setting = setting_name + '_imit'

        arguments = ['python3', 'main_irl.py', imit_setting]
        collect_data(imit_setting, 'imit', random_seeds, arguments, type_)
        remove_npy('Experiments/' + imit_setting)
        
        plot(setting_name)
        plot_supp(setting_name)
    else:
        print('Invalid setting')
        return


def main():
    parser = argparse.ArgumentParser(description='plotband.py requires a "setting_name" as described in README')
    parser.add_argument('-s', '--setting_name', required=True, dest='setting_name',
                        help='plotband.py requires a "setting_name" as described in README.')
    args = parser.parse_args()

    all_settings = ['mnist', 'cifar', 'equation', 'regression', 'class10', 'irlH', 'irlE']
    teachers = ['coop', 'adv']
    argList = args.setting_name.split('_')
    if len(argList) != 2 or argList[0] not in all_settings or argList[1] not in teachers:
        print('--Invalid setting')
        exit()

    CollectDataAndPlot(args.setting_name, seed_range = 20)
if __name__ == '__main__':
    main()
