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
                    '7': 'cont_sgd', '8': 'ITAL', '6': 'Expert', 'batch': 'Batch', 'sgd': 'SGD'}
    if type_ == 'irl':
        methods_code = {'0': 'IMT', '1': 'ITAL', '2': 'Batch', '3': 'SGD', '': ''}

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
            print(d)
            data.append(d)
            method += [methods_code[methods[t]] for j in range(length)]
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
    #child_processes = []

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
        #p.wait()

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
        if mode == 'omni':
            save_csv('teacher_rewards', setting_name, mode, random_seeds, type_)
    print('collected data\n')


def plot(setting_name):
    main_multi_settings = {'regression_coop', 'regression_adv', 'class10_coop', 'class10_adv'}
    classification = {'class10_coop', 'class10_adv', 'mnist'}
    irl_settings = {'irlH_coop', 'irlH_adv', 'irlE_coop', 'irlE_adv'}

    paper_rc = {'lines.linewidth': 2.5}
    sns.set(style="darkgrid")
    sns.set(font_scale=1.95, rc = paper_rc)

    directory = 'Experiments/'

    omni_path = directory + setting_name + '_omni/'
    imit_path = directory + setting_name + '_imit/'

    palette = {"Omniscient ITAL":sns.xkcd_rgb["red"],"Imitate Dim-20 ITAL":sns.xkcd_rgb["burnt orange"], "Imitate Dim-30 ITAL":sns.xkcd_rgb["orange"], \
                "Batch":sns.xkcd_rgb["blue"], "SGD":sns.xkcd_rgb["purple"], \
                'Omniscient IMT': sns.xkcd_rgb['green'], 'Imitate Dim-20 IMT': sns.xkcd_rgb['dark green'], 'Imitate Dim-30 IMT': sns.xkcd_rgb['olive green'],\
                "Imitate CNN-9 ITAL":sns.xkcd_rgb["burnt orange"], "Imitate CNN-12 ITAL":sns.xkcd_rgb["orange"], \
                'Omniscient IMT': sns.xkcd_rgb['green'], 'Imitate CNN-9 IMT': sns.xkcd_rgb['dark green'], 'Imitate CNN-12 IMT': sns.xkcd_rgb['olive green'],\
                "Imitate ITAL":sns.xkcd_rgb["orange"], 'Imitate IMT': sns.xkcd_rgb['dark green'], \
                'Imitate Dim-50 ITAL':sns.xkcd_rgb["orange"], 'Imitate Dim-40 ITAL':sns.xkcd_rgb["burnt orange"], \
                'Imitate Dim-50 IMT': sns.xkcd_rgb['olive green'], 'Imitate Dim-40 IMT': sns.xkcd_rgb['dark green']}

    dash = {"Omniscient ITAL": '',"Imitate ITAL": (5, 5),"Batch":'', "SGD": '', \
            'Omniscient IMT': '', 'Imitate IMT': (5, 5)}

    display_methods = [ 'Batch', 'SGD', 'IMT', 'ITAL']

    plt.figure()
    f, axes = plt.subplots(1, 2, constrained_layout = True, figsize=(20, 6))

    if setting_name in main_multi_settings:
        results0_omni = pd.read_csv(omni_path + '%s.csv' % ('dist_'+setting_name+'_omni'))
        results0_imit = pd.read_csv(imit_path + '%s.csv' % ('dist_'+setting_name+'_imit'))

        if setting_name in classification:
            results1_omni = pd.read_csv(omni_path + '%s.csv' % ('accuracies_'+setting_name+'_omni'))
            results1_imit = pd.read_csv(imit_path + '%s.csv' % ('accuracies_'+setting_name+'_imit'))
        else:
            results1_omni = pd.read_csv(omni_path + '%s.csv' % ('losses_'+setting_name+'_omni'))
            results1_imit = pd.read_csv(imit_path + '%s.csv' % ('losses_'+setting_name+'_imit'))

        df1 = results0_omni.loc[results0_omni['method'] == display_methods[0]]
        df2 = results1_omni.loc[results1_omni['method'] == display_methods[0]]

        sgd1 = results0_omni.loc[results0_omni['method'] == display_methods[1]]
        sgd2 = results1_omni.loc[results1_omni['method'] == display_methods[1]]

        df1 = pd.concat([df1, sgd1])
        df2 = pd.concat([df2, sgd2])

        for method in display_methods[2:]:
            df1_omni = results0_omni.loc[results0_omni['method'] == method]
            df2_omni = results1_omni.loc[results1_omni['method'] == method]

            df1_imit = results0_imit.loc[results0_imit['method'] == method]
            df2_imit = results1_imit.loc[results1_imit['method'] == method]

            df1_omni['method'] = 'Omniscient ' + method
            df2_omni['method'] = 'Omniscient ' + method
            df1_imit['method'] = 'Imitate ' + method
            df2_imit['method'] = 'Imitate ' + method

            df1 = pd.concat([df1, df1_omni])
            df2 = pd.concat([df2, df2_omni])

            df1 = pd.concat([df1, df1_imit])
            df2 = pd.concat([df2, df2_imit])

        plt1 = sns.lineplot(x="iteration", y="data",
                 hue="method", data=df1, ax=axes[0], palette=palette)

        plt2 = sns.lineplot(x="iteration", y="data",
                hue="method", data=df2, ax=axes[1], palette=palette)


        axes[1].set_title('Square Loss', fontweight="bold", size=29)
        if setting_name in classification:
            if setting_name == 'class4':
                axes[1].set_title('4-Class Classification Accuracy', fontweight="bold", size=29)
            else:
                axes[1].set_title('10-Class Classification Accuracy', fontweight="bold", size=29)

        plt1.legend_.remove()
        plt2.legend_.remove()
        plt1.set(xlabel='', ylabel='')
        plt2.set(xlabel='', ylabel='')
        axes[0].set_title('L2 Distance', fontweight="bold", size=29)

    elif setting_name == 'equation_coop' or setting_name == 'equation_adv':
        results0_omni = pd.read_csv(omni_path + '%s.csv' % ('dist_'+setting_name+'_omni'))
        results1_omni = pd.read_csv(omni_path + '%s.csv' % ('losses_'+setting_name+'_omni'))

        print(results0_omni)
        imit_dim = ['50','40']
        #imit_path = {dim : directory + 'imit' + dim +'_' + directory for dim in imit_dim}
        results0_imit = {}
        results1_imit = {}
        results2_imit = {}
        for dim in imit_dim:
            results0_imit[dim] = pd.read_csv(imit_path[:-1] + '_' + dim + '/'  + '%s.csv' % ('dist' + '_'+setting_name+'_imit_' + dim))
            results1_imit[dim] = pd.read_csv(imit_path[:-1] + '_' + dim + '/'  + '%s.csv' % ('losses'+'_'+setting_name+'_imit_' + dim))

        df0 = results0_omni.loc[results0_omni['method'] == display_methods[0]]
        df1 = results1_omni.loc[results1_omni['method'] == display_methods[0]]

        sgd0 = results0_omni.loc[results0_omni['method'] == display_methods[1]]
        sgd1 = results1_omni.loc[results1_omni['method'] == display_methods[1]]

        df0 = pd.concat([df0, sgd0])
        df1 = pd.concat([df1, sgd1])

        for method in display_methods[2:]:
            df0_omni = results0_omni.loc[results0_omni['method'] == method]
            df1_omni = results1_omni.loc[results1_omni['method'] == method]

            df0_imit = {dim : results0_imit[dim].loc[results0_imit[dim]['method'] == method] for dim in imit_dim}
            df1_imit = {dim : results1_imit[dim].loc[results1_imit[dim]['method'] == method] for dim in imit_dim}

            df0_omni['method'] = 'Omniscient ' + method
            df1_omni['method'] = 'Omniscient ' + method

            df0 = pd.concat([df0, df0_omni])
            df1 = pd.concat([df1, df1_omni])

            for dim in imit_dim:
                df0_imit[dim]['method'] = 'Imitate Dim-' + dim + ' ' + method
                df1_imit[dim]['method'] = 'Imitate Dim-' + dim + ' ' + method

                df0 = pd.concat([df0, df0_imit[dim]])
                df1 = pd.concat([df1, df1_imit[dim]])

        plt1 = sns.lineplot(x="iteration", y="data",
                 hue="method",data=df0, ax=axes[0], palette=palette)
        plt1.legend_.remove()
        plt2 = sns.lineplot(x="iteration", y="data",
                 hue="method",data=df1, ax=axes[1], palette=palette)
        plt2.legend_.remove()

        plt1.set(ylabel='')
        plt2.set(ylabel='')
        axes[0].set_title('L2 Distance', fontweight="bold", size=29)
        axes[1].set_title('Square Loss', fontweight="bold", size=29)
        axes[0].set_xlabel('Training Iteration', fontweight="bold", size=29)
        axes[1].set_xlabel('Training Iteration', fontweight="bold", size=29)

    elif setting_name == 'mnist_coop' or setting_name == 'mnist_adv' :
        results0_omni = pd.read_csv(omni_path + '%s.csv' % ('dist_'+setting_name+'_omni'))
        results1_omni = pd.read_csv(omni_path + '%s.csv' % ('accuracies_'+setting_name+'_omni'))

        imit_dim = ['30','20']
        results0_imit = {}
        results1_imit = {}
        results2_imit = {}
        for dim in imit_dim:
            results0_imit[dim] = pd.read_csv(imit_path[:-1] + '_' + dim + '/'  + '%s.csv' % ('dist' + '_'+setting_name+'_imit_' + dim))
            results1_imit[dim] = pd.read_csv(imit_path[:-1] + '_' + dim + '/'  + '%s.csv' % ('accuracies'+ '_'+setting_name+'_imit_' + dim ))

        df0 = results0_omni.loc[results0_omni['method'] == display_methods[0]]
        df1 = results1_omni.loc[results1_omni['method'] == display_methods[0]]

        df0['method'] = 'Batch'
        df1['method'] = 'Batch'

        sgd0 = results0_omni.loc[results0_omni['method'] == display_methods[1]]
        sgd1 = results1_omni.loc[results1_omni['method'] == display_methods[1]]

        sgd0['method'] = 'SGD'
        sgd1['method'] = 'SGD'

        df0 = pd.concat([df0, sgd0])
        df1 = pd.concat([df1, sgd1])

        for method in display_methods[2:]:
            df0_omni = results0_omni.loc[results0_omni['method'] == method]
            df1_omni = results1_omni.loc[results1_omni['method'] == method]

            df0_imit = {dim : results0_imit[dim].loc[results0_imit[dim]['method'] == method] for dim in imit_dim}
            df1_imit = {dim : results1_imit[dim].loc[results1_imit[dim]['method'] == method] for dim in imit_dim}

            df0_omni['method'] = 'Omniscient ' + method
            df1_omni['method'] = 'Omniscient ' + method

            df0 = pd.concat([df0, df0_omni])
            df1 = pd.concat([df1, df1_omni])

            for dim in imit_dim:
                df0_imit[dim]['method'] = 'Imitate Dim-' + dim + ' ' + method
                df1_imit[dim]['method'] = 'Imitate Dim-' + dim + ' ' + method

                df0 = pd.concat([df0, df0_imit[dim]])
                df1 = pd.concat([df1, df1_imit[dim]])

        plt1 = sns.lineplot(x="iteration", y="data",
                 hue="method",data=df0, ax=axes[0], palette=palette)
        plt1.legend_.remove()
        plt2 = sns.lineplot(x="iteration", y="data",
                 hue="method",data=df1, ax=axes[1], palette=palette)
        plt2.legend_.remove()

        plt1.set(xlabel='')
        plt2.set(xlabel='')
        axes[0].set_title('L2 Distance', fontweight="bold", size=29)
        axes[1].set_title('10-Class Classification Accuracy', fontweight="bold", size=29)
        axes[0].set_ylabel('')
        axes[1].set_ylabel('')


    elif setting_name == 'cifar_coop' or setting_name == 'cifar_adv': #to be modified to add more settings
        results0_omni = pd.read_csv(omni_path + '%s.csv' % ('dist_'+setting_name+'_omni'))
        results1_omni = pd.read_csv(omni_path + '%s.csv' % ('accuracies_'+setting_name+'_omni'))

        imit_dim = ['9','12']
        results0_imit = {}
        results1_imit = {}
        results2_imit = {}
        for dim in imit_dim:
            results0_imit[dim] = pd.read_csv(imit_path[:-1] + '_' + dim + '/'  + '%s.csv' % ('dist' + '_'+setting_name+'_imit_' + dim))
            results1_imit[dim] = pd.read_csv(imit_path[:-1] + '_' + dim + '/'  + '%s.csv' % ('accuracies'+'_imit' + '_'+setting_name+'_imit_' + dim))

        df0 = results0_omni.loc[results0_omni['method'] == display_methods[0]]
        df1 = results1_omni.loc[results1_omni['method'] == display_methods[0]]

        sgd0 = results0_omni.loc[results0_omni['method'] == display_methods[1]]
        sgd1 = results1_omni.loc[results1_omni['method'] == display_methods[1]]

        df0 = pd.concat([df0, sgd0])
        df1 = pd.concat([df1, sgd1])

        for method in display_methods[2:]:
            df0_omni = results0_omni.loc[results0_omni['method'] == method]
            df1_omni = results1_omni.loc[results1_omni['method'] == method]

            df0_imit = {dim : results0_imit[dim].loc[results0_imit[dim]['method'] == method] for dim in imit_dim}
            df1_imit = {dim : results1_imit[dim].loc[results1_imit[dim]['method'] == method] for dim in imit_dim}

            df0_omni['method'] = 'Omniscient ' + method
            df1_omni['method'] = 'Omniscient ' + method

            df0 = pd.concat([df0, df0_omni])
            df1 = pd.concat([df1, df1_omni])

            for dim in imit_dim:
                df0_imit[dim]['method'] = 'Imitate CNN-' + dim + ' ' + method
                df1_imit[dim]['method'] = 'Imitate CNN-' + dim + ' ' + method

                df0 = pd.concat([df0, df0_imit[dim]])
                df1 = pd.concat([df1, df1_imit[dim]])

        plt1 = sns.lineplot(x="iteration", y="data",
                 hue="method",data=df0, ax=axes[0], palette=palette)
        plt1.legend_.remove()
        plt2 = sns.lineplot(x="iteration", y="data",
                 hue="method",data=df1, ax=axes[1], palette=palette)
        plt2.legend_.remove()

        plt2.set(xlabel='')
        axes[0].set_title('L2 Distance', fontweight="bold", size=29)
        axes[1].set_title('10-Class Classification Accuracy', fontweight="bold", size=29)
        axes[0].set_ylabel('')
        axes[1].set_ylabel('')

    elif setting_name in irl_settings: #to be modified to add more settings

        results0_omni = pd.read_csv(omni_path + '%s.csv' % ('reward_dist_'+setting_name+'_omni'))
        results1_omni = pd.read_csv(omni_path + '%s.csv' % ('action_dist_'+setting_name+'_omni'))

        results0_imit = pd.read_csv(imit_path + '%s.csv' % ('reward_dist_'+setting_name+'_imit'))
        results1_imit = pd.read_csv(imit_path + '%s.csv' % ('action_dist_'+setting_name+'_imit'))

        df0 = results0_omni.loc[results0_omni['method'] == display_methods[0]]
        df1 = results1_omni.loc[results1_omni['method'] == display_methods[0]]

        sgd0 = results0_omni.loc[results0_omni['method'] == display_methods[1]]
        sgd1 = results1_omni.loc[results1_omni['method'] == display_methods[1]]

        df0 = pd.concat([df0, sgd0])
        df1 = pd.concat([df1, sgd1])

        for method in display_methods[2:]:
            df0_omni = results0_omni.loc[results0_omni['method'] == method]
            df1_omni = results1_omni.loc[results1_omni['method'] == method]

            df0_imit = results0_imit.loc[results0_imit['method'] == method]
            df1_imit = results1_imit.loc[results1_imit['method'] == method]

            df0_omni['method'] = 'Omniscient ' + method
            df1_omni['method'] = 'Omniscient ' + method

            df0 = pd.concat([df0, df0_omni])
            df1 = pd.concat([df1, df1_omni])

            df0_imit['method'] = 'Imitate' + ' ' + method
            df1_imit['method'] = 'Imitate' + ' ' + method

            df0 = pd.concat([df0, df0_imit])
            df1 = pd.concat([df1, df1_imit])

        plt1 = sns.lineplot(x="iteration", y="data",
                 hue="method", data=df0, ax=axes[0], palette=palette)

        plt2 = sns.lineplot(x="iteration", y="data",
                 hue="method", data=df1, ax=axes[1], palette=palette)

        plt1.legend_.remove()
        plt2.legend_.remove()
        plt1.set(ylabel='')
        plt2.set(ylabel='')
        axes[0].set_title('L2 Distance', fontweight="bold", size=29)
        axes[1].set_title('Total Policy Variance', fontweight="bold", size=29)
        axes[0].set_xlabel('Training Iteration', fontweight="bold", size=29)
        axes[1].set_xlabel('Training Iteration', fontweight="bold", size=29)
    plt.savefig('Experiments/' + setting_name + '-main.pdf', dpi=300)
    #plt.savefig('Experiments/' + setting_name + '/' + setting_name + '-main.pdf', dpi=300)

def plot_supp(setting_name):
    classification = {'class10_coop', 'class10_adv'}
    irl_settings = {'irlH_coop', 'irlH_adv', 'irlE_coop', 'irlE_adv'}

    paper_rc = {'lines.linewidth': 2.5}
    sns.set(style="darkgrid")
    sns.set(font_scale=1.95, rc = paper_rc)

    directory = 'Experiments/'

    omni_path = directory + setting_name + '_omni/'
    imit_path = directory + setting_name + '_imit/'

    palette = {"Omniscient ITAL":sns.xkcd_rgb["red"],"Imitate Dim-20 ITAL":sns.xkcd_rgb["burnt orange"], "Imitate Dim-30 ITAL":sns.xkcd_rgb["orange"], \
                "Batch":sns.xkcd_rgb["blue"], "SGD":sns.xkcd_rgb["purple"], \
                'Omniscient IMT': sns.xkcd_rgb['green'], 'Imitate Dim-20 IMT': sns.xkcd_rgb['dark green'], 'Imitate Dim-30 IMT': sns.xkcd_rgb['olive green'],\
                "Imitate CNN-9 ITAL":sns.xkcd_rgb["burnt orange"], "Imitate CNN-12 ITAL":sns.xkcd_rgb["orange"], \
                'Omniscient IMT': sns.xkcd_rgb['green'], 'Imitate CNN-9 IMT': sns.xkcd_rgb['dark green'], 'Imitate CNN-12 IMT': sns.xkcd_rgb['olive green'],\
                "Imitate ITAL":sns.xkcd_rgb["orange"], 'Imitate IMT': sns.xkcd_rgb['dark green'], \
                'Imitate Dim-50 ITAL':sns.xkcd_rgb["orange"], 'Imitate Dim-40 ITAL':sns.xkcd_rgb["burnt orange"], \
                'Imitate Dim-50 IMT': sns.xkcd_rgb['olive green'], 'Imitate Dim-40 IMT': sns.xkcd_rgb['dark green'],
                'Teacher Rewards': sns.xkcd_rgb['grey']}

    display_methods = [ 'Batch', 'SGD', 'IMT', 'ITAL']
    dash = {"Omniscient ITAL": '',"Imitate ITAL": '',"Batch":'', "SGD": '', \
            'Omniscient IMT': '', 'Imitate IMT': '', 'Teacher Rewards': (5, 5)}    
    plt.figure()
    f, axes = plt.subplots(1, 1, constrained_layout = True, figsize=(10, 6))

    if setting_name in classification:
        results0_omni = pd.read_csv(omni_path + '%s.csv' % ('losses_'+setting_name+'_omni'))
        results0_imit = pd.read_csv(imit_path + '%s.csv' % ('losses_'+setting_name+'_imit'))

        df1 = results0_omni.loc[results0_omni['method'] == display_methods[0]]

        sgd1 = results0_omni.loc[results0_omni['method'] == display_methods[1]]

        for method in display_methods[2:]:
            df1_omni = results0_omni.loc[results0_omni['method'] == method]
            df1_imit = results0_imit.loc[results0_imit['method'] == method]

            df1_omni['method'] = 'Omniscient ' + method
            df1_imit['method'] = 'Imitate ' + method

            df1 = pd.concat([df1, df1_omni])
            df1 = pd.concat([df1, df1_imit])


        plt1 = sns.lineplot(x="iteration", y="data",
                 hue="method", data=df1, ax=axes, palette=palette)

        axes.set_title('Cross Entropy Loss', fontweight="bold", size=29)

        plt1.legend_.remove()
        plt1.set(xlabel='Training Iteration', ylabel='')

    elif setting_name == 'mnist_coop' or setting_name == 'mnist_adv' :
        results1_omni = pd.read_csv(omni_path + '%s.csv' % ('losses_'+setting_name+'_omni'))

        imit_dim = ['30','20']

        results1_imit = {}
        for dim in imit_dim:
            results1_imit[dim] = pd.read_csv(imit_path[:-1] + '_' + dim + '/' + '%s.csv' % ('losses'+ '_'+setting_name+'_imit_' + dim ))

        df1 = results1_omni.loc[results1_omni['method'] == display_methods[0]]

        sgd1 = results1_omni.loc[results1_omni['method'] == display_methods[1]]

        df1 = pd.concat([df1, sgd1])

        for method in display_methods[2:]:
            df1_omni = results1_omni.loc[results1_omni['method'] == method]

            df1_imit = {dim : results1_imit[dim].loc[results1_imit[dim]['method'] == method] for dim in imit_dim}

            df1_omni['method'] = 'Omniscient ' + method
            df1 = pd.concat([df1, df1_omni])

            for dim in imit_dim:
                df1_imit[dim]['method'] = 'Imitate Dim-' + dim + ' ' + method

                df1 = pd.concat([df1, df1_imit[dim]])

        plt1 = sns.lineplot(x="iteration", y="data",
                 hue="method",data=df1, ax=axes, palette=palette)
        plt1.legend_.remove()

        plt1.set(xlabel='Training Iteration')
        axes.set_title('Cross Entropy Loss')
        axes.set_ylabel('')

    elif setting_name == 'cifar_coop' or setting_name == 'cifar_adv':
        results1_omni = pd.read_csv(omni_path + '%s.csv' % ('losses_'+setting_name+'_omni'))

        imit_dim = ['9', '12']
        results1_imit = {}
        for dim in imit_dim:
            results1_imit[dim] = pd.read_csv(imit_path[:-1] + '_' + dim + '/' + '%s.csv' % ('losses'+ '_'+setting_name+'_imit_' + dim ))

        df1 = results1_omni.loc[results1_omni['method'] == display_methods[0]]

        sgd1 = results1_omni.loc[results1_omni['method'] == display_methods[1]]
        df1 = pd.concat([df1, sgd1])

        for method in display_methods[2:]:
            df1_omni = results1_omni.loc[results1_omni['method'] == method]

            df1_imit = {dim : results1_imit[dim].loc[results1_imit[dim]['method'] == method] for dim in imit_dim}

            df1_omni['method'] = 'Omniscient ' + method
            df1 = pd.concat([df1, df1_omni])

            for dim in imit_dim:
                df1_imit[dim]['method'] = 'Imitate Dim-' + dim + ' ' + method

                df1 = pd.concat([df1, df1_imit[dim]])

        plt1 = sns.lineplot(x="iteration", y="data",
                 hue="method",data=df1, ax=axes, palette=palette)
        plt1.legend_.remove()

        plt1.set(xlabel='Training Iteration')
        axes.set_title('Cross Entropy Loss')
        axes.set_ylabel('')
        
    elif setting_name == 'equation_coop' or setting_name == 'equation_adv':
        results0_omni = pd.read_csv(omni_path + '%s.csv' % ('search_acc_'+setting_name+'_omni'))

        imit_dim = ['50','40']
        
        results0_imit = {}
        for dim in imit_dim:
            results0_imit[dim] = pd.read_csv(imit_path[:-1] + '_' + dim + '/'  + '%s.csv' % ('search_acc_' + '_'+setting_name+'_imit_' + dim))

        df0 = results0_omni.loc[results0_omni['method'] == display_methods[0]]

        sgd0 = results0_omni.loc[results0_omni['method'] == display_methods[1]]

        df0 = pd.concat([df0, sgd0])

        for method in display_methods[2:]:
            df0_omni = results0_omni.loc[results0_omni['method'] == method]

            df0_imit = {dim : results0_imit[dim].loc[results0_imit[dim]['method'] == method] for dim in imit_dim}

            df0_omni['method'] = 'Omniscient ' + method

            df0 = pd.concat([df0, df0_omni])

            for dim in imit_dim:
                df0_imit[dim]['method'] = 'Imitate Dim-' + dim + ' ' + method

                df0 = pd.concat([df0, df0_imit[dim]])

        plt1 = sns.lineplot(x="iteration", y="data",
                 hue="method",data=df0, ax=axes[0], palette=palette)
        # axes.axhline(np.load(omni_path + 'gt_search_acc.npy'), color=sns.xkcd_rgb['grey'], linestyle='-')
        # plt1.lines[8].set_linestyle('dashed')
        plt1.legend_.remove()

        axes.set_title('Simplification Accuracy', fontweight="bold", size=29)
        axes.set_xlabel('Training Iteration', fontweight="bold", size=29)
        axes.set_ylabel('')

    elif setting_name in irl_settings:
        results0_omni = pd.read_csv(omni_path + '%s.csv' % ('rewards_'+setting_name+'_omni'))
        results0_imit = pd.read_csv(imit_path + '%s.csv' % ('rewards_'+setting_name+'_imit'))

        teacher_rewards = pd.read_csv(omni_path + "%s.csv" % ('teacher_rewards_'+setting_name+'_omni'))
        teacher_rewards['method'] = "Teacher Rewards"

        df0 = results0_omni.loc[results0_omni['method'] == display_methods[0]]
        sgd0 = results0_omni.loc[results0_omni['method'] == display_methods[1]]
        df0 = pd.concat([df0, sgd0])

        for method in display_methods[2:]:
            df0_omni = results0_omni.loc[results0_omni['method'] == method]

            df0_imit = results0_imit.loc[results0_imit['method'] == method]

            df0_omni['method'] = 'Omniscient ' + method

            df0 = pd.concat([df0, df0_omni])

            df0_imit['method'] = 'Imitate' + ' ' + method
            df0 = pd.concat([df0, df0_imit])

        df0 = pd.concat([df0, teacher_rewards])
        plt1 = sns.lineplot(x="iteration", y="data",
                 hue="method", data=df0, ax=axes, palette=palette)

        plt1.legend_.remove()
        plt1.set(ylabel='')
        plt1.lines[6].set_linestyle("dashed")
        # plt1.set_xticks([0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100])
        # plt1.set_xticklabels([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
        axes.set_title('Actual Rewards', fontweight="bold", size=29)
        axes.set_xlabel('Training Iteration', fontweight="bold", size=29)

    #plt.savefig('Experiments/' + setting_name + '/' + setting_name + '-supp.pdf', dpi=300)

    plt.savefig('Experiments/' + setting_name + '-supp.pdf', dpi=300)

def remove_npy(dir):
    print(dir)
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
        omni_setting = setting_name + '_omni'
        imit_setting = setting_name + '_imit'

        arguments = ['python3', 'main_multi.py', omni_setting]
        collect_data(omni_setting, 'omni', random_seeds, arguments, type_)
        remove_npy('Experiments/' + omni_setting)

        arguments = ['python3', 'main_multi.py', imit_setting]
        collect_data(imit_setting, 'imit', random_seeds, arguments, type_)
        remove_npy('Experiments/' + imit_setting)

    elif setting_name == 'equation_coop' or setting_name == 'equation_adv':
        omni_setting = setting_name + '_omni'
        imit_setting1 = setting_name + '_imit_40'
        imit_setting2 = setting_name + '_imit_50'

        arguments = ['python3', 'main_multi.py', omni_setting]
        collect_data(omni_setting, 'omni', random_seeds, arguments, type_)
        remove_npy('Experiments/' + omni_setting)

        arguments = ['python3', 'main_multi.py', imit_setting1]
        collect_data(imit_setting1, 'imit', random_seeds, arguments, type_)
        remove_npy('Experiments/' + imit_setting1)

        arguments = ['python3', 'main_multi.py', imit_setting2]
        collect_data(imit_setting2, 'imit', random_seeds, arguments, type_)
        remove_npy('Experiments/' + imit_setting2)

    elif setting_name == 'mnist_coop' or setting_name == 'mnist_adv':
        omni_setting = setting_name + '_omni'
        imit_setting1 = setting_name + '_imit_20'
        imit_setting2 = setting_name + '_imit_30'

        arguments = ['python3', 'main_multi.py', omni_setting]
        collect_data(omni_setting, 'omni', random_seeds, arguments, type_)
        remove_npy('Experiments/' + omni_setting)

        arguments = ['python3', 'main_multi.py', imit_setting1]
        collect_data(imit_setting1, 'imit', random_seeds, arguments, type_)
        remove_npy('Experiments/' + imit_setting1)

        arguments = ['python3', 'main_multi.py', imit_setting2]
        collect_data(imit_setting2, 'imit', random_seeds, arguments, type_)
        remove_npy('Experiments/' + imit_setting2)

    elif setting_name == 'cifar_coop' or setting_name == 'cifar_adv':
        omni_setting = setting_name + '_omni'
        imit_setting1 = setting_name + '_imit_9'
        imit_setting2 = setting_name + '_imit_12'

        arguments = ['python3', 'main_multi.py', omni_setting]
        collect_data(omni_setting, 'omni', random_seeds, arguments, type_)
        remove_npy('Experiments/' + omni_setting)

        arguments = ['python3', 'main_multi.py', imit_setting1]
        collect_data(imit_setting1, 'imit', random_seeds, arguments, type_)
        remove_npy('Experiments/' + imit_setting1)

        arguments = ['python3', 'main_multi.py', imit_setting2]
        collect_data(imit_setting2, 'imit', random_seeds, arguments, type_)
        remove_npy('Experiments/' + imit_setting2)

    elif setting_name in irl_settings:
        omni_setting = setting_name + '_omni'
        imit_setting = setting_name + '_imit'

        arguments = ['python3', 'main_irl.py', omni_setting]
        collect_data(omni_setting, 'omni', random_seeds, arguments, type_)
        remove_npy('Experiments/' + omni_setting)

        arguments = ['python3', 'main_irl.py', imit_setting]
        collect_data(imit_setting, 'imit', random_seeds, arguments, type_)
        remove_npy('Experiments/' + imit_setting)

    else:
        print('Invalid setting')
        return

    plot(setting_name)
    plot_supp(setting_name)


def main():
    parser = argparse.ArgumentParser(description='plotband.py requires a "setting_name" as described in README')
    parser.add_argument('-s', '--setting_name', required=True, dest='setting_name',
                        help='plotband.py requires a "setting_name" as described in README.')
    args = parser.parse_args()

    all_settings = ['mnist', 'cifar', 'equation', 'regression', 'class10', 'irlH', 'irlE']
    load_data = ['mnist', 'cifar', 'equation']

    argList = args.setting_name.split('_')
    if argList[0] not in all_settings:
        print('--Invalid setting')
        exit()

    CollectDataAndPlot(args.setting_name, 1)

if __name__ == '__main__':
    main()
