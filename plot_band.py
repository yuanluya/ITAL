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

def get_path(data_cate, arguments, type_ = 'd'):
    #add dist or dist_ or ... as argyment
    #note data_cate in different main functions are different
    titles = []
    methods = edict()
    
    if type_ != 'irl':
        lines = ['1', '8', 'batch', 'sgd']
    else:
        lines = ['0', '1', '2', '3']
        
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

def save_csv(data_cate, setting_name, mode, random_seeds, arguments, type_ = 'd'):
    methods_code = {'0': 'No Rep', '1': 'IMT', '2': 'Rand Rep', '3': 'prag', '4': 'Prag (Strt Lin)', '5': 'IMT (Strt Lin)', \
                    '7': 'cont_sgd', '8': 'ITAL', '6': 'Expert', 'batch': 'Batch', 'sgd': 'SGD'} 
    if type_ == 'irl':
        methods_code = {'0': 'IMT', '1': 'ITAL', '2': 'Batch', '3': 'SGD'} 
    
    titles, methods = get_path(data_cate, arguments, type_)
    data = []
    method = []
    iterations = []
    for t in titles:
        for s in random_seeds:
            filename = setting_name + '/' + t + '_' + str(s) + '.npy'
            d = np.load(filename, allow_pickle = True)
            #d = d[:int(len(d)/4)]
            length = len(d)
            print(d)
            data.append(d)
            method += [methods_code[methods[t]] for j in range(length)]
            iterations += [j for j in range(length)]
    data = np.array(data).flatten()

    #save_name = setting_name + '_csv/' + mode + '_' + setting_name + '_csv/' + data_cate + '_' + mode + '_' + setting_name + '.csv'
    save_name = setting_name + '/' + data_cate + '_' + mode + '_' + setting_name + '.csv'

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
            if 'regression' in  arguments:
                arguments_ = arguments[0:-1] + [str(s)] + [arguments[-1]] 
            else:
                arguments_ = arguments[:] + [str(s)]

            p = subprocess.Popen(arguments_)
            child_processes.append(p)
        for cp in child_processes:
            cp.wait()
        #subprocess.call(arguments_)
        #subprocess.call(arguments.append(str(s)))    
        #p.wait()
    
    if type_ == 'd' or type_ == 'c':
        save_csv('dist', setting_name, mode, random_seeds, arguments, type_)
        save_csv('dist_', setting_name, mode, random_seeds, arguments, type_)
        save_csv('losses', setting_name, mode, random_seeds, arguments, type_)
        save_csv('accuracies', setting_name, mode, random_seeds, arguments, type_)
    else:
        save_csv('action_dist', setting_name, mode, random_seeds, arguments, type_)
        save_csv('reward_dist', setting_name, mode, random_seeds, arguments, type_)
        save_csv('q_dist', setting_name, mode, random_seeds, arguments, type_)
        save_csv('rewards', setting_name, mode, random_seeds, arguments, type_)
    print('collected data\n')


def plot(setting_name):
    main_multi_settings = {'regression_coop', 'regression_adv', 'class10_coop', 'class10_adv'}
    classification = {'class10_coop', 'class10_adv', 'mnist'}
    irl_settings = {'irlH_coop', 'irlH_adv', 'irlE_coop', 'irlE_adv'}

    paper_rc = {'lines.linewidth': 2.5}
    sns.set(style="darkgrid")
    sns.set(font_scale=1.95, rc = paper_rc)
    
    directory = setting_name + '/'
    #omni_path = directory + 'omni_' + directory
    #imit_path = directory + 'imit_' + directory
    omni_path = directory 
    imit_path = directory 

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
        results0_omni = pd.read_csv(omni_path + '%s.csv' % ('dist'+'_omni_'+setting_name))
        results1_omni = pd.read_csv(omni_path + '%s.csv' % ('losses'+'_omni_'+setting_name))

        results0_imit = pd.read_csv(imit_path + '%s.csv' % ('dist'+'_imit_'+setting_name))
        results1_imit = pd.read_csv(imit_path + '%s.csv' % ('losses'+'_imit_'+setting_name))

        df1 = results0_omni.loc[results0_omni['method'] == display_methods[0]]
        df2 = results1_omni.loc[results1_omni['method'] == display_methods[0]]

        sgd1 = results0_omni.loc[results0_omni['method'] == display_methods[1]]
        sgd2 = results1_omni.loc[results1_omni['method'] == display_methods[1]]

        df1 = pd.concat([df1, sgd1])
        df2 = pd.concat([df2, sgd2])

        if setting_name in classification:
            results2_omni = pd.read_csv(omni_path + '%s.csv' % ('accuracies'+'_omni_'+setting_name))
            results2_imit = pd.read_csv(imit_path + '%s.csv' % ('accuracies'+'_imit_'+setting_name))
            df0 = results2_omni.loc[results2_omni['method'] == display_methods[0]]

            sgd0 = results2_omni.loc[results2_omni['method'] == display_methods[1]]
            df0 = pd.concat([df0, sgd0])

        for method in display_methods[2:]:
            df1_omni = results0_omni.loc[results0_omni['method'] == method]
            df2_omni = results1_omni.loc[results1_omni['method'] == method]

            df1_imit = results0_imit.loc[results0_imit['method'] == method]
            df2_imit = results1_imit.loc[results1_imit['method'] == method]
            
            if setting_name in classification:
                df0_omni = results2_omni.loc[results2_omni['method'] == method]
                df0_imit = results2_imit.loc[results2_imit['method'] == method]

            df1_omni['method'] = 'Omniscient ' + method
            df2_omni['method'] = 'Omniscient ' + method
            df1_imit['method'] = 'Imitate ' + method
            df2_imit['method'] = 'Imitate ' + method
            if setting_name in classification:
                df0_omni['method'] = 'Omniscient ' + method
                df0_imit['method'] = 'Imitate ' + method

            df1 = pd.concat([df1, df1_omni])
            df2 = pd.concat([df2, df2_omni])

            df1 = pd.concat([df1, df1_imit])
            df2 = pd.concat([df2, df2_imit])

            if setting_name in classification:
                df0 = pd.concat([df0, df0_omni])
                df0 = pd.concat([df0, df0_imit])

        plt1 = sns.lineplot(x="iteration", y="data",
                 hue="method", data=df1, ax=axes[0], palette=palette)

        if setting_name not in classification:
            plt2 = sns.lineplot(x="iteration", y="data",
                     hue="method", data=df2, ax=axes[1], palette=palette)
        else:
            plt2 = sns.lineplot(x="iteration", y="data",
                     hue="method", data=df0, ax=axes[1], palette=palette)
      
        
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
        results0_omni = pd.read_csv(omni_path + '%s.csv' % ('dist'+'_omni_'+setting_name))
        results1_omni = pd.read_csv(omni_path + '%s.csv' % ('losses'+'_omni_'+setting_name))

        print(results0_omni)
        imit_dim = ['50','40']
        #imit_path = {dim : directory + 'imit' + dim +'_' + directory for dim in imit_dim}
        results0_imit = {}
        results1_imit = {}
        results2_imit = {}
        for dim in imit_dim:      
            results0_imit[dim] = pd.read_csv(imit_path + '%s.csv' % ('dist'+'_imit' + dim + '_'+setting_name))
            results1_imit[dim] = pd.read_csv(imit_path + '%s.csv' % ('losses'+'_imit' + dim + '_'+setting_name))

        df0 = results0_omni.loc[results0_omni['method'] == display_methods[0]]
        df1 = results1_omni.loc[results1_omni['method'] == display_methods[0]]
        print(df0)
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

            if method == 'cont_prag':
                method = 'ITAL'

            df0_omni['method'] = 'Omniscient ' + method
            df1_omni['method'] = 'Omniscient ' + method

            df0 = pd.concat([df0, df0_omni])
            df1 = pd.concat([df1, df1_omni])
            
            for dim in imit_dim:
                df0_imit[dim]['method'] = 'Imitate Dim-' + dim + ' ' + method
                df1_imit[dim]['method'] = 'Imitate Dim-' + dim + ' ' + method

                df0 = pd.concat([df0, df0_imit[dim]])
                df1 = pd.concat([df1, df1_imit[dim]])
        print(df0)
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
        results0_omni = pd.read_csv(omni_path + '%s.csv' % ('dist'+'_omni_'+setting_name))
        results1_omni = pd.read_csv(omni_path + '%s.csv' % ('accuracies'+'_omni_'+setting_name))

        imit_dim = ['30','20']
        results0_imit = {}
        results1_imit = {}
        results2_imit = {}
        for dim in imit_dim:      
            results0_imit[dim] = pd.read_csv(imit_path + '%s.csv' % ('dist'+'_imit' + dim + '_'+setting_name))
            results1_imit[dim] = pd.read_csv(imit_path + '%s.csv' % ('accuracies'+'_imit' + dim + '_'+setting_name))

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
        results0_omni = pd.read_csv(omni_path + '%s.csv' % ('dists'+'_omni_'+setting_name))
        results1_omni = pd.read_csv(omni_path + '%s.csv' % ('accuracies'+'_omni_'+setting_name))

        imit_dim = ['9','12']
        results0_imit = {}
        results1_imit = {}
        results2_imit = {}
        for dim in imit_dim:      
            results0_imit[dim] = pd.read_csv(imit_path + '%s.csv' % ('dists'+'_imit' + dim + '_'+setting_name))
            results1_imit[dim] = pd.read_csv(imit_path + '%s.csv' % ('accuracies'+'_imit' + dim + '_'+setting_name))

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

        results0_omni = pd.read_csv(omni_path + '%s.csv' % ('reward_dist'+'_omni_'+setting_name))
        results1_omni = pd.read_csv(omni_path + '%s.csv' % ('action_dist'+'_omni_'+setting_name))

        results0_imit = pd.read_csv(imit_path + '%s.csv' % ('reward_dist'+'_imit_'+setting_name))
        results1_imit = pd.read_csv(imit_path + '%s.csv' % ('action_dist'+'_imit_'+setting_name))

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

    plt.savefig(setting_name + '/' + setting_name + '-main.pdf', dpi=300)

def plot_supp(setting_name):
    classification = {'class10_coop', 'class10_adv'}
    irl_settings = {'irlH_coop', 'irlH_adv', 'irlE_coop', 'irlE_adv'}

    paper_rc = {'lines.linewidth': 2.5}
    sns.set(style="darkgrid")
    sns.set(font_scale=1.95, rc = paper_rc)
    
    directory = setting_name + '/'
    #omni_path = directory + 'omni_' + directory
    #imit_path = directory + 'imit_' + directory

    omni_path = directory 
    imit_path = directory 

    palette = {"Omniscient ITAL":sns.xkcd_rgb["red"],"Imitate Dim-20 ITAL":sns.xkcd_rgb["burnt orange"], "Imitate Dim-30 ITAL":sns.xkcd_rgb["orange"], \
                "Batch":sns.xkcd_rgb["blue"], "SGD":sns.xkcd_rgb["purple"], \
                'Omniscient IMT': sns.xkcd_rgb['green'], 'Imitate Dim-20 IMT': sns.xkcd_rgb['dark green'], 'Imitate Dim-30 IMT': sns.xkcd_rgb['olive green'],\
                "Imitate CNN-9 ITAL":sns.xkcd_rgb["burnt orange"], "Imitate CNN-12 ITAL":sns.xkcd_rgb["orange"], \
                'Omniscient IMT': sns.xkcd_rgb['green'], 'Imitate CNN-9 IMT': sns.xkcd_rgb['dark green'], 'Imitate CNN-12 IMT': sns.xkcd_rgb['olive green'],\
                "Imitate ITAL":sns.xkcd_rgb["orange"], 'Imitate IMT': sns.xkcd_rgb['dark green'], \
                'Imitate Dim-50 ITAL':sns.xkcd_rgb["orange"], 'Imitate Dim-40 ITAL':sns.xkcd_rgb["burnt orange"], \
                'Imitate Dim-50 IMT': sns.xkcd_rgb['olive green'], 'Imitate Dim-40 IMT': sns.xkcd_rgb['dark green']}

    display_methods = [ 'Batch', 'SGD', 'IMT', 'ITAL']

    plt.figure()     
    f, axes = plt.subplots(1, 1, constrained_layout = True, figsize=(10, 6)) 

    if setting_name in classification:
        results0_omni = pd.read_csv(omni_path + '%s.csv' % ('losses'+'_omni_'+setting_name))
        results0_imit = pd.read_csv(imit_path + '%s.csv' % ('losses'+'_imit_'+setting_name))
  
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
        results1_omni = pd.read_csv(omni_path + '%s.csv' % ('losses'+'_omni_'+setting_name))

        imit_dim = ['30','20']
        results0_imit = {}
        results1_imit = {}
        results2_imit = {}
        for dim in imit_dim:      
            results1_imit[dim] = pd.read_csv(imit_path + '%s.csv' % ('losses'+'_imit' + dim + '_'+setting_name))

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
        results1_omni = pd.read_csv(omni_path + '%s.csv' % ('losses'+'_omni_'+setting_name))

        imit_dim = ['9', '12']
        results0_imit = {}
        results1_imit = {}
        results2_imit = {}
        for dim in imit_dim:      
            results1_imit[dim] = pd.read_csv(imit_path + '%s.csv' % ('losses'+'_imit' + dim + '_'+setting_name))

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

    plt.savefig(setting_name + '/' + setting_name + '-supp.pdf', dpi=300)

def remove_npy(dir):
    for f in glob.glob(dir + "/*.npy"):
        os.remove(f)
            #os.remove(os.path.join(dir, f))

def CollectDataAndPlot(setting_name):
    type_ = 'irl'
    random_seeds = [j for j in range(1)]
    irl_settings = {'irlH_coop', 'irlH_adv', 'irlE_coop', 'irlE_adv'}

    if setting_name not in irl_settings:
        type_ = 'c'  
    if setting_name == 'regression_coop' or setting_name == 'regression_adv':
        arguments = ['python3', 'main_multi.py', setting_name, 'omni', '-1']
        collect_data(setting_name, 'omni', random_seeds, arguments, type_)
        remove_npy(setting_name)
        arguments = ['python3', 'main_multi.py', setting_name, 'imit', '-1']
        collect_data(setting_name, 'imit', random_seeds, arguments, type_)
        remove_npy(setting_name)

    elif setting_name == 'equation_coop' or setting_name == 'equation_adv':
        arguments = ['python3', 'main_multi.py', setting_name, 'omni', '-1']
        collect_data(setting_name, 'omni', random_seeds, arguments, type_)
        remove_npy(setting_name)
        arguments = ['python3', 'main_multi.py', setting_name, 'imit', '40']
        collect_data(setting_name, 'imit40', random_seeds, arguments, type_)
        remove_npy(setting_name)
        arguments = ['python3', 'main_multi.py', setting_name, 'imit', '50']
        collect_data(setting_name, 'imit50', random_seeds, arguments, type_)
        remove_npy(setting_name)

    elif setting_name == 'mnist_coop' or setting_name == 'mnist_adv':
        arguments = ['python3', 'main_multi.py', setting_name, 'omni', '-1']
        collect_data(setting_name, 'omni', random_seeds, arguments, type_)
        remove_npy(setting_name)
        arguments = ['python3', 'main_multi.py', setting_name, 'imit', '20']
        collect_data(setting_name, 'imit20', random_seeds, arguments, type_)
        remove_npy(setting_name)
        arguments = ['python3', 'main_multi.py', setting_name, 'imit', '30']
        collect_data(setting_name, 'imit30', random_seeds, arguments, type_)
        remove_npy(setting_name)

    elif setting_name == 'class10_coop' or setting_name == 'class10_adv':
        arguments = ['python3', 'main_multi.py', setting_name, 'omni', '-1']
        collect_data(setting_name, 'omni', random_seeds, arguments, type_)
        remove_npy(setting_name)
        arguments = ['python3', 'main_multi.py', setting_name, 'imit', '-1']
        collect_data(setting_name, 'imit', random_seeds, arguments, type_)
        remove_npy(setting_name)
    
    elif setting_name == 'cifar_coop' or setting_name == 'cifar_adv':
        arguments = ['python3', 'main_multi.py', setting_name, 'omni', '-1']
        collect_data(setting_name, 'omni', random_seeds, arguments, type_)
        remove_npy(setting_name)
        arguments = ['python3', 'main_multi.py', setting_name, 'imit', '9']
        collect_data(setting_name, 'imit9', random_seeds, arguments, type_)
        remove_npy(setting_name)
        arguments = ['python3', 'main_multi.py', setting_name, 'imit', '12']
        collect_data(setting_name, 'imit12', random_seeds, arguments, type_)
        remove_npy(setting_name)

    elif setting_name in irl_settings:
        arguments = ['python3', 'main_irl.py', setting_name, 'omni']
        collect_data(setting_name, 'omni', random_seeds, arguments, type_)
        remove_npy(setting_name)
        arguments = ['python3', 'main_irl.py', setting_name, 'imit']
        collect_data(setting_name, 'imit', random_seeds, arguments, type_)
        remove_npy(setting_name)

    else:
        print('Invalid setting')
        return
    
    plot(setting_name)
    plot_supp(setting_name)


def main():
    if len(sys.argv) != 2:
        print('--Invalid arguments; use python3 plotband.py data "setting_name" to collect data; use python3 plotband.py plot "setting_name" to get plots;')
        print('\tuse python3 plotband.py plot "setting_name" use get supplementary plots')
        exit()

    setting_name = sys.argv[1]
    CollectDataAndPlot(setting_name)
    '''
    type_ = 'irl'
    random_seeds = [j for j in range(2)]
    setting_name = sys.argv[2]
    irl_settings = {'imit_peak_8', 'imit_random_8', 'imit_peak_10', 'imit_random_10', 'omni_peak_8', 'omni_random_8'}
    
    if setting_name not in irl_settings:
        type_ = 'c'
    if sys.argv[1] == 'data':
        
        if setting_name == 'omni_equation':
            arguments = ['python3', 'main_multi.py', '45', '0', '0', '0.05', '1000', '0.001', '0.01', 'regression']
        elif setting_name == 'imit_equation':
            arguments = ['python3', 'main_multi.py', '45', '2', '0', '0.05', '1000', '0.001', '0.01', 'regression'] 
        elif setting_name == 'omni_class10':
            arguments = ['python3', 'main_multi.py', '30', '0', '0', '0.1', '1000', '0', '0.1']
        elif setting_name == 'imit_class10':
            arguments = ['python3', 'main_multi.py', '30', '2', '0.01', '0.1', '1000', '0.01', '0.1']
        elif setting_name == 'omni_class4':
            arguments = ['python3', 'main_multi.py', '50', '0', '0.01', '0.1', '200', '0.01', '0.1']
        elif setting_name == 'imit_class4':
            arguments = ['python3', 'main_multi.py', '50', '2', '0.01', '0.1', '200', '0.01', '0.1']
        elif setting_name == 'omni_regression':
            arguments = ['python3', 'main_multi.py', '100', '0', '0.1', '0.3', '300', '0.001', '0.05', 'regression']
        elif setting_name == 'imit_regression':
            arguments = ['python3', 'main_multi.py', '100', '2', '0.1', '0.3', '200', '0', '0.05', 'regression']
        elif setting_name == 'omni_mnist':
            arguments = ['python3', 'main_multi.py', '24', '0', '0.01', '0.1', '200', '0', '0.05',]
        elif setting_name == 'imit_mnist':
            arguments = ['python3', 'main_multi.py', '24', '2', '0.02', '0.1', '1000', '0', '0.05']
        elif setting_name == 'imit_peak_8':
            arguments = ['python3', 'main_irl.py', '1', 'E', '8', '0', '0.2', '70', '1', '200']
        elif setting_name == 'imit_random_8':
            arguments = ['python3', 'main_irl.py', '1', 'H', '8', '0.005', '0.3', '200', '5', '220']
        elif setting_name == 'omni_peak_8':                         
            arguments = ['python3', 'main_irl.py', '0', 'E', '8', '0', '0.3', '300', '1', '200']          
        elif setting_name == 'omni_random_8':
            arguments = ['python3', 'main_irl.py', '0', 'H', '8', '0', '0.3', '200', '5', '220']
        else:
            print('possible setting_names are omni_equation, imit_equation, omni_class10, imit_class10, ')
            print('omni_class4, imit_class4, omni_regression, imit_regression, omni_mnist, imit_mnist')
            exit()
        

    elif sys.argv[1] == 'plot':
        plot(setting_name)

    elif sys.argv[1] == 'plot_supp':
        plot_supp(setting_name)
    else:
        print('--Invalid arguments; use python3 plotband.py data "setting_name" to collect data; use python3 plotband.py plot "setting_name" to get plots')
        print('\tuse python3 plotband.py plot "setting_name" use get supplementary plots')
        print()
    '''
if __name__ == '__main__':
    main()
