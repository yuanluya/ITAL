import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from easydict import EasyDict as edict
import multiprocessing
import time

def get_path(data_cate, arguments, type_ = 'd'):
    #add dist or dist_ or ... as argyment
    #note data_cate in different main functions are different
    titles = []
    methods = edict()
    title = '_'
    
    if type_ == 'd':
        lines = ['0', '1', '2', '3', '4', '5', '6', 'batch', 'sgd']
        
        mode_idx = int(arguments[3])
        modes = ['omni', 'surr', 'imit']
        mode = modes[mode_idx]
        title += mode
        title += '_'
        task = 'classification' if 'regression' not in  arguments else 'regression'
        title += task
        title += '_'

        dd = int(arguments[2])
        num_classes = 10 if dd == 24 or dd == 30 else 4
        if task == 'regression':
            num_classes = 1
        title += 'num_classes'
        title += '_'
        title += str(num_classes)
        title += '_'
        if dd == 48:
            title += 'equation'
        elif dd == 24:
            title += 'mnist'
        else:
            title += 'gaussian'
    
    elif type_ == 'c':
        lines = ['1', '7', '8', 'batch', 'sgd']
        
        mode_idx = int(arguments[3])
        modes = ['omni', 'surr', 'imit']
        mode = modes[mode_idx]
        title += mode
        title += '_'
        task = 'classification' if 'regression' not in  arguments else 'regression'
        title += task
        title += '_'

        dd = int(arguments[2])
        num_classes = 10 if dd == 24 or dd == 30 else 4
        if task == 'regression':
            num_classes = 1
        title += 'num_classes'
        title += '_'
        title += str(num_classes)
        title += '_'
        if dd == 48:
            title += 'equation'
        elif dd == 24:
            title += 'mnist'
        else:
            title += 'gaussian'

    else:
        lines = ['0', '1', '2', '3']

        title = '_'
        mode_idx = int(arguments[2])
        modes = ['omni',  'imit']
        mode = modes[mode_idx]
        title += mode
        title += '_'
        
        title += arguments[4]
        title += '_'
        title += 'beta'
        title += '_'
        title += arguments[8]
        
    for l in lines:
            '''
            titles.append('dist'+l+title)
            titles.append('dist'+l+'_'+title)
            titles.append('logpdfs'+l+title)
            titles.append('accuracies'+l+title)
            '''
            if data_cate == 'dist_':
                t = 'dist'+l+'_'+title
            else:
                t = data_cate+l+title
            titles.append(t)
            methods[t] = l 

    return titles, methods

def save_csv(data_cate, setting_name, random_seeds, arguments, type_ = 'd'):
    if type_ == 'd' or type_ == 'c':
        methods_code = {'0': 'No Rep', '1': 'IMT', '2': 'Rand Rep', '3': 'prag', '4': 'Prag (Strt Lin)', '5': 'IMT (Strt Lin)', \
                    '7': 'cont_sgd', '8': 'cont_prag', '6': 'Expert', 'batch': 'batch', 'sgd': 'sgd'} 
    else:
        methods_code = {'0': 'zero', '1': 'one', '2': 'random', '3': 'pragmatic'}
    titles, methods = get_path(data_cate, arguments, type_)
    data = []
    method = []
    iterations = []
    for t in titles:
        for s in random_seeds:
            filename = t + '_' + str(s) + '.npy'
            d = np.load(filename, allow_pickle = True)
            #d = d[:int(len(d)/4)]
            length = len(d)
            data.append(d)
            method += [methods_code[methods[t]] for j in range(length)]
            iterations += [j for j in range(length)]
    data = np.array(data).flatten()

    save_name = setting_name + '_csv/' + data_cate + '_' + setting_name + '.csv'
    df = pd.DataFrame({'method': method,
                       'iteration': iterations,
                       'data': data})
    df.to_csv(save_name)
    print('saved file to %s \n' % save_name)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def collect_data(setting_name, random_seeds, arguments, type_):
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
        save_csv('dist', setting_name, random_seeds, arguments, type_)
        save_csv('dist_', setting_name, random_seeds, arguments, type_)
        save_csv('losses', setting_name, random_seeds, arguments, type_)
        save_csv('accuracies', setting_name, random_seeds, arguments, type_)
    else:
        save_csv('dists', setting_name, random_seeds, arguments, type_)
        save_csv('dist_', setting_name, random_seeds, arguments, type_)
        save_csv('distsq', setting_name, random_seeds, arguments, type_)
        save_csv('ar', setting_name, random_seeds, arguments, type_)
    print('collected data\n')


def plot(setting_name):
    main_multi_settings = {'regression', 'class4', 'class10', 'mnist', 'equation'}

    paper_rc = {'lines.linewidth': 2.5}
    sns.set(style="darkgrid")
    sns.set(font_scale=1.9, rc = paper_rc)
    
    directory = setting_name + '_csv/'
    omni_path = directory + 'omni_' + directory
    imit_path = directory + 'imit_' + directory

    if setting_name in main_multi_settings:
        results0_omni = pd.read_csv(omni_path + '%s.csv' % ('dist'+'_omni_'+setting_name))
        results1_omni = pd.read_csv(omni_path + '%s.csv' % ('accuracies'+'_omni_'+setting_name))

        results0_imit = pd.read_csv(imit_path + '%s.csv' % ('dist'+'_imit_'+setting_name))
        results1_imit = pd.read_csv(imit_path + '%s.csv' % ('accuracies'+'_imit_'+setting_name))

        display_methods = [ 'batch', 'sgd', 'IMT', 'cont_prag']

        df1 = results0_omni.loc[results0_omni['method'] == display_methods[0]]
        df2 = results1_omni.loc[results1_omni['method'] == display_methods[0]]
        df1['method'] = 'Batch' 
        df2['method'] = 'Batch'

        sgd1 = results0_omni.loc[results0_omni['method'] == display_methods[1]]
        sgd2 = results1_omni.loc[results1_omni['method'] == display_methods[1]]
        sgd1['method'] = 'SGD'
        sgd2['method'] = 'SGD'

        df1 = pd.concat([df1, sgd1])
        df2 = pd.concat([df2, sgd2])

        for method in display_methods[2:]:
            df1_omni = results0_omni.loc[results0_omni['method'] == method]
            df2_omni = results1_omni.loc[results1_omni['method'] == method]

            df1_imit = results0_imit.loc[results0_imit['method'] == method]
            df2_imit = results1_imit.loc[results1_imit['method'] == method]
            
            if method == 'cont_prag':
                method = 'ITAL'
            df1_omni['method'] = 'Omniscient ' + method
            df2_omni['method'] = 'Omniscient ' + method
            df1_imit['method'] = 'Imitate ' + method
            df2_imit['method'] = 'Imitate ' + method

            df1 = pd.concat([df1, df1_omni])
            df2 = pd.concat([df2, df2_omni])

            df1 = pd.concat([df1, df1_imit])
            df2 = pd.concat([df2, df2_imit])
        

        plt.figure() 
        f, axes = plt.subplots(1, 2, constrained_layout = True, figsize=(30, 10))   

        palette ={"Omniscient ITAL":sns.xkcd_rgb["red"],"Imitate ITAL":sns.xkcd_rgb["red"],"Batch":sns.xkcd_rgb["blue"], "SGD":sns.xkcd_rgb["green"], \
                      'Omniscient IMT': sns.xkcd_rgb['dusty purple'], 'Imitate IMT': sns.xkcd_rgb['dusty purple']}

        dash = {"Omniscient ITAL": '',"Imitate ITAL": (5, 5),"Batch":'', "SGD": '', \
                      'Omniscient IMT': '', 'Imitate IMT': (5, 5)}

        plt1 = sns.lineplot(x="iteration", y="data",
                 hue="method", style = 'method',data=df1, ax=axes[0], palette=palette, dashes = dash)

        plt2 = sns.lineplot(x="iteration", y="data",
                 hue="method", style = 'method',data=df2, ax=axes[1], palette=palette, dashes = dash)

        plt1.set(xlabel='training iteration', ylabel='')
        plt2.set(xlabel='training iteration', ylabel='')
        axes[0].set_title('L2 DISTANCE')
        axes[1].set_title('CROSS ENTROPY LOSS')

    elif setting_name == 'cifar': #to be modified to add more settings 
        results0_omni = pd.read_csv(omni_path + '%s.csv' % ('dists'+'_omni_'+setting_name))
        results1_omni = pd.read_csv(omni_path + '%s.csv' % ('accuracies'+'_omni_'+setting_name))
        results2_omni = pd.read_csv(omni_path + '%s.csv' % ('losses'+'_omni_'+setting_name))

        imit_dim = ['6','9','12']
        imit_path = {dim : directory + 'imit' + dim +'_' + directory for dim in imit_dim}
        results0_imit = {}
        results1_imit = {}
        results2_imit = {}
        for dim in imit_dim:      
            results0_imit[dim] = pd.read_csv(imit_path[dim] + '%s.csv' % ('dists'+'_imit' + dim + '_'+setting_name))
            results1_imit[dim] = pd.read_csv(imit_path[dim] + '%s.csv' % ('accuracies'+'_imit' + dim + '_'+setting_name))
            results2_imit[dim] = pd.read_csv(imit_path[dim] + '%s.csv' % ('losses'+'_imit' + dim + '_'+setting_name))

        display_methods = [ 'batch', 'sgd', 'IMT', 'cont_prag']

        df0 = results0_omni.loc[results0_omni['method'] == display_methods[0]]
        df1 = results1_omni.loc[results1_omni['method'] == display_methods[0]]
        df2 = results2_omni.loc[results2_omni['method'] == display_methods[0]]
        
        df0['method'] = 'Batch'
        df1['method'] = 'Batch'
        df2['method'] = 'Batch'
        
        sgd0 = results0_omni.loc[results0_omni['method'] == display_methods[1]]
        sgd1 = results1_omni.loc[results1_omni['method'] == display_methods[1]]
        sgd2 = results2_omni.loc[results2_omni['method'] == display_methods[1]]

        sgd0['method'] = 'SGD'
        sgd1['method'] = 'SGD'
        sgd2['method'] = 'SGD'
        
        df0 = pd.concat([df0, sgd0])
        df1 = pd.concat([df1, sgd1])
        df2 = pd.concat([df2, sgd2])

        for method in display_methods[2:]:
            df0_omni = results0_omni.loc[results0_omni['method'] == method]
            df1_omni = results1_omni.loc[results1_omni['method'] == method]
            df2_omni = results2_omni.loc[results2_omni['method'] == method]

            df0_imit = {dim : results0_imit[dim].loc[results0_imit[dim]['method'] == method] for dim in imit_dim}
            df1_imit = {dim : results1_imit[dim].loc[results1_imit[dim]['method'] == method] for dim in imit_dim}
            df2_imit = {dim : results2_imit[dim].loc[results2_imit[dim]['method'] == method] for dim in imit_dim}

            if method == 'cont_prag':
                method = 'ITAL'

            df0_omni['method'] = 'Omniscient ' + method
            df1_omni['method'] = 'Omniscient ' + method
            df2_omni['method'] = 'Omniscient ' + method

            df0 = pd.concat([df0, df0_omni])
            df1 = pd.concat([df1, df1_omni])
            df2 = pd.concat([df2, df2_omni])
            
            for dim in imit_dim:
                df0_imit[dim]['method'] = 'Imitate CNN-' + dim + ' ' + method
                df1_imit[dim]['method'] = 'Imitate CNN-' + dim + ' ' + method
                df2_imit[dim]['method'] = 'Imitate CNN-' + dim + ' ' + method

                df0 = pd.concat([df0, df0_imit[dim]])
                df1 = pd.concat([df1, df1_imit[dim]])
                df2 = pd.concat([df2, df2_imit[dim]])
        
        plt.figure() 
        f, axes = plt.subplots(1, 3, constrained_layout = True, figsize=(30, 10))   

        palette ={"Omniscient ITAL":sns.xkcd_rgb["red"],"Imitate CNN-6 ITAL":sns.xkcd_rgb["orange"], "Imitate CNN-9 ITAL":sns.xkcd_rgb["burnt orange"], \
                    "Imitate CNN-12 ITAL":sns.xkcd_rgb["brick red"], "Batch":sns.xkcd_rgb["blue"], "SGD":sns.xkcd_rgb["purple"], \
                      'Omniscient IMT': sns.xkcd_rgb['green'], 'Imitate CNN-6 IMT': sns.xkcd_rgb['olive green'], 'Imitate CNN-9 IMT': sns.xkcd_rgb['dark green'],\
                      'Imitate CNN-12 IMT': sns.xkcd_rgb['mustard yellow']}
        
        plt1 = sns.lineplot(x="iteration", y="data",
                 hue="method",data=df0, ax=axes[0], palette=palette)
        plt1.legend_.remove()
        plt2 = sns.lineplot(x="iteration", y="data",
                 hue="method",data=df1, ax=axes[1], palette=palette)
        plt2.legend_.remove()
        plt3 = sns.lineplot(x="iteration", y="data",
                 hue="method",data=df2, ax=axes[2], palette=palette)
        
        plt1.set(xlabel='training iteration')
        plt2.set(xlabel='training iteration')
        plt3.set(xlabel='training iteration')
        axes[0].set_title('L2 DISTANCE')
        axes[1].set_title('10-CLASS CLASSIFICATION ACCURACY')
        axes[2].set_title('CROSS ENTROPY LOSS')
        axes[0].set_ylabel('')
        axes[1].set_ylabel('')
        axes[2].set_ylabel('')

    plt.savefig(setting_name + '.pdf', dpi=300)
    plt.show()

def main():
    if len(sys.argv) != 3:
        print('--Invalid arguments; use python3 plotband.py data "setting_name" to collect data; use python3 plotband.py plot "setting_name" to get plots')
        exit()

    type_ = 'd'
    random_seeds = [j for j in range(20)]
    setting_name = sys.argv[2]
    irl_settings = {'imit_peak_8', 'imit_random_8', 'imit_peak_10', 'imit_random_10', 'omni_peak_8', 'omni_random_8'}
    irl = True
    if setting_name not in irl_settings:
        irl = False
        type_ = 'c'
    if sys.argv[1] == 'data':
        if setting_name == 'omni_equation':
            arguments = ['python3', 'main_multi.py', '48', '0', '0', '0.05', '1000', '0.001', '0.01', 'regression']
        elif setting_name == 'imit_equation':
            arguments = ['python3', 'main_multi.py', '48', '2', '0', '0.05', '1000', '0.001', '0.01', 'regression'] 
        if setting_name == 'omni_equation_cont':
            arguments = ['python3', 'main_multi.py', '48', '0', '0', '0.05', '1000', '0.001', '0.01', 'regression']
        elif setting_name == 'imit_equation_cont':
            arguments = ['python3', 'main_multi.py', '48', '2', '0', '0.05', '1000', '0.001', '0.01', 'regression']  
        elif setting_name == 'omni_class10':
            arguments = ['python3', 'main_multi.py', '30', '0', '0', '0.1', '1000', '0', '0.1']
        elif setting_name == 'imit_class10':
            arguments = ['python3', 'main_multi.py', '30', '2', '0.01', '0.1', '1000', '0.01', '0.1']
        elif setting_name == 'omni_class4':
            arguments = ['python3', 'main_multi.py', '50', '0', '0.01', '0.1', '200', '0.01', '0.1']
        elif setting_name == 'imit_class4':
            arguments = ['python3', 'main_multi.py', '50', '2', '0.01', '0.1', '200', '0.01', '0.1']
        elif setting_name == 'omni_regression':
            arguments = ['python3', 'main_multi.py', '50', '0', '0.1', '0.3', '300', '0.001', '0.05', 'regression']
        elif setting_name == 'imit_regression':
            arguments = ['python3', 'main_multi.py', '50', '2', '0.1', '0.3', '200', '0', '0.05', 'regression']
        elif setting_name == 'omni_mnist':
            arguments = ['python3', 'main_multi.py', '24', '0', '0.01', '0.1', '200', '0', '0.05',]
        elif setting_name == 'imit_mnist':
            arguments = ['python3', 'main_multi.py', '24', '2', '0.02', '0.1', '1000', '0', '0.05']
        elif setting_name == 'imit_peak_8':
            arguments = ['python3', 'main_irl.py', '1', 'E', '8', '0', '0.2', '70', '1', '200']
        elif setting_name == 'imit_peak_10':
            arguments = ['python3', 'main_irl.py', '1', 'E', '8', '0', '0.2', '70', '1', '200']
        elif setting_name == 'imit_random_8':
            arguments = ['python3', 'main_irl.py', '1', 'H', '8', '0.005', '0.3', '200', '5', '220']
        elif setting_name == 'imit_random_10':            
            arguments = ['python3', 'main_irl.py', '1', 'E', '8', '0', '0.2', '70', '1', '200']
        elif setting_name == 'omni_peak_8':                         
            arguments = ['python3', 'main_irl.py', '0', 'E', '8', '0', '0.3', '300', '1', '200']
        elif setting_name == 'imit_random_10':
            arguments = ['python3', 'main_irl.py', '1', 'E', '8', '0', '0.2', '70', '1', '200']            
        elif setting_name == 'omni_random_8':
            arguments = ['python3', 'main_irl.py', '0', 'H', '8', '0', '0.3', '200', '5', '220']
        else:
            print('possible setting_names are omni_equation, imit_equation, omni_class10, imit_class10, ')
            print('omni_class4, imit_class4, omni_regression, imit_regression, omni_mnist, imit_mnist')
            exit()
        collect_data(setting_name, random_seeds, arguments, type_)
        
    elif sys.argv[1] == 'plot':
        plot(setting_name)
    
    else:
        print('--Invalid arguments; use python3 plotband.py data "setting_name" to collect data; use python3 plotband.py plot "setting_name" to get plots')
        print()

if __name__ == '__main__':
    main()
