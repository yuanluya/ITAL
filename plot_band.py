import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from easydict import EasyDict as edict
import multiprocessing
import time

def get_path(data_cate, arguments):
    #add dist or dist_ or ... as argyment
    lines = ['0', '1', '2', '3', '4', '5', 'batch', 'sgd']
    titles = []
    methods = edict()

    title = '_'
    mode_idx = int(arguments[3])
    modes = ['omni', 'surr', 'imit']
    mode = modes[mode_idx]
    title += mode
    title += '_'
    task = 'classification' if len(arguments) == 7 else 'regression'
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

def save_csv(data_cate, setting_name, random_seeds, arguments):    
    methods_code = {'0': 'No Rep', '1': 'IMT', '2': 'Rand Rep', '3': 'prag', '4': 'Prag (Strt Lin)', '5': 'IMT (Strt Lin)', 'batch': 'batch', 'sgd': 'sgd'} 
    titles, methods = get_path(data_cate, arguments)
    data = []
    method = []
    iterations = []
    for t in titles:
        for s in random_seeds:
            filename = t + '_' + str(s) + '.npy'
            d = np.load(filename, allow_pickle = True)
            length = len(d)
            data.append(d)
            method += [methods_code[methods[t]] for j in range(length)]
            iterations += [j for j in range(length)]
    data = np.array(data).flatten()

    save_name = data_cate + '_' + setting_name + '.csv'
    df = pd.DataFrame({'method': method,
                       'iteration': iterations,
                       'data': data})
    df.to_csv(save_name)
    print('saved file to %s \n' % save_name)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def collect_data(setting_name, random_seeds, arguments):
    #child_processes = []
    cpu_cnt = int(multiprocessing.cpu_count()/2)
    
    random_seed = list(chunks(random_seeds, cpu_cnt))
    for ss in random_seed:
        child_processes = []
        for s in ss:
            if len(arguments) > 7:
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

    save_csv('dist', setting_name, random_seeds, arguments)
    save_csv('dist_', setting_name, random_seeds, arguments)
    save_csv('logpdfs', setting_name, random_seeds, arguments)
    save_csv('accuracies', setting_name, random_seeds, arguments)
    print('collected data\n')

def plot(setting_name):
    paper_rc = {'lines.linewidth': 2.5}
    sns.set(style="darkgrid")
    sns.set(font_scale=1, rc = paper_rc)

    f, axes = plt.subplots(2, 2, constrained_layout = True)
    
    results00 = pd.read_csv('%s.csv' % ('dist'+'_'+setting_name))
    results01 = pd.read_csv('%s.csv' % ('accuracies'+'_'+setting_name))
    results10 = pd.read_csv('%s.csv' % ('dist_'+'_'+setting_name))
    results11 = pd.read_csv('%s.csv' % ('logpdfs'+'_'+setting_name))    
    
    sns.lineplot(x="iteration", y="data",
                 hue="method", data=results00, ax=axes[0,0])
    sns.lineplot(x="iteration", y="data",
                 hue="method", data=results01, ax=axes[0,1])
    sns.lineplot(x="iteration", y="data",
                 hue="method", data=results10, ax=axes[1,0])
    sns.lineplot(x="iteration", y="data",
                 hue="method", data=results11, ax=axes[1,1])
    axes[0, 0].set_title('mean_dist')
    axes[1, 1].set_title('log pdf per 20 iters')
    axes[0, 1].set_title('test loss')
    axes[1, 0].set_title('dist mean')
    plt.savefig('omni_regression.png')
    plt.show()

def main():
    if len(sys.argv) != 3:
        print('--Invalid arguments; use python3 plotband.py data "setting_name" to collect data; use python3 plotband.py plot "setting_name" to get plots')
        exit()

    random_seeds = [j for j in range(1)]
    setting_name = sys.argv[2]

    if sys.argv[1] == 'data':
        if setting_name == 'omni_equation':
            arguments = ['python3', 'main_multi.py', '48', '0', '0', '0.05', '1000', 'regression']
        elif setting_name == 'imit_equation':
            arguments = ['python3', 'main_multi.py', '48', '2', '0', '0.05', '1000', 'regression'] 
        elif setting_name == 'omni_class10':
            arguments = ['python3', 'main_multi.py', '30', '0', '0', '0.1', '1000']
        elif setting_name == 'imit_class10':
            arguments = ['python3', 'main_multi.py', '30', '2', '0.01', '0.1', '1000']
        elif setting_name == 'omni_class4':
            arguments = ['python3', 'main_multi.py', '50', '0', '0.01', '0.1', '200']
        elif setting_name == 'imit_class4':
            arguments = ['python3', 'main_multi.py', '50', '2', '0.01', '0.1', '200']
        elif setting_name == 'omni_regression':
            arguments = ['python3', 'main_multi.py', '50', '0', '0.01', '0.1', '200', 'regression']
        elif setting_name == 'imit_regression':
            arguments = ['python3', 'main_multi.py', '50', '2', '0.01', '0.1', '200', 'regression']
        elif setting_name == 'omni_mnist':
            arguments = ['python3', 'main_multi.py', '24', '0', '0', '0.05', '1000']
        elif setting_name == 'imit_mnist':
            arguments = ['python3', 'main_multi.py', '24', '2', '0', '0.05', '1000']
        else:
            print('possible setting_names are omni_equation, imit_equation, omni_class10, imit_class10, ')
            print('omni_class4, imit_class4, omni_regression, imit_regression, omni_mnist, imit_mnist')
            exit()
        collect_data(setting_name, random_seeds, arguments)
        
    elif sys.argv[1] == 'plot':
        plot(setting_name)
    
    else:
        print('--Invalid arguments; use python3 plotband.py data "setting_name" to collect data; use python3 plotband.py plot "setting_name" to get plots')
        print()

if __name__ == '__main__':
    main()
