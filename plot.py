import numpy as np
import os
import sys
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from easydict import EasyDict as edict

def main():
    fig, axs = plt.subplots(2, 2,constrained_layout= True)
    
    path = '../data_npy/'
    
    title = '_'
    mode_idx = int(sys.argv[2])
    modes = ['omni', 'surr', 'imit']
    mode = modes[mode_idx]
    title += mode
    title += '_'
    task = 'classification' if len(sys.argv) == 6 else 'regression'
    title += task
    title += '_'
    
    lr = 1e-3
    dd = int(sys.argv[1])
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
    
    dps = 3 * dd if task == 'classification' else 6 * dd
    num_particles = 3000
    train_iter_simple = 2000
    train_iter_smart = 2000
    reg_coef = 0
    
    config_T = edict({'data_pool_size_class': dps, 'data_dim': dd,'lr': lr, 'sample_size': 20,
                      'transform': mode == 'imit', 'num_classes': num_classes, 'task': task,
                      })
    config_LS = edict({'particle_num': num_particles, 'data_dim': dd, 'reg_coef': reg_coef, 'lr': lr, 'task': task,
                       'num_classes': num_classes, 'noise_scale_min': float(sys.argv[3]), 'noise_scale_max': float(sys.argv[4]),
                       'noise_scale_decay': float(sys.argv[5]), 'target_ratio': 0, 'new_ratio': 1, 'replace_count': 1, "prob": 1})


    title = title + '.npy'
    dists3 = np.load(path + 'dist3' + title, allow_pickle = True)
    dists4 = np.load(path + 'dist4' + title, allow_pickle = True)
    #dists5 = np.load(path + 'dist5' + title, allow_pickle = True)
    dists5 = np.load('dist5' + title, allow_pickle = True)              

    dists3_ = np.load(path + 'dist3_' + title, allow_pickle = True)
    dists4_ = np.load(path + 'dist4_' + title, allow_pickle = True)
    #dists5_ = np.load(path + 'dist5_' + title, allow_pickle = True)
    dists5_ = np.load('dist5_' + title, allow_pickle = True) 
   
    logpdfs3 = np.load(path + 'logpdfs3' + title, allow_pickle = True)
    logpdfs4 = np.load(path + 'logpdfs4' + title, allow_pickle = True)
    #logpdfs5 = np.load(path + 'logpdfs5' + title, allow_pickle = True)
    logpdfs5 = np.load('logpdfs5' + title, allow_pickle = True)

    accuracies3 = np.load(path + 'accuracies3' + title, allow_pickle = True)
    accuracies4 = np.load(path + 'accuracies4' + title, allow_pickle = True)
    #accuracies5 = np.load(path + 'accuracies5' + title, allow_pickle = True)
    accuracies5 = np.load('accuracies5' + title, allow_pickle = True)   

    '''
    print(dists5_.shape)
    print(logpdfs5.shape)
    print(accuracies5.shape)
    print()
    print(dists3.shape)
    print(dists3_.shape)
    print(logpdfs3.shape)
    print(accuracies3.shape)
    print()
    exit()
    '''
    line3, = axs[0, 0].plot(dists3, label = 'smarter')
    line4, = axs[0, 0].plot(dists4, label = 'noise')
    line5, = axs[0, 0].plot(dists5, label = 'remove')
    axs[0, 0].set_title('mean_dist')


    line3, = axs[1, 1].plot(logpdfs3, label = 'smarter')
    line4, = axs[1, 1].plot(logpdfs4, label = 'noise')
    line5, = axs[1, 1].plot(logpdfs5, label = 'remove')
    axs[1, 1].set_title('log pdf per 20 iters')

    line3, = axs[0, 1].plot(accuracies3, label = 'smarter')
    line4, = axs[0, 1].plot(accuracies4, label = 'noise')
    line5, = axs[0, 1].plot(accuracies5, label = 'remove')
    
    axs[0, 1].set_title('test loss')

    line3, = axs[1, 0].plot(dists3_, label = 'smarter')
    line4, = axs[1, 0].plot(dists4_, label = 'noise')
    line5, = axs[1, 0].plot(dists5_, label = 'remove')
    axs[1, 0].set_title('dist mean')


    axs[0, 1].legend([line3, line4, line5],
               ['Pragmatic Replacement', 'noise','particle removal'], prop={'size': 10})
    
    
    fig.suptitle('%s class: %d: dim:%d_data:%d/%d/%d_particle:%d_noise: %f, %f, %d, ratio: %f, %f, lr:  %f' %\
              (mode, num_classes, dd, config_LS.replace_count, config_T.sample_size, dps, num_particles,
               config_LS.noise_scale_min, config_LS.noise_scale_max, config_LS.noise_scale_decay,
               config_LS.target_ratio, config_LS.new_ratio, config_LS.lr))
    plt.show()
    #plt.savefig('%s.png' % title)


if __name__ == '__main__':
    main()
