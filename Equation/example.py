import numpy as np
from easydict import EasyDict as edict
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def decode(l):
    codebook_ = [str(digit) for digit in range(10)] + ['+', '-', '/', '^', '='] + ['x', 'y', 'z', 'w', '1'][0:-1] + [' '] + ['']
    equation = ""
    for s in l:
        equation += codebook_[s] 
    return equation

def main():
    file_name = '../Data/equations.txt'
    lower_tests = np.load('lower_tests.npy', allow_pickle=True)
    higher_tests = np.load('higher_tests.npy', allow_pickle=True)
    l = [62,83,89]
    for c in range(100):
        if c in l:
            print('lower equation', decode(lower_tests[c]))
            print('higher equation', decode(higher_tests[c]))
            print()

if __name__ == '__main__':
    main()
