import numpy as np
from easydict import EasyDict as edict
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from neg_training_set import str_tuple
from equation import Equation

def decode(l):
    codebook_ = [str(digit) for digit in range(10)] + ['+', '-', '/', '^', '='] + ['x', 'y'] + [' ']  + ['']
    equation = ""
    for s in l:
        equation += codebook_[s] 
    return equation

def main():
    file_name = '../Data/equations.txt'
    lower_tests = np.load('lower_tests.npy', allow_pickle=True)
    higher_tests = np.load('higher_tests.npy', allow_pickle=True)
    eq = Equation(2, 4, 20, 5)
    
    '''
    l = [281,323,402,406,546,591,643,701,947]
    for c in range(1000):
        if c in l:
            print('lower equation', decode(lower_tests[c][:-1]))
            print('higher equation', decode(higher_tests[c][:-1]))
            print()
    '''
    equation = '+11/3y^1 -1/3x^4 +5x^1y^2 +17 -8/3y^2 -19y^2 -6/5x^2 = +14x^3y^1 -20/3y^1e'
    print(str_tuple(equation))
    history = eq.simplify(str_tuple(equation))
    print(history)
if __name__ == '__main__':
    main()
