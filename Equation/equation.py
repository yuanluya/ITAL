import numpy as np
from scipy.special import comb
import copy

class Equation:
    def __init__(self, num_var, order, coef_range):
        assert(num_var <= 4)
        self.num_var_ = num_var
        self.order_ = order
        self.coef_range_ = coef_range
        all_var_names = ['x', 'y', 'z', 'w']
        self.all_var_names_ = all_var_names[0: self.num_var_]
        self.all_var_names_.append('1')
        all_variables_stack = [[]]
        self.all_variables_ = []
        self.all_coefs_ = []

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

        for i in range(1, coef_range + 1):
            for j in range(1, coef_range + 1):
                g = np.gcd(i, j)
                ii = i / g
                jj = j / g
                if jj == 1:
                    self.all_coefs_.append('%d' % ii)
                else:
                    self.all_coefs_.append('%d/%d' % (ii, jj))
        self.all_coefs_ = list(set(self.all_coefs_))

def main():
    eq = Equation(1, 6, 20)
    print(eq.all_variables_)
    print(eq.all_coefs_)

if __name__ == '__main__':
    main()

