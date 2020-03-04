import main_multi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
    #equation
    main_multi.main(['main_multi.py', '48', '0', '0', '0.05', '1000', 'regression'])
    main_multi.main(['main_multi.py', '48', '2', '0', '0.05', '1000', 'regression'])

    #class 4
    main_multi.main(['main_multi.py', '30', '0', '0', '0.1', '1000'])
    main_multi.main(['main_multi.py', '30', '2', '0.01', '0.1', '1000'])

    #class 10
    main_multi.main(['main_multi.py', '50', '0', '0.01', '0.1', '200'])
    main_multi.main(['main_multi.py', '50', '2', '0.01', '0.1', '200'])

    #regression
    main_multi.main(['main_multi.py', '50', '0', '0.01', '0.1', '200', 'regression'])
    main_multi.main(['main_multi.py', '50', '2', '0.01', '0.1', '200', 'regression'])

    #mnist
    main_multi.main(['main_multi.py', '24', '0', '0', '0.05', '1000'])
    main_multi.main(['main_multi.py', '24', '2', '0', '0.05', '20000'])
    
if __name__ == '__main__':
    main()
