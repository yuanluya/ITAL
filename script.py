import subprocess 

def main():
    #equation 
    #subprocess.call(['python3', 'main_multi.py', '48', '0', '0', '0.05', '1000', 'regression'])
    subprocess.call(['python3', 'main_multi.py', '48', '2', '0', '0.05', '1000', 'regression'])
    #class 10
    subprocess.call(['python3', 'main_multi.py', '30', '0', '0', '0.1', '1000'])
    subprocess.call(['python3', 'main_multi.py', '30', '2', '0.01', '0.1', '1000'])

    #class 4  
    subprocess.call(['python3', 'main_multi.py', '50', '0', '0.01', '0.1', '200'])
    subprocess.call(['python3', 'main_multi.py', '50', '2', '0.01', '0.1', '200'])
    
    #regression
    subprocess.call(['python3', 'main_multi.py', '50', '0', '0.01', '0.1', '200', 'regression'])
    subprocess.call(['python3', 'main_multi.py', '50', '2', '0.01', '0.1', '200', 'regression'])

    #mnist
    subprocess.call(['python3', 'main_multi.py', '24', '0', '0', '0.05', '1000'])
    subprocess.call(['python3', 'main_multi.py', '24', '2', '0', '0.05', '20000'])     
    
if __name__ == '__main__':
    main()
