# Iterative Teacher-Aware Learning

## Note
This is our experimental code. It can replicate our experiment results, but hasn’t been thoroughly cleaned. A version ready to be released and distributed will become available later.

## Usage

To collect data and plots for a cooperative or adversarial teacher for a particular experiment, run with the command line
```bash
python3 plot_band.py -s setting_name 
```
where _setting_name_ is specified by ${experiment}\_${type of teacher}. Detailed settings are described in the Settings section. <br /> And an example command to collect data and plots for Linear Classifiers on MNIST Dataset with a cooperative teacher is
```bash
python3 plot_band.py -s mnist_coop 
```

To use the main_multi.py or main_irl.py script, run with the command line 
```bash
python3 main_multi.py detailed_setting_name random_seed
```
where _detailed_setting_name_ is specified by ${experiment}\_${type of teacher}\_${mode of teacher} followed by '_' and the imitate teacher's data dimension(MNIST, Equation) or data type(CIFAR). <br /> An example command to collect data for Linear Classifiers on MNIST Dataset with a cooperative omniscient teacher with data dimension 20 and random seed 0 is
```bash
python3 main_multi.py mnist_coop_omni 0
```
<br /> And an example command to collect data for Linear Classifiers on MNIST Dataset with a cooperative imitate teacher with data dimension 20 and random seed 0 is
```bash
python3 main_multi.py mnist_coop_imit_20 0
```

## Settings

The ${experiment} part of a setting is specified by the table 
|Experiment|command line| 
|-------|-------|  
|Linear Regression on Synthesized Data|regression|
|Linear Classifiers on Synthesized Data|class10|
|Linear Classifiers on MNIST Dataset|mnist|
|Linear Classifiers on CIFAR-10|cifar|
|Linear Regression for Equation Simplification|equation|
|Online Inverse Reinforcement Learning|irlH/irlE|

The imitate teacher's data dimension(MNIST, Equation) or data type(CIFAR) part of a setting is specified by the table 
|Experiment|Imitate Setting 1|Imitate Setting 2|     
|-------|-------|  -------|   
|mnist|20|30|
|equation|40|50|
|cifar|9|12|

The ${type of teacher} part of a setting is specified by the table 
|Type of Teacher|command line|    
|-------|-------|  
|cooperative|coop|
|adversarial|adv|

The ${mode of teacher} part of a setting is specified by the table 
|Mode of Teacher|command line|    
|-------|-------| 
|omniscient|omni|
|imitate|imit|

## Versions

In this project, we the following version of libraries:<br /> 

Tensorflow v1.15<br /> 
scikit-learn v0.22.1<br /> 
numpy v1.18.2<br /> 
