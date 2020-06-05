# Iterative Teacher-Aware Learning


## Usage

To collect data and plots for a cooperative or adversarial teacher for a particular experiment, run with the command line
```bash
python3 plot_band.py -s setting_name 
```
where _setting_name_ is specified by ${experiment}\_${type of teacher}. Detailed settings are described in the Settings section.

To use the main_multi.py or main_irl.py script, run with the command line 
```bash
python3 main_multi.py detailed_setting_name random_seed
```
where _detailed_setting_name_ is specified by ${experiment}\_${type of teacher}\_${mode of teacher} followed by '_' and the imitate teacher's data dimension(MNIST, Equation) or data type(CIFAR).
## Settings


|Experiment|experiment| 
|-------|-------|  
|Linear Regression on Synthesized Data|regression|
|Linear Classifiers on Synthesized Data|class10|
|Linear Classifiers on CIFAR-10|cifar|
|Linear Regression for Equation Simplification|equation|
|Online Inverse Reinforcement Learning|irlH/irlE|

|Experiment|Imitate Setting 1|Imitate Setting 2|     
|-------|-------|  -------|   
|mnist|20|30|
|equation|40|50|
|cifar|9|12|
