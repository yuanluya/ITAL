# Iterative Teacher-Aware Learning


## Usage

To collect data and plots for a cooperative or adversarial teacher for a particular experiment, run with the command line
```bash
python3 plot_band.py -s setting_name 
```
where setting_name is specified by ${experiment}_${type of teacher}. Detailed settings are described in the Settings section.

To use the main_multi.py or main_irl.py script, run with the command line 
```bash
python3 main_multi.py detailed_setting_name random_seed
```
where detailed_setting_name is specified by ${experiment}_${type of teacher}_${mode of teacher} followed by '_' and the imitate teacher's data dimension(MNIST, Equation) or data type(CIFAR).
## Settings
