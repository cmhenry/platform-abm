import minitiebout

### Experiment 1: single platform institutional comparisons

param_1a_direct = {
    'n_comms': 100,
    'n_plats': 1,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'direct',
    'extremists': 'no',
    'percent_extremists': 5,
    'coalitions': 5,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 2,
    'stop_condition': 'steps',
    'seed': 1999
}

param_1b_coalition = {
    'n_comms': 100,
    'n_plats': 1,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'coalition',
    'extremists': 'no',
    'percent_extremists': 5,
    'coalitions': 3,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 2,
    'stop_condition': 'steps',
    'seed': 1999
}

param_1c_algorithm = {
    'n_comms': 100,
    'n_plats': 1,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'algorithm',
    'extremists': 'no',
    'percent_extremists': 5,
    'coalitions': 3,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 3,
    'stop_condition': 'steps',
    'seed': 1999
}

model_1a = minitiebout.MiniTiebout(param_1a_direct)
model_1b = minitiebout.MiniTiebout(param_1b_coalition)
model_1c = minitiebout.MiniTiebout(param_1c_algorithm)

results_1a = model_1a.run()
results_1b = model_1b.run()
results_1c = model_1c.run()

results_1a.reporters 
results_1b.reporters 
results_1c.reporters 

### Experiment 1 results:

# >>> results_1a.reporters 
#    seed  average_moves  average_utility
# 0  1999            1.0             5.07
# >>> results_1b.reporters
#    seed  average_moves  average_utility
# 0  1999            1.0             4.77
# >>> results_1c.reporters
#    seed  average_moves  average_utility
# 0  1999            1.0             5.16

### Experiment 1 trial 2: multiple platform institutional comparisons

param_1a_t2_direct = {
    'n_comms': 100,
    'n_plats': 5,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'direct',
    'extremists': 'no',
    'percent_extremists': 5,
    'coalitions': 5,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 2,
    'stop_condition': 'steps',
    'seed': 1999
}

param_1b_t2_coalition = {
    'n_comms': 100,
    'n_plats': 5,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'coalition',
    'extremists': 'no',
    'percent_extremists': 5,
    'coalitions': 3,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 2,
    'stop_condition': 'steps',
    'seed': 1999
}

param_1c_t2_algorithm = {
    'n_comms': 100,
    'n_plats': 5,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'algorithm',
    'extremists': 'no',
    'percent_extremists': 5,
    'coalitions': 3,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 3,
    'stop_condition': 'steps',
    'seed': 1999
}

model_1a_t2 = minitiebout.MiniTiebout(param_1a_t2_direct)
model_1b_t2 = minitiebout.MiniTiebout(param_1b_t2_coalition)
model_1c_t2 = minitiebout.MiniTiebout(param_1c_t2_algorithm)

results_1a_t2 = model_1a_t2.run()
results_1b_t2 = model_1b_t2.run()
results_1c_t2 = model_1c_t2.run()

results_1a_t2.reporters 
results_1b_t2.reporters 
results_1c_t2.reporters 

# >>> results_1a_t2.reporters
#    seed  average_moves  average_utility
# 0  1999           23.6             5.68
# >>> results_1b_t2.reporters
#    seed  average_moves  average_utility
# 0  1999           39.5             4.74
# >>> results_1c_t2.reporters
#    seed  average_moves  average_utility
# 0  1999           1.88             6.75

### Experiment 2: mixed institutions, multiple platforms

param_2_mixed = {
    'n_comms': 300,
    'n_plats': 15,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'mixed',
    'extremists': 'no',
    'percent_extremists': 5,
    'coalitions': 3,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 3,
    'stop_condition': 'steps',
    'seed': 1999
}

model_2 = minitiebout.MiniTiebout(param_2_mixed)

results_2 = model_2.run()

results_2.reporters

# >>> print(results_2.reporters)
#    seed  average_moves  average_utility  n_direct_comms  n_coalition_comms  n_algo_comms  ratio_direct  ratio_coalition  ratio_algo  \
# 0  1999          38.28             5.54              58                 55           187      0.193333         0.183333    0.623333   

#    avg_utility_direct  avg_utility_coalition  avg_utility_algo  
# 0           18.051724               5.054545           5.59893

### Experiment 3 trial 1: multiple platform institutional comparisons + extremists

param_3a_t1_direct = {
    'n_comms': 100,
    'n_plats': 5,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'direct',
    'extremists': 'yes',
    'percent_extremists': 15,
    'coalitions': 5,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 2,
    'stop_condition': 'steps',
    'seed': 1999
}

param_3b_t1_coalition = {
    'n_comms': 100,
    'n_plats': 5,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'coalition',
    'extremists': 'yes',
    'percent_extremists': 15,
    'coalitions': 3,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 2,
    'stop_condition': 'steps',
    'seed': 1999
}

param_3c_t1_algorithm = {
    'n_comms': 100,
    'n_plats': 5,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'algorithm',
    'extremists': 'yes',
    'percent_extremists': 15,
    'coalitions': 3,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 3,
    'stop_condition': 'steps',
    'seed': 1999
}

model_3a_t1 = minitiebout.MiniTiebout(param_3a_t1_direct)
model_3b_t1 = minitiebout.MiniTiebout(param_3b_t1_coalition)
model_3c_t1 = minitiebout.MiniTiebout(param_3c_t1_algorithm)

results_3a_t1 = model_3a_t1.run()
results_3b_t1 = model_3b_t1.run()
results_3c_t1 = model_3c_t1.run()

results_3a_t1.reporters 
results_3b_t1.reporters 
results_3c_t1.reporters 

### with 5 percent extremists
# >>> results_3a_t1.reporters 
#    seed  average_moves  average_utility  average_extremist_utility  average_mainstream_utility
# 0  1999          37.44             5.88                       26.4                         4.8
# >>> results_3b_t1.reporters 
#    seed  average_moves  average_utility  average_extremist_utility  average_mainstream_utility
# 0  1999          44.77             4.71                       23.8                    3.705263
# >>> results_3c_t1.reporters 
#    seed  average_moves  average_utility  average_extremist_utility  average_mainstream_utility
# 0  1999          29.84             6.56                       17.8                    5.968421

### with 10 percent extremists
# >>> results_3a_t1.reporters 
#    seed  average_moves  average_utility  average_extremist_utility  average_mainstream_utility
# 0  1999          40.69             5.65                       23.0                    3.722222
# >>> results_3b_t1.reporters 
#    seed  average_moves  average_utility  average_extremist_utility  average_mainstream_utility
# 0  1999          44.13             4.91                       20.5                    3.177778
# >>> results_3c_t1.reporters 
#    seed  average_moves  average_utility  average_extremist_utility  average_mainstream_utility
# 0  1999          34.72             6.43                       20.4                    4.877778

### with 15 percent extremists
# >>> results_3a_t1.reporters 
#    seed  average_moves  average_utility  average_extremist_utility  average_mainstream_utility
# 0  1999           43.5             5.65                  24.666667                    2.294118
# >>> results_3b_t1.reporters 
#    seed  average_moves  average_utility  average_extremist_utility  average_mainstream_utility
# 0  1999           43.5             4.32                       20.6                    1.447059
# >>> results_3c_t1.reporters 
#    seed  average_moves  average_utility  average_extremist_utility  average_mainstream_utility
# 0  1999           43.5             5.42                  18.333333                    3.141176

### Experiment 3b: mixed institutions, multiple platforms + extremists

param_3b_mixed = {
    'n_comms': 300,
    'n_plats': 15,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'mixed',
    'extremists': 'yes',
    'percent_extremists': 5,
    'coalitions': 3,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 3,
    'stop_condition': 'steps',
    'seed': 1999
}

model_3b = minitiebout.MiniTiebout(param_3b_mixed)

results_3b = model_3b.run()

results_3b.reporters

### with 5 percent extremists + 15 platforms
# >>> results_3b.reporters
#    seed  average_moves  average_utility  n_direct_comms  n_coalition_comms  n_algo_comms  ratio_direct  ratio_coalition  ratio_algo  \
# 0  1999      44.546667         5.746667              62                 52           186      0.206667         0.173333        0.62   

#    avg_utility_direct  avg_utility_coalition  avg_utility_algo  average_extremist_utility  average_mainstream_utility  
# 0           17.967742               4.153846          5.989247                  29.066667                    4.519298

param_3b_t2_mixed = {
    'n_comms': 300,
    'n_plats': 3,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'mixed',
    'extremists': 'yes',
    'percent_extremists': 5,
    'coalitions': 6,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 3,
    'stop_condition': 'steps',
    'seed': 1999
}

model_3b_t2 = minitiebout.MiniTiebout(param_3b_t2_mixed)

results_3b_t2 = model_3b_t2.run()

results_3b_t2.reporters

### with 5 percent extremists + 6 platforms
# >>> results_3b_t2.reporters
#    seed  average_moves  average_utility  n_direct_comms  n_coalition_comms  n_algo_comms  ratio_direct  ratio_coalition  ratio_algo  \
# 0  1999      46.736667         5.636667              60                 67           173           0.2         0.223333    0.576667   

#    avg_utility_direct  avg_utility_coalition  avg_utility_algo  average_extremist_utility  average_mainstream_utility  
# 0           16.583333                5.61194          5.751445                       48.0                    3.407018  

param_3b_t3_mixed = {
    'n_comms': 300,
    'n_plats': 6,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'mixed',
    'extremists': 'yes',
    'percent_extremists': 5,
    'coalitions': 6,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 3,
    'stop_condition': 'steps',
    'seed': 1999
}

model_3b_t3 = minitiebout.MiniTiebout(param_3b_t3_mixed)

results_3b_t3 = model_3b_t3.run()

results_3b_t3.reporters

### with 5 percent extremists + 6 platforms + 6 coalitions
# >>> results_3b_t2.reporters
#    seed  average_moves  average_utility  n_direct_comms  n_coalition_comms  n_algo_comms  ratio_direct  ratio_coalition  ratio_algo  \
# 0  1999      47.703333         5.556667              77                 47           176      0.256667         0.156667    0.586667   

#    avg_utility_direct  avg_utility_coalition  avg_utility_algo  average_extremist_utility  average_mainstream_utility  
# 0           12.688312               7.212766          5.551136                  53.533333                    3.031579 

param_3b_jumbo_mixed = {
    'n_comms': 1500,
    'n_plats': 6,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'mixed',
    'extremists': 'yes',
    'percent_extremists': 5,
    'coalitions': 6,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 3,
    'stop_condition': 'steps',
    'seed': 1999
}

model_3b_jumbo = minitiebout.MiniTiebout(param_3b_jumbo_mixed)

results_3b_jumbo = model_3b_jumbo.run()

results_3b_jumbo.reporters

### with 5 percent extremists + 6 platforms + 6 coalitions + 1500 communities
# >>> results_3b_jumbo.reporters
#    seed  average_moves  average_utility  n_direct_comms  n_coalition_comms  n_algo_comms  ratio_direct  ratio_coalition  ratio_algo  \
# 0  1999           48.5         5.475333             307                290           903      0.204667         0.193333       0.602   

#    avg_utility_direct  avg_utility_coalition  avg_utility_algo  average_extremist_utility  average_mainstream_utility  
# 0           15.990228               3.106897          5.436323                 243.626667                   -7.058947 
## QUICKTAKE: direct voting on policies -> more utility for communities, both types; coalition utility goes up when
## there are more coalition slots available and communities can shrink; extremists netting higher overall utility
## than mainstream; although some extremists end up on "extremist" platforms with favorable policies, big utility
## gains for extremists are those who stick around non-extremist platforms with just-unfavorable policies &
## steal utility from mainstream