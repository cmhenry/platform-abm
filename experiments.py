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

param_2_mixed_small = {
    'n_comms': 100,
    'n_plats': 3,
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

param_2_mixed_med = {
    'n_comms': 300,
    'n_plats': 9,
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

param_2_mixed_large = {
    'n_comms': 900,
    'n_plats': 27,
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

model_2_small = minitiebout.MiniTiebout(param_2_mixed_small)
model_2_med = minitiebout.MiniTiebout(param_2_mixed_med)
model_2_large = minitiebout.MiniTiebout(param_2_mixed_large)


results_2_small = model_2_small.run()
esults_2_med = model_2_med.run()
results_2_large = model_2_large.run()


model_2_small.reporters
model_2_med.reporters
model_2_large.reporters

direct_util = model_2_large.communities.select(
    model_2_large.communities.platform.institution == 'direct').current_utility
coal_util = model_2_large.communities.select(
    model_2_large.communities.platform.institution == 'coalition').current_utility
algo_util = model_2_large.communities.select(
    model_2_large.communities.platform.institution == 'algorithmic').current_utility



# # >>> results_2_small.reporters
# >>> model_2_small.reporters
# {'seed': 1999, 'average_moves': 21.97, 'average_utility': 6.0, 
# 'n_direct_comms': 30, 'n_coalition_comms': 10, 'n_algo_comms': 60, 
# 'ratio_direct': 0.3, 'ratio_coalition': 0.1, 'ratio_algo': 0.6, 
# 'util_direct': 186, 'util_coalition': 54, 'util_algo': 360, 
# 'avg_utility_direct': 6.2, 'avg_utility_coalition': 5.4, 
# 'avg_utility_algo': 6.0}
# >>> model_2_med.reporters
# {'seed': 1999, 'average_moves': 34.35666666666667, 'average_utility': 5.69, 
# 'n_direct_comms': 82, 'n_coalition_comms': 47, 'n_algo_comms': 171, 
# 'ratio_direct': 0.2733333333333333, 'ratio_coalition': 0.15666666666666668, 
# 'ratio_algo': 0.57, 'util_direct': 508, 'util_coalition': 245, 
# 'util_algo': 954, 'avg_utility_direct': 6.195121951219512,
#  'avg_utility_coalition': 5.212765957446808, 
#  'avg_utility_algo': 5.578947368421052}
# >>> model_2_large.reporters
# {'seed': 1999, 'average_moves': 42.27111111111111, 
# 'average_utility': 5.7877777777777775, 'n_direct_comms': 174, 
# 'n_coalition_comms': 192, 'n_algo_comms': 534,
#  'ratio_direct': 0.19333333333333333, 'ratio_coalition': 0.21333333333333335,
#   'ratio_algo': 0.5933333333333334, 'util_direct': 1055, 'util_coalition': 1099,
#    'util_algo': 3055, 'avg_utility_direct': 6.063218390804598, 
#    'avg_utility_coalition': 5.723958333333333, 
#    'avg_utility_algo': 5.7209737827715355}

# kwargs = dict(alpha=0.9, bins=10, histtype = 'step', density=False, stacked=True, rwidth = 0.9)

# plt.hist(np.array(algo_util), **kwargs, color='g', label = 'Algorithmic')
# plt.hist(np.array(coal_util), **kwargs, color='b', label = 'Movement')
# plt.hist(np.array(direct_util), **kwargs, color='r', label = 'Direct')
# plt.grid(axis='y', alpha=0.75)
# plt.gca().set(title='', ylabel='Community Count', xlabel = 'Utility')
# plt.xticks(range(0,11))
# plt.legend()

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