
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
