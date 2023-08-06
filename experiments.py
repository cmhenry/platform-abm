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

n_algo_comms = 0
for platform in model_2.platforms.select(model_2.platforms.institution == 'algorithmic'):
                n_algo_comms += len(platform.communities) 
n_algo = sum(1 for plat in model_2.platforms if plat.institution == 'algorithmic')

n_algo_comms / n_algo

results_2 = model_2.run()

results_2.reporters