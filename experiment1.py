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

param_1a_coalition = {
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

param_1a_algorithm = {
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
model_1a = minitiebout.MiniTiebout(param_1a_coalition)
model_1a = minitiebout.MiniTiebout(param_1a_algorithm)

results_1a = model_1a.run()
results_1a = model_1b.run()
results_1a = model_1c.run()

results_1a.reporters 
results_1a.reporters 
results_1a.reporters 

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

param_1b_direct = {
    'n_comms': 100,
    'n_plats': 5,
    'p_space': 25,
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
    'n_plats': 5,
    'p_space': 25,
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

param_1b_algorithm = {
    'n_comms': 100,
    'n_plats': 5,
    'p_space': 25,
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

model_1b = minitiebout.MiniTiebout(param_1b_direct)
model_1b = minitiebout.MiniTiebout(param_1b_coalition)
model_1b = minitiebout.MiniTiebout(param_1b_algorithm)

results_1b = model_1b.run()
results_1b = model_1b.run()
results_1b = model_1b.run()

results_1b.reporters 
results_1b.reporters 
results_1b.reporters 

# >>> results_1a_t2.reporters
#    seed  average_moves  average_utility
# 0  1999           23.6             5.68
# >>> results_1b_t2.reporters
#    seed  average_moves  average_utility
# 0  1999           39.5             4.74
# >>> results_1c_t2.reporters
#    seed  average_moves  average_utility
# 0  1999           1.88             6.75


### Experiment 1 trial 3: varying multiple platform institutional comparisons
### DIRECT
param_1c_small_direct = {
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

param_1c_mid_direct = {
    'n_comms': 200,
    'n_plats': 10,
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

param_1c_jumbo_direct = {
    'n_comms': 400,
    'n_plats': 20,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'direct',
    'extremists': 'no',
    'percent_extremists': 5,
    'coalitions': 3,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 2,
    'stop_condition': 'steps',
    'seed': 1999
}

model_1c_small_direct = minitiebout.MiniTiebout(param_1c_small_direct)
model_1c_mid_direct = minitiebout.MiniTiebout(param_1c_mid_direct)
model_1c_jumbo_direct = minitiebout.MiniTiebout(param_1c_jumbo_direct)

results_1c_small_direct = model_1c_small_direct.run()
results_1c_mid_direct = model_1c_mid_direct.run()
results_1c_jumbo_direct = model_1c_jumbo_direct.run()

results_1c_small_direct.reporters 
results_1c_mid_direct.reporters 
results_1c_jumbo_direct.reporters 

# >>> results_1c_small_direct.reporters 
#    seed  average_moves  average_utility
# 0  1999          23.73             5.75
# >>> results_1c_mid_direct.reporters 
#    seed  average_moves  average_utility
# 0  1999          32.47              5.5
# >>> results_1c_jumbo_direct.reporters 
#    seed  average_moves  average_utility
# 0  1999          33.98             5.71

### COALITION

param_1c_small_coalition = {
    'n_comms': 100,
    'n_plats': 5,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'coalition',
    'extremists': 'no',
    'percent_extremists': 5,
    'coalitions': 5,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 2,
    'stop_condition': 'steps',
    'seed': 1999
}

param_1c_mid_coalition = {
    'n_comms': 200,
    'n_plats': 10,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'coalition',
    'extremists': 'no',
    'percent_extremists': 5,
    'coalitions': 5,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 2,
    'stop_condition': 'steps',
    'seed': 1999
}

param_1c_jumbo_coalition = {
    'n_comms': 400,
    'n_plats': 20,
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

model_1c_small_coalition = minitiebout.MiniTiebout(param_1c_small_coalition)
model_1c_mid_coalition = minitiebout.MiniTiebout(param_1c_mid_coalition)
model_1c_jumbo_coalition = minitiebout.MiniTiebout(param_1c_jumbo_coalition)

results_1c_small_coalition = model_1c_small_coalition.run()
results_1c_mid_coalition = model_1c_mid_coalition.run()
results_1c_jumbo_coalition = model_1c_jumbo_coalition.run()

results_1c_small_coalition.reporters 
results_1c_mid_coalition.reporters 
results_1c_jumbo_coalition.reporters 

# >>> results_1c_small_coalition.reporters 
#    seed  average_moves  average_utility
# 0  1999           40.7             4.49
# >>> results_1c_mid_coalition.reporters 
#    seed  average_moves  average_utility
# 0  1999          45.84             4.19
# >>> results_1c_jumbo_coalition.reporters 
#    seed  average_moves  average_utility
# 0  1999          48.26             4.55

### ALGO

param_1c_small_algorithmic = {
    'n_comms': 100,
    'n_plats': 5,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'algorithmic',
    'extremists': 'no',
    'percent_extremists': 5,
    'coalitions': 5,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 2,
    'stop_condition': 'steps',
    'seed': 1999
}

param_1c_mid_algorithmic = {
    'n_comms': 200,
    'n_plats': 10,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'algorithmic',
    'extremists': 'no',
    'percent_extremists': 5,
    'coalitions': 5,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 2,
    'stop_condition': 'steps',
    'seed': 1999
}

param_1c_jumbo_algorithmic = {
    'n_comms': 400,
    'n_plats': 20,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'algorithmic',
    'extremists': 'no',
    'percent_extremists': 5,
    'coalitions': 3,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 2,
    'stop_condition': 'steps',
    'seed': 1999
}

model_1c_small_algorithmic = minitiebout.MiniTiebout(param_1c_small_algorithmic)
model_1c_mid_algorithmic = minitiebout.MiniTiebout(param_1c_mid_algorithmic)
model_1c_jumbo_algorithmic = minitiebout.MiniTiebout(param_1c_jumbo_algorithmic)

results_1c_small_algorithmic = model_1c_small_algorithmic.run()
results_1c_mid_algorithmic = model_1c_mid_algorithmic.run()
results_1c_jumbo_algorithmic = model_1c_jumbo_algorithmic.run()

results_1c_small_algorithmic.reporters 
results_1c_mid_algorithmic.reporters 
results_1c_jumbo_algorithmic.reporters 