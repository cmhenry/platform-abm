import minitiebout

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
