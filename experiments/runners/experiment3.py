"""Experiment 3: multiple platform institutional comparisons + extremists."""

from platform_abm import MiniTiebout

### Experiment 3 trial 1: multiple platform institutional comparisons + extremists

param_3a_t1_direct = {
    "n_comms": 100,
    "n_plats": 5,
    "p_space": 10,
    "p_type": "binary",
    "steps": 50,
    "institution": "direct",
    "extremists": "yes",
    "percent_extremists": 5,
    "coalitions": 5,
    "mutations": 2,
    "search_steps": 10,
    "svd_groups": 2,
    "stop_condition": "steps",
    "seed": 1999,
}

param_3b_t1_coalition = {
    "n_comms": 100,
    "n_plats": 5,
    "p_space": 10,
    "p_type": "binary",
    "steps": 50,
    "institution": "coalition",
    "extremists": "yes",
    "percent_extremists": 5,
    "coalitions": 3,
    "mutations": 2,
    "search_steps": 10,
    "svd_groups": 2,
    "stop_condition": "steps",
    "seed": 1999,
}

# Fix: was 'algorithm' instead of 'algorithmic'
param_3c_t1_algorithmic = {
    "n_comms": 100,
    "n_plats": 5,
    "p_space": 10,
    "p_type": "binary",
    "steps": 50,
    "institution": "algorithmic",
    "extremists": "yes",
    "percent_extremists": 5,
    "coalitions": 3,
    "mutations": 2,
    "search_steps": 10,
    "svd_groups": 3,
    "stop_condition": "steps",
    "seed": 1999,
}

model_3a_t1 = MiniTiebout(param_3a_t1_direct)
model_3b_t1 = MiniTiebout(param_3b_t1_coalition)
model_3c_t1 = MiniTiebout(param_3c_t1_algorithmic)

results_3a_t1 = model_3a_t1.run()
results_3b_t1 = model_3b_t1.run()
results_3c_t1 = model_3c_t1.run()

print("Experiment 3a direct:", results_3a_t1.reporters)
print("Experiment 3b coalition:", results_3b_t1.reporters)
print("Experiment 3c algorithmic:", results_3c_t1.reporters)

### Experiment 3b: mixed institutions, multiple platforms + extremists

param_3b_mixed = {
    "n_comms": 900,
    "n_plats": 3,
    "p_space": 10,
    "p_type": "binary",
    "steps": 50,
    "institution": "mixed",
    "extremists": "yes",
    "percent_extremists": 10,
    "coalitions": 3,
    "mutations": 2,
    "search_steps": 10,
    "svd_groups": 3,
    "stop_condition": "steps",
    "seed": 1999,
}

model_3b = MiniTiebout(param_3b_mixed)
results_3b = model_3b.run()
print("Experiment 3b mixed:", model_3b.reporters)
