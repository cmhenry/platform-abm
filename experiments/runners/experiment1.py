"""Experiment 1: single/multiple platform institutional comparisons."""

from platform_abm import MiniTiebout

### Experiment 1: single platform institutional comparisons

param_1a_direct = {
    "n_comms": 100,
    "n_plats": 1,
    "p_space": 10,
    "p_type": "binary",
    "steps": 50,
    "institution": "direct",
    "extremists": "no",
    "percent_extremists": 5,
    "coalitions": 5,
    "mutations": 2,
    "search_steps": 10,
    "svd_groups": 2,
    "stop_condition": "steps",
    "seed": 1999,
}

param_1a_coalition = {
    "n_comms": 100,
    "n_plats": 1,
    "p_space": 10,
    "p_type": "binary",
    "steps": 50,
    "institution": "coalition",
    "extremists": "no",
    "percent_extremists": 5,
    "coalitions": 3,
    "mutations": 2,
    "search_steps": 10,
    "svd_groups": 2,
    "stop_condition": "steps",
    "seed": 1999,
}

param_1a_algorithmic = {
    "n_comms": 100,
    "n_plats": 1,
    "p_space": 10,
    "p_type": "binary",
    "steps": 50,
    "institution": "algorithmic",
    "extremists": "no",
    "percent_extremists": 5,
    "coalitions": 3,
    "mutations": 2,
    "search_steps": 10,
    "svd_groups": 3,
    "stop_condition": "steps",
    "seed": 1999,
}

model_1a = MiniTiebout(param_1a_direct)
model_1b = MiniTiebout(param_1a_coalition)
model_1c = MiniTiebout(param_1a_algorithmic)

results_1a = model_1a.run()
results_1b = model_1b.run()
results_1c = model_1c.run()

print("Experiment 1a (direct):", results_1a.reporters)
print("Experiment 1b (coalition):", results_1b.reporters)
print("Experiment 1c (algorithmic):", results_1c.reporters)

### Experiment 1 trial 2: multiple platform institutional comparisons

param_1b_direct = {
    "n_comms": 100,
    "n_plats": 5,
    "p_space": 25,
    "p_type": "binary",
    "steps": 50,
    "institution": "direct",
    "extremists": "no",
    "percent_extremists": 5,
    "coalitions": 5,
    "mutations": 2,
    "search_steps": 10,
    "svd_groups": 2,
    "stop_condition": "steps",
    "seed": 1999,
}

param_1b_coalition = {
    "n_comms": 100,
    "n_plats": 5,
    "p_space": 25,
    "p_type": "binary",
    "steps": 50,
    "institution": "coalition",
    "extremists": "no",
    "percent_extremists": 5,
    "coalitions": 3,
    "mutations": 2,
    "search_steps": 10,
    "svd_groups": 2,
    "stop_condition": "steps",
    "seed": 1999,
}

param_1b_algorithmic = {
    "n_comms": 100,
    "n_plats": 5,
    "p_space": 25,
    "p_type": "binary",
    "steps": 50,
    "institution": "algorithmic",
    "extremists": "no",
    "percent_extremists": 5,
    "coalitions": 3,
    "mutations": 2,
    "search_steps": 10,
    "svd_groups": 3,
    "stop_condition": "steps",
    "seed": 1999,
}

# Fix: use distinct variable names (was model_1b assigned 3 times)
model_1b_direct = MiniTiebout(param_1b_direct)
model_1b_coalition = MiniTiebout(param_1b_coalition)
model_1b_algorithmic = MiniTiebout(param_1b_algorithmic)

results_1b_direct = model_1b_direct.run()
results_1b_coalition = model_1b_coalition.run()
results_1b_algorithmic = model_1b_algorithmic.run()

print("Experiment 1b direct:", results_1b_direct.reporters)
print("Experiment 1b coalition:", results_1b_coalition.reporters)
print("Experiment 1b algorithmic:", results_1b_algorithmic.reporters)

### Experiment 1 trial 3: varying multiple platform institutional comparisons

### DIRECT
param_1c_small_direct = {
    "n_comms": 100,
    "n_plats": 5,
    "p_space": 10,
    "p_type": "binary",
    "steps": 50,
    "institution": "direct",
    "extremists": "no",
    "percent_extremists": 5,
    "coalitions": 5,
    "mutations": 2,
    "search_steps": 10,
    "svd_groups": 2,
    "stop_condition": "steps",
    "seed": 1999,
}

param_1c_mid_direct = {
    "n_comms": 200,
    "n_plats": 10,
    "p_space": 10,
    "p_type": "binary",
    "steps": 50,
    "institution": "direct",
    "extremists": "no",
    "percent_extremists": 5,
    "coalitions": 5,
    "mutations": 2,
    "search_steps": 10,
    "svd_groups": 2,
    "stop_condition": "steps",
    "seed": 1999,
}

param_1c_jumbo_direct = {
    "n_comms": 400,
    "n_plats": 20,
    "p_space": 10,
    "p_type": "binary",
    "steps": 50,
    "institution": "direct",
    "extremists": "no",
    "percent_extremists": 5,
    "coalitions": 3,
    "mutations": 2,
    "search_steps": 10,
    "svd_groups": 2,
    "stop_condition": "steps",
    "seed": 1999,
}

model_1c_small_direct = MiniTiebout(param_1c_small_direct)
model_1c_mid_direct = MiniTiebout(param_1c_mid_direct)
model_1c_jumbo_direct = MiniTiebout(param_1c_jumbo_direct)

results_1c_small_direct = model_1c_small_direct.run()
results_1c_mid_direct = model_1c_mid_direct.run()
results_1c_jumbo_direct = model_1c_jumbo_direct.run()

print("1c small direct:", results_1c_small_direct.reporters)
print("1c mid direct:", results_1c_mid_direct.reporters)
print("1c jumbo direct:", results_1c_jumbo_direct.reporters)

### COALITION
param_1c_small_coalition = {
    "n_comms": 100,
    "n_plats": 5,
    "p_space": 10,
    "p_type": "binary",
    "steps": 50,
    "institution": "coalition",
    "extremists": "no",
    "percent_extremists": 5,
    "coalitions": 5,
    "mutations": 2,
    "search_steps": 10,
    "svd_groups": 2,
    "stop_condition": "steps",
    "seed": 1999,
}

param_1c_mid_coalition = {
    "n_comms": 200,
    "n_plats": 10,
    "p_space": 10,
    "p_type": "binary",
    "steps": 50,
    "institution": "coalition",
    "extremists": "no",
    "percent_extremists": 5,
    "coalitions": 5,
    "mutations": 2,
    "search_steps": 10,
    "svd_groups": 2,
    "stop_condition": "steps",
    "seed": 1999,
}

param_1c_jumbo_coalition = {
    "n_comms": 400,
    "n_plats": 20,
    "p_space": 10,
    "p_type": "binary",
    "steps": 50,
    "institution": "coalition",
    "extremists": "no",
    "percent_extremists": 5,
    "coalitions": 3,
    "mutations": 2,
    "search_steps": 10,
    "svd_groups": 2,
    "stop_condition": "steps",
    "seed": 1999,
}

model_1c_small_coalition = MiniTiebout(param_1c_small_coalition)
model_1c_mid_coalition = MiniTiebout(param_1c_mid_coalition)
model_1c_jumbo_coalition = MiniTiebout(param_1c_jumbo_coalition)

results_1c_small_coalition = model_1c_small_coalition.run()
results_1c_mid_coalition = model_1c_mid_coalition.run()
results_1c_jumbo_coalition = model_1c_jumbo_coalition.run()

print("1c small coalition:", results_1c_small_coalition.reporters)
print("1c mid coalition:", results_1c_mid_coalition.reporters)
print("1c jumbo coalition:", results_1c_jumbo_coalition.reporters)

### ALGORITHMIC
param_1c_small_algorithmic = {
    "n_comms": 100,
    "n_plats": 5,
    "p_space": 10,
    "p_type": "binary",
    "steps": 50,
    "institution": "algorithmic",
    "extremists": "no",
    "percent_extremists": 5,
    "coalitions": 5,
    "mutations": 2,
    "search_steps": 10,
    "svd_groups": 2,
    "stop_condition": "steps",
    "seed": 1999,
}

param_1c_mid_algorithmic = {
    "n_comms": 200,
    "n_plats": 10,
    "p_space": 10,
    "p_type": "binary",
    "steps": 50,
    "institution": "algorithmic",
    "extremists": "no",
    "percent_extremists": 5,
    "coalitions": 5,
    "mutations": 2,
    "search_steps": 10,
    "svd_groups": 2,
    "stop_condition": "steps",
    "seed": 1999,
}

param_1c_jumbo_algorithmic = {
    "n_comms": 400,
    "n_plats": 20,
    "p_space": 10,
    "p_type": "binary",
    "steps": 50,
    "institution": "algorithmic",
    "extremists": "no",
    "percent_extremists": 5,
    "coalitions": 3,
    "mutations": 2,
    "search_steps": 10,
    "svd_groups": 2,
    "stop_condition": "steps",
    "seed": 1999,
}

model_1c_small_algorithmic = MiniTiebout(param_1c_small_algorithmic)
model_1c_mid_algorithmic = MiniTiebout(param_1c_mid_algorithmic)
model_1c_jumbo_algorithmic = MiniTiebout(param_1c_jumbo_algorithmic)

results_1c_small_algorithmic = model_1c_small_algorithmic.run()
results_1c_mid_algorithmic = model_1c_mid_algorithmic.run()
results_1c_jumbo_algorithmic = model_1c_jumbo_algorithmic.run()

print("1c small algorithmic:", results_1c_small_algorithmic.reporters)
print("1c mid algorithmic:", results_1c_mid_algorithmic.reporters)
print("1c jumbo algorithmic:", results_1c_jumbo_algorithmic.reporters)
