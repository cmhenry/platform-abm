"""Experiment 2: mixed institutions, multiple platforms."""

from platform_abm import MiniTiebout

### Experiment 2: mixed institutions, multiple platforms

param_2_mixed_small = {
    "n_comms": 100,
    "n_plats": 3,
    "p_space": 10,
    "p_type": "binary",
    "steps": 50,
    "institution": "mixed",
    "extremists": "no",
    "percent_extremists": 5,
    "coalitions": 3,
    "mutations": 2,
    "search_steps": 10,
    "svd_groups": 3,
    "stop_condition": "steps",
    "seed": 1999,
}

param_2_mixed_med = {
    "n_comms": 300,
    "n_plats": 9,
    "p_space": 10,
    "p_type": "binary",
    "steps": 50,
    "institution": "mixed",
    "extremists": "no",
    "percent_extremists": 5,
    "coalitions": 3,
    "mutations": 2,
    "search_steps": 10,
    "svd_groups": 3,
    "stop_condition": "steps",
    "seed": 1999,
}

param_2_mixed_large = {
    "n_comms": 900,
    "n_plats": 27,
    "p_space": 10,
    "p_type": "binary",
    "steps": 50,
    "institution": "mixed",
    "extremists": "no",
    "percent_extremists": 5,
    "coalitions": 3,
    "mutations": 2,
    "search_steps": 10,
    "svd_groups": 3,
    "stop_condition": "steps",
    "seed": 1999,
}

model_2_small = MiniTiebout(param_2_mixed_small)
model_2_med = MiniTiebout(param_2_mixed_med)
model_2_large = MiniTiebout(param_2_mixed_large)

results_2_small = model_2_small.run()
# Fix: was `esults_2_med` (missing `r`)
results_2_med = model_2_med.run()
results_2_large = model_2_large.run()

print("Experiment 2 small:", model_2_small.reporters)
print("Experiment 2 med:", model_2_med.reporters)
print("Experiment 2 large:", model_2_large.reporters)
