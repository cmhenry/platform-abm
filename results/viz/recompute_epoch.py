#!/usr/bin/env python3
"""Baseline-normalize superposed epoch data per iteration, then average.

For each config, reads per_iter_displacement.json, computes per-iteration
baseline (mean of steps t=-8 to t=-4, indices 0-4), subtracts it, then
averages the deltas across iterations with SE = std / sqrt(n).

Outputs epoch_baselined_{config}.json into results/viz/.
"""

import json
import os
import numpy as np
from pathlib import Path

CONFIGS = [
    "exp2_np3_rho015_alpha10",
    "exp2_np9_rho015_alpha10",
    "exp2_np27_rho015_alpha5",
]

SERIES = [
    "direct_count",
    "coalition_count",
    "algorithmic_count",
    "mainstream_util",
    "extremist_util",
]

BASELINE_INDICES = slice(0, 5)  # indices 0-4 → steps t=-8 to t=-4

base_dir = Path(__file__).resolve().parent.parent
exp2_dir = base_dir / "exp2"
viz_dir = base_dir / "viz"
viz_dir.mkdir(exist_ok=True)


def process_config(config: str) -> None:
    path = exp2_dir / config / "per_iter_displacement.json"
    with open(path) as f:
        data = json.load(f)

    # Collect delta arrays per series
    deltas = {s: [] for s in SERIES}

    for iter_key, iter_data in data.items():
        if iter_data["n_events"] == 0:
            continue

        epoch = iter_data["superposed_epoch"]

        for series in SERIES:
            values = np.array(epoch[f"{series}_mean"])
            baseline = np.mean(values[BASELINE_INDICES])
            deltas[series].append(values - baseline)

    n_iters = len(deltas[SERIES[0]])
    relative_steps = list(range(-8, 9))

    result = {
        "config": config,
        "n_iterations": n_iters,
        "relative_steps": relative_steps,
    }

    for series in SERIES:
        arr = np.array(deltas[series])  # shape (n_iters, 17)
        result[f"{series}_delta_mean"] = np.mean(arr, axis=0).tolist()
        result[f"{series}_delta_se"] = (np.std(arr, axis=0, ddof=1) / np.sqrt(n_iters)).tolist()

    out_path = viz_dir / f"epoch_baselined_{config}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  {config}: {n_iters} iterations → {out_path.name}")


if __name__ == "__main__":
    print("Recomputing baseline-normalized epoch data...")
    for cfg in CONFIGS:
        process_config(cfg)
    print("Done.")
