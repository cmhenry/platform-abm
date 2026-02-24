"""Smoke test utilities: reduce configs and validate outputs."""

from __future__ import annotations

import csv
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from experiments.configs.experiment_config import ExperimentConfig


def build_smoke_configs(full_configs: list[ExperimentConfig]) -> list[ExperimentConfig]:
    """Reduce configs to minimal sizes for fast validation.

    Overrides: n_communities=30, t_max=10, n_iterations=2.
    """
    smoke = []
    for cfg in full_configs:
        n_plats = min(cfg.n_platforms, 30)
        reduced = replace(
            cfg,
            n_communities=30,
            t_max=10,
            n_iterations=2,
            n_platforms=n_plats,
            # Keep svd_groups small to avoid KMeans issues with few communities per platform
            svd_groups=min(cfg.svd_groups, 2),
        )
        smoke.append(reduced)
    return smoke


def validate_smoke_results(config: ExperimentConfig, config_dir: Path) -> list[str]:
    """Validate outputs for a completed config. Returns list of failure messages."""
    failures: list[str] = []

    # 1. Output files exist
    for fname in ["config.json", "summary.csv", "raw.csv"]:
        if not (config_dir / fname).exists():
            failures.append(f"Missing file: {fname}")

    # 2. Correct row count in raw.csv
    raw_path = config_dir / "raw.csv"
    if raw_path.exists():
        with open(raw_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        expected = config.n_iterations
        if len(rows) != expected:
            failures.append(f"raw.csv: expected {expected} rows, got {len(rows)}")

        # 3. No NaN or inf in numeric columns
        numeric_cols = [
            "avg_utility", "avg_utility_mainstream", "avg_utility_extremist",
            "norm_utility_mainstream", "norm_utility_extremist",
            "total_relocations", "avg_relocations_per_community",
            "settling_time_90pct",
        ]
        for row in rows:
            for col in numeric_cols:
                val = row.get(col, "")
                if val == "":
                    continue
                try:
                    fval = float(val)
                    if math.isnan(fval) or math.isinf(fval):
                        failures.append(f"raw.csv: NaN/inf in {col} (iter {row.get('iteration')})")
                        break
                except ValueError:
                    failures.append(f"raw.csv: non-numeric value in {col}: {val}")
                    break

        # 4. norm_utility values in [0, 1]
        for row in rows:
            for col in ["norm_utility_mainstream", "norm_utility_extremist"]:
                val = row.get(col, "")
                if val == "":
                    continue
                try:
                    fval = float(val)
                    if fval < -0.01 or fval > 1.01:
                        failures.append(
                            f"raw.csv: {col}={fval:.4f} outside [0,1] "
                            f"(iter {row.get('iteration')})"
                        )
                except ValueError:
                    pass

    # 5. Summary non-empty
    summary_path = config_dir / "summary.csv"
    if summary_path.exists():
        with open(summary_path) as f:
            reader = csv.DictReader(f)
            summary_rows = list(reader)
        if len(summary_rows) == 0:
            failures.append("summary.csv is empty")

    return failures
