"""Sensitivity analysis: OAT + interaction effects.

Checks Exp2 baseline exists, runs OAT configs + interaction configs.
Generates sensitivity CSVs, tornado data, and interaction tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from experiments.configs.builders import (
    build_interaction_configs,
    build_oat_configs,
    _BASELINE,
)
from experiments.runner import ExperimentRunner
from experiments.tables import format_oat_table, format_interaction_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# The Exp2 config that matches the OAT baseline
_BASELINE_CONFIG_NAME = "exp2_np9_rho010_alpha5"

_OAT_OUTCOMES = [
    "avg_utility",
    "norm_utility_mainstream",
    "norm_utility_extremist",
    "total_relocations",
    "settling_time_90pct",
]


def _read_summary(summary_path: Path) -> dict[str, dict[str, str]]:
    """Read summary.csv into measure_name -> row dict."""
    result: dict[str, dict[str, str]] = {}
    if not summary_path.exists():
        return result
    with open(summary_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[row["Measure"]] = dict(row)
    return result


def _check_baseline(results_dir: Path) -> Path:
    """Check that the Exp2 baseline config exists. Returns its directory."""
    baseline_dir = results_dir / "exp2" / _BASELINE_CONFIG_NAME
    summary_path = baseline_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Exp2 baseline not found at {summary_path}. "
            f"Run Experiment 2 first (python experiments/run_exp2.py)."
        )
    return baseline_dir


def _build_oat_csv(oat_dir: Path, baseline_dir: Path) -> None:
    """Build sensitivity_oat.csv: param, test_value, outcome, baseline_value, test_result, pct_change."""
    baseline_summary = _read_summary(baseline_dir / "summary.csv")

    # Map OAT config names to (param_name, test_value)
    oat_mapping = {
        "oat_nc50": ("n_communities", "50"),
        "oat_nc200": ("n_communities", "200"),
        "oat_np3": ("n_platforms", "3"),
        "oat_np6": ("n_platforms", "6"),
        "oat_pspace5": ("p_space", "5"),
        "oat_pspace20": ("p_space", "20"),
        "oat_rho005": ("rho_extremist", "0.05"),
        "oat_rho020": ("rho_extremist", "0.20"),
        "oat_alpha2": ("alpha", "2.0"),
        "oat_alpha10": ("alpha", "10.0"),
    }

    rows: list[dict[str, Any]] = []
    for cfg_name, (param, test_val) in oat_mapping.items():
        test_summary = _read_summary(oat_dir / cfg_name / "summary.csv")
        for outcome in _OAT_OUTCOMES:
            baseline_val = float(baseline_summary.get(outcome, {}).get("Mean", "0"))
            test_result = float(test_summary.get(outcome, {}).get("Mean", "0"))
            if abs(baseline_val) < 1e-10:
                pct = 0.0
            else:
                pct = ((test_result - baseline_val) / abs(baseline_val)) * 100
            rows.append({
                "param": param,
                "test_value": test_val,
                "outcome": outcome,
                "baseline_value": f"{baseline_val:.6f}",
                "test_result": f"{test_result:.6f}",
                "pct_change": f"{pct:.2f}",
            })

    oat_csv = oat_dir / "sensitivity_oat.csv"
    with open(oat_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("OAT CSV saved to %s", oat_csv)


def _build_tornado_csv(oat_dir: Path) -> None:
    """Build sensitivity_tornado.csv: max |pct_change| per param per outcome."""
    oat_csv = oat_dir / "sensitivity_oat.csv"
    if not oat_csv.exists():
        return

    with open(oat_csv) as f:
        reader = csv.DictReader(f)
        oat_rows = list(reader)

    # Group by (param, outcome), find max absolute pct change
    groups: dict[tuple[str, str], float] = {}
    for row in oat_rows:
        key = (row["param"], row["outcome"])
        pct = abs(float(row["pct_change"]))
        if key not in groups or pct > groups[key]:
            groups[key] = pct

    tornado_rows = [
        {"param": param, "outcome": outcome, "max_abs_pct_change": f"{val:.2f}"}
        for (param, outcome), val in sorted(groups.items())
    ]

    tornado_csv = oat_dir / "sensitivity_tornado.csv"
    with open(tornado_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["param", "outcome", "max_abs_pct_change"])
        writer.writeheader()
        writer.writerows(tornado_rows)
    logger.info("Tornado CSV saved to %s", tornado_csv)


def _build_interaction_alpha_np(results_dir: Path, interact_dir: Path) -> None:
    """Build alpha_x_np.csv from Exp2 results (no new runs needed).

    Reads N_p={3,6,9} x alpha={2,5,10} at rho=0.10 from Exp2.
    """
    rows: list[dict[str, Any]] = []
    for np_val in [3, 6, 9]:
        for alpha in [2, 5, 10]:
            cfg_name = f"exp2_np{np_val}_rho010_alpha{alpha}"
            summary = _read_summary(results_dir / "exp2" / cfg_name / "summary.csv")

            row: dict[str, Any] = {
                "alpha": alpha,
                "n_platforms": np_val,
            }
            for outcome in _OAT_OUTCOMES:
                m = summary.get(outcome, {})
                row[f"{outcome}_mean"] = m.get("Mean", "")
                row[f"{outcome}_sd"] = m.get("SD", "")
            rows.append(row)

    csv_path = interact_dir / "alpha_x_np.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    logger.info("alpha x N_p interaction saved to %s", csv_path)


def _build_interaction_alpha_pspace(
    results_dir: Path, interact_dir: Path
) -> None:
    """Build alpha_x_pspace.csv combining Exp2 (p_space=10) + interaction runs.

    Reads alpha={2,5,10} x p_space={5,10,20} at N_p=9, rho=0.10.
    p_space=10 comes from Exp2; p_space={5,20} from interaction configs.
    """
    rows: list[dict[str, Any]] = []
    for alpha in [2, 5, 10]:
        for ps in [5, 10, 20]:
            if ps == 10:
                # From Exp2
                cfg_name = f"exp2_np9_rho010_alpha{alpha}"
                summary = _read_summary(results_dir / "exp2" / cfg_name / "summary.csv")
            else:
                # From interaction configs
                cfg_name = f"interact_alpha{alpha}_pspace{ps}"
                summary = _read_summary(interact_dir / cfg_name / "summary.csv")

            row: dict[str, Any] = {
                "alpha": alpha,
                "p_space": ps,
            }
            for outcome in _OAT_OUTCOMES:
                m = summary.get(outcome, {})
                row[f"{outcome}_mean"] = m.get("Mean", "")
                row[f"{outcome}_sd"] = m.get("SD", "")
            rows.append(row)

    csv_path = interact_dir / "alpha_x_pspace.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    logger.info("alpha x p_space interaction saved to %s", csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sensitivity analysis")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Print configs without running")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Max parallel workers for iterations (default: sequential)",
    )
    args = parser.parse_args()

    results_dir = Path(args.output_dir)

    # Check Exp2 baseline exists
    baseline_dir = _check_baseline(results_dir)
    logger.info("Exp2 baseline found: %s", baseline_dir)

    # Build OAT + interaction configs
    oat_configs = build_oat_configs()
    interaction_configs = build_interaction_configs()
    all_configs = oat_configs + interaction_configs

    logger.info("Sensitivity: %d OAT configs + %d interaction configs = %d total",
                len(oat_configs), len(interaction_configs), len(all_configs))

    if args.dry_run:
        for cfg in all_configs:
            print(f"  {cfg.name}: {cfg.n_communities}c, {cfg.n_platforms}p, "
                  f"p_space={cfg.p_space}, rho={cfg.rho_extremist}, "
                  f"alpha={cfg.alpha}, {cfg.n_iterations}i")
        return

    # Run OAT configs
    runner = ExperimentRunner(output_dir=args.output_dir, max_workers=args.workers)

    logger.info("--- Running OAT configs ---")
    oat_dir = results_dir / "sensitivity" / "oat"
    oat_dir.mkdir(parents=True, exist_ok=True)
    # Override experiment name to route to sensitivity/oat/
    for cfg in oat_configs:
        cfg_copy = ExperimentConfig(**{**cfg.to_dict(), "experiment": "sensitivity/oat"})
        runner.run_config(cfg_copy)

    logger.info("--- Running interaction configs ---")
    interact_dir = results_dir / "sensitivity" / "interactions"
    interact_dir.mkdir(parents=True, exist_ok=True)
    for cfg in interaction_configs:
        cfg_copy = ExperimentConfig(**{**cfg.to_dict(), "experiment": "sensitivity/interactions"})
        runner.run_config(cfg_copy)

    # Generate OAT analysis
    logger.info("--- Generating sensitivity analysis outputs ---")
    _build_oat_csv(oat_dir, baseline_dir)
    _build_tornado_csv(oat_dir)

    # Generate OAT LaTeX
    oat_latex = format_oat_table(oat_dir, baseline_dir / "summary.csv")
    with open(oat_dir / "tables.tex", "w") as f:
        f.write(oat_latex)

    # Generate interaction CSVs
    _build_interaction_alpha_np(results_dir, interact_dir)
    _build_interaction_alpha_pspace(results_dir, interact_dir)

    logger.info("Sensitivity analysis complete!")


if __name__ == "__main__":
    main()
