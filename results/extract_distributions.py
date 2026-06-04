"""Extract per-iteration distributional data from existing per_iter JSON files.

Reads per_iter_burst_analysis.json, per_iter_displacement.json, and
per_iter_enclave_analysis.json for all 27 exp2 configs and produces
distributional CSVs for downstream analysis.

Usage:
    python results/extract_distributions.py
    python results/extract_distributions.py --results-dir results/exp2 --output-dir results/distributions
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


def parse_config_name(config_name: str) -> dict:
    """Parse config name like exp2_np27_rho015_alpha10 into components.

    Returns dict with n_platforms (int), rho_e (float), alpha (int).
    """
    m = re.match(r"exp2_np(\d+)_rho(\d+)_alpha(\d+)", config_name)
    if not m:
        raise ValueError(f"Cannot parse config name: {config_name!r}")
    n_platforms = int(m.group(1))
    rho_raw = int(m.group(2))
    # rho_raw is like 005 -> 0.05, 010 -> 0.10, 015 -> 0.15
    rho_e = rho_raw / 100.0
    alpha = int(m.group(3))
    return {"n_platforms": n_platforms, "rho_e": rho_e, "alpha": alpha}


def extract_escalation_slopes(
    config_name: str,
    config_meta: dict,
    burst_data: dict,
    rows: list[dict],
) -> None:
    """Extract escalation slope rows (one per platform-iteration, n_bursts >= 3)."""
    for iteration_str, platforms in burst_data.items():
        iteration = int(iteration_str)
        for platform_id, pdata in platforms.items():
            n_bursts = pdata.get("n_bursts", 0)
            if n_bursts < 3:
                continue
            rows.append({
                "config": config_name,
                "n_platforms": config_meta["n_platforms"],
                "rho_e": config_meta["rho_e"],
                "alpha": config_meta["alpha"],
                "iteration": iteration,
                "platform_id": platform_id,
                "n_bursts": n_bursts,
                "slope": pdata.get("escalation_slope", ""),
                "r2": pdata.get("escalation_r2", ""),
                "classification": pdata.get("classification", ""),
            })


def extract_burst_sizes(
    config_name: str,
    config_meta: dict,
    burst_data: dict,
    rows: list[dict],
) -> None:
    """Extract one row per burst event."""
    for iteration_str, platforms in burst_data.items():
        iteration = int(iteration_str)
        for platform_id, pdata in platforms.items():
            burst_sizes = pdata.get("burst_sizes", [])
            burst_steps = pdata.get("burst_steps", [])
            for burst_index, (bsize, bstep) in enumerate(zip(burst_sizes, burst_steps)):
                rows.append({
                    "config": config_name,
                    "n_platforms": config_meta["n_platforms"],
                    "rho_e": config_meta["rho_e"],
                    "alpha": config_meta["alpha"],
                    "iteration": iteration,
                    "platform_id": platform_id,
                    "burst_index": burst_index,
                    "burst_step": bstep,
                    "burst_size": bsize,
                })


def extract_displacement_events(
    config_name: str,
    config_meta: dict,
    displacement_data: dict,
    rows: list[dict],
) -> None:
    """Extract one row per iteration from displacement data (aggregated per iteration)."""
    for iteration_str, idata in displacement_data.items():
        iteration = int(iteration_str)
        n_events = idata.get("n_events", 0)
        flow = idata.get("flow_analysis", {})
        rows.append({
            "config": config_name,
            "n_platforms": config_meta["n_platforms"],
            "rho_e": config_meta["rho_e"],
            "alpha": config_meta["alpha"],
            "iteration": iteration,
            "n_events": n_events,
            "mainstream_util_delta_mean": flow.get("mainstream_util_delta_mean", ""),
            "mainstream_util_delta_median": flow.get("mainstream_util_delta_median", ""),
            "mainstream_util_delta_negative_fraction": flow.get(
                "mainstream_util_delta_negative_fraction", ""
            ),
            "fraction_to_algorithmic": flow.get("fraction_to_algorithmic", ""),
            "fraction_to_coalition": flow.get("fraction_to_coalition", ""),
            "burst_size_mean": flow.get("burst_size_mean", ""),
            "burst_displacement_correlation": flow.get("burst_displacement_correlation", ""),
        })


def extract_enclave_metrics(
    config_name: str,
    config_meta: dict,
    enclave_data: dict,
    rows: list[dict],
) -> None:
    """Extract one row per coalition platform-iteration from enclave data."""
    for iteration_str, idata in enclave_data.items():
        iteration = int(iteration_str)
        platforms = idata.get("platforms", {})
        for platform_id, pdata in platforms.items():
            rows.append({
                "config": config_name,
                "n_platforms": config_meta["n_platforms"],
                "rho_e": config_meta["rho_e"],
                "alpha": config_meta["alpha"],
                "iteration": iteration,
                "platform_id": platform_id,
                "mean_homogeneity": pdata.get("mean_homogeneity", ""),
                "fraction_enclaved": pdata.get("fraction_enclaved", ""),
                "settling_step": pdata.get("settling_step", ""),
                "n_disruptions": pdata.get("n_disruptions", ""),
                "mean_recovery_steps": pdata.get("mean_recovery_steps", ""),
            })


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Write rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract distributional CSVs from existing per_iter JSON files."
    )
    parser.add_argument(
        "--results-dir",
        default="results/exp2",
        help="Directory containing exp2 config subdirectories (default: results/exp2)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/distributions",
        help="Directory to write output CSVs (default: results/distributions)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    if not results_dir.exists():
        print(f"ERROR: results-dir does not exist: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect all exp2 config directories
    config_dirs = sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith("exp2_np")
    )

    if not config_dirs:
        print(f"ERROR: No exp2 config directories found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(config_dirs)} config directories in {results_dir}")

    # Accumulators
    slope_rows: list[dict] = []
    burst_rows: list[dict] = []
    displacement_rows: list[dict] = []
    enclave_rows: list[dict] = []

    for config_dir in config_dirs:
        config_name = config_dir.name
        try:
            config_meta = parse_config_name(config_name)
        except ValueError as e:
            print(f"  SKIP {config_name}: {e}")
            continue

        print(f"  Processing {config_name} (np={config_meta['n_platforms']}, "
              f"rho={config_meta['rho_e']}, alpha={config_meta['alpha']})...")

        # --- per_iter_burst_analysis.json ---
        burst_path = config_dir / "per_iter_burst_analysis.json"
        if burst_path.exists():
            with open(burst_path) as f:
                burst_data = json.load(f)
            extract_escalation_slopes(config_name, config_meta, burst_data, slope_rows)
            extract_burst_sizes(config_name, config_meta, burst_data, burst_rows)
        else:
            print(f"    WARNING: missing {burst_path.name}")

        # --- per_iter_displacement.json ---
        displacement_path = config_dir / "per_iter_displacement.json"
        if displacement_path.exists():
            with open(displacement_path) as f:
                displacement_data = json.load(f)
            extract_displacement_events(
                config_name, config_meta, displacement_data, displacement_rows
            )
        else:
            print(f"    WARNING: missing {displacement_path.name}")

        # --- per_iter_enclave_analysis.json ---
        enclave_path = config_dir / "per_iter_enclave_analysis.json"
        if enclave_path.exists():
            with open(enclave_path) as f:
                enclave_data = json.load(f)
            extract_enclave_metrics(
                config_name, config_meta, enclave_data, enclave_rows
            )
        else:
            print(f"    WARNING: missing {enclave_path.name}")

    # Write output CSVs
    output_dir.mkdir(parents=True, exist_ok=True)

    slopes_path = output_dir / "escalation_slopes.csv"
    write_csv(
        slopes_path,
        ["config", "n_platforms", "rho_e", "alpha",
         "iteration", "platform_id", "n_bursts", "slope", "r2", "classification"],
        slope_rows,
    )
    print(f"\nWrote {len(slope_rows)} rows -> {slopes_path}")

    burst_path_out = output_dir / "burst_sizes.csv"
    write_csv(
        burst_path_out,
        ["config", "n_platforms", "rho_e", "alpha",
         "iteration", "platform_id", "burst_index", "burst_step", "burst_size"],
        burst_rows,
    )
    print(f"Wrote {len(burst_rows)} rows -> {burst_path_out}")

    displacement_path_out = output_dir / "displacement_events.csv"
    write_csv(
        displacement_path_out,
        ["config", "n_platforms", "rho_e", "alpha",
         "iteration", "n_events",
         "mainstream_util_delta_mean", "mainstream_util_delta_median",
         "mainstream_util_delta_negative_fraction",
         "fraction_to_algorithmic", "fraction_to_coalition",
         "burst_size_mean", "burst_displacement_correlation"],
        displacement_rows,
    )
    print(f"Wrote {len(displacement_rows)} rows -> {displacement_path_out}")

    enclave_path_out = output_dir / "enclave_metrics.csv"
    write_csv(
        enclave_path_out,
        ["config", "n_platforms", "rho_e", "alpha",
         "iteration", "platform_id",
         "mean_homogeneity", "fraction_enclaved",
         "settling_step", "n_disruptions", "mean_recovery_steps"],
        enclave_rows,
    )
    print(f"Wrote {len(enclave_rows)} rows -> {enclave_path_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
