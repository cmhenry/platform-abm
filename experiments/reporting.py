"""Reporting pipeline for Experiments 1 & 2.

Phases:
  1. Per-iteration analysis (convergence, burst, displacement, enclaves)
  2. Per-config aggregation (pool across iterations, compute statistics)
  3. Cross-config master summaries (combine configs, ANOVA, factorial tables)
  4. Visualization-ready data extraction (CSV/JSON for figures)
"""
from __future__ import annotations

import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Analysis modules from results/ are imported lazily since the directory
# is gitignored and may not exist in all environments (e.g. worktrees).
_RESULTS_DIR = str(Path(__file__).resolve().parent.parent / "results")


def _ensure_results_imports() -> None:
    """Add results/ to sys.path if needed for burst_analysis and displacement_diagnostic."""
    if _RESULTS_DIR not in sys.path:
        sys.path.insert(0, _RESULTS_DIR)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nan_to_none(val: Any) -> Any:
    """Convert NaN/inf to None for JSON serialization."""
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    if hasattr(val, 'item'):
        val = val.item()
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
    return val


def _clean_for_json(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _write_master_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of dicts to CSV with union of all keys."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _step_log_to_dataframe(step_log: list[dict]) -> pd.DataFrame:
    """Flatten a single iteration's step_log into a DataFrame.

    Produces columns matching displacement_diagnostic expectations:
      step, per_governance_community_count_{gov},
      per_governance_utilities_{gov}, per_type_utility_{type}, etc.
    """
    rows = []
    for entry in step_log:
        row: dict[str, Any] = {'step': entry['step']}
        # Flatten nested dicts
        for nested_key in ('per_governance_utilities', 'per_governance_community_count',
                           'per_type_utility', 'per_type_relocations'):
            if nested_key in entry and isinstance(entry[nested_key], dict):
                for sub_key, value in entry[nested_key].items():
                    row[f'{nested_key}_{sub_key}'] = value
        # Flat scalars
        for key in ('avg_utility', 'n_relocations'):
            if key in entry:
                row[key] = entry[key]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase 1: Per-iteration analysis
# ---------------------------------------------------------------------------

def _classify_convergence(utility_series: list[float], reloc_series: list[float]) -> dict:
    """Classify convergence pattern for a single iteration's trajectory."""
    util = np.array(utility_series)
    reloc = np.array(reloc_series)
    n = len(util)
    if n < 5:
        return {'pattern': 'INSUFFICIENT_DATA'}

    tail_start = int(n * 0.8)
    tail_util = util[tail_start:]
    tail_steps = np.arange(len(tail_util))

    slope = float(np.polyfit(tail_steps, tail_util, 1)[0]) if len(tail_util) > 1 else 0.0
    tail_mean = np.mean(tail_util)
    tail_cv = float(np.std(tail_util) / tail_mean) if tail_mean > 0 else 0.0

    if len(tail_util) > 2:
        diffs = np.diff(tail_util)
        if len(diffs) > 1 and np.std(diffs[:-1]) > 0 and np.std(diffs[1:]) > 0:
            autocorr = float(np.corrcoef(diffs[:-1], diffs[1:])[0, 1])
        else:
            autocorr = 0.0
    else:
        autocorr = 0.0

    if abs(slope) < 0.001 and tail_cv < 0.005:
        pattern = "CONVERGED"
    elif slope > 0.001:
        pattern = "STILL_CLIMBING"
    elif tail_cv > 0.02 and autocorr < -0.3:
        pattern = "OSCILLATING"
    elif tail_cv > 0.01:
        pattern = "NOISY_PLATEAU"
    else:
        pattern = "PLATEAU"

    # Settling step: first step after which utility stays within 1% of final
    final_util = float(util[-1])
    threshold = 0.01 * abs(final_util) if final_util != 0 else 0.01
    settling_step = n  # default: never settled
    for i in range(n):
        if all(abs(util[j] - final_util) < threshold for j in range(i, n)):
            settling_step = i + 1  # 1-based
            break

    mid = n // 2
    return {
        'pattern': pattern,
        'settling_step': settling_step,
        'final_utility': float(util[-1]),
        'half_utility': float(util[mid]),
        'second_half_gain': float(util[-1] - util[mid]),
        'final_relocations_per_step': float(reloc[-1]),
    }


def run_phase1_convergence(config_dir: Path, step_metrics: dict) -> dict:
    """Task 1.1: Per-iteration convergence analysis."""
    results = {}
    for iter_key, step_log in step_metrics.items():
        utilities = [s['avg_utility'] for s in step_log]
        relocations = [s['n_relocations'] for s in step_log]
        conv = _classify_convergence(utilities, relocations)

        # Per-governance convergence
        gov_types: set[str] = set()
        for s in step_log:
            if 'per_governance_utilities' in s:
                gov_types.update(s['per_governance_utilities'].keys())
        per_gov = {}
        for gov in sorted(gov_types):
            gov_utils = [s.get('per_governance_utilities', {}).get(gov, 0.0) for s in step_log]
            gov_relocs = [0.0] * len(step_log)
            gc = _classify_convergence(gov_utils, gov_relocs)
            per_gov[gov] = {'pattern': gc['pattern'], 'final_utility': gc['final_utility']}
        conv['per_governance'] = per_gov
        results[iter_key] = conv

    output_path = config_dir / "per_iter_convergence.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    return results


def run_phase1_burst(config_dir: Path, per_iter_raiding: dict) -> dict | None:
    """Task 1.2: Per-iteration burst analysis (exp2 only).

    Retains burst_steps and burst_sizes (NOT stripped like CLI main()).
    """
    _ensure_results_imports()
    from burst_analysis import analyze_all_platforms  # noqa: E402

    results = {}
    for iter_key, raiding_data in per_iter_raiding.items():
        # Convert to format expected by analyze_all_platforms
        formatted = {
            pid: {'outflow_series': series}
            for pid, series in raiding_data.items()
        }
        burst_results = analyze_all_platforms(formatted, burst_threshold=10.0)

        # Convert numpy types for JSON serialization, retain all fields
        clean = {}
        for pid, r in burst_results.items():
            clean[pid] = {
                k: (v.tolist() if hasattr(v, 'tolist') else
                    v.item() if hasattr(v, 'item') else v)
                for k, v in r.items()
            }
        results[iter_key] = clean

    output_path = config_dir / "per_iter_burst_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(results, f)
    return results


def run_phase1_displacement(
    config_dir: Path,
    per_iter_burst: dict,
    step_metrics: dict,
) -> dict | None:
    """Task 1.3: Per-iteration displacement analysis (exp2 only)."""
    _ensure_results_imports()
    from displacement_diagnostic import run_displacement_analysis  # noqa: E402

    results = {}
    for iter_key in per_iter_burst:
        burst_data = per_iter_burst[iter_key]
        step_log = step_metrics.get(iter_key, [])
        if not step_log:
            continue

        stepwise_df = _step_log_to_dataframe(step_log)

        try:
            disp = run_displacement_analysis(stepwise_df, burst_data)
        except Exception as e:
            logger.warning("Displacement analysis failed for iter %s: %s", iter_key, e)
            disp = {'n_events': 0, 'error': str(e)}

        results[iter_key] = _clean_for_json(disp)

    output_path = config_dir / "per_iter_displacement.json"
    with open(output_path, 'w') as f:
        json.dump(results, f)
    return results


def run_phase1_enclaves(config_dir: Path, per_iter_enclaves: dict) -> dict | None:
    """Task 1.4: Per-iteration enclave analysis (exp2 only).

    Computes settling step, disruption events, and recovery times per coalition platform.
    """
    results = {}
    for iter_key, enclave_data in per_iter_enclaves.items():
        platforms = {}
        for pid, pdata in enclave_data.items():
            series = pdata['homogeneity_series']
            if not series:
                continue
            arr = np.array(series)
            n = len(arr)
            mean_hom = float(np.mean(arr))
            frac_enclaved = float(np.mean(arr > 0.9))

            # Settling step: first step after which homogeneity > 0.9 for 10 consecutive steps
            settling_step = None
            window = min(10, n)
            for i in range(n - window + 1):
                if all(arr[i:i + window] > 0.9):
                    settling_step = i + 1
                    break

            # Disruption events: drops below 0.9 after settling
            n_disruptions = 0
            recovery_times: list[int] = []
            if settling_step is not None:
                in_disruption = False
                disruption_start = 0
                for i in range(settling_step - 1, n):
                    if arr[i] <= 0.9 and not in_disruption:
                        in_disruption = True
                        disruption_start = i
                        n_disruptions += 1
                    elif arr[i] > 0.9 and in_disruption:
                        in_disruption = False
                        recovery_times.append(i - disruption_start)

            platforms[pid] = {
                'mean_homogeneity': mean_hom,
                'fraction_enclaved': frac_enclaved,
                'settling_step': settling_step,
                'n_disruptions': n_disruptions,
                'mean_recovery_steps': float(np.mean(recovery_times)) if recovery_times else None,
            }

        # System-level summary
        settling_steps = [
            p['settling_step'] for p in platforms.values() if p['settling_step'] is not None
        ]
        results[iter_key] = {
            'platforms': platforms,
            'system': {
                'mean_settling_step': float(np.mean(settling_steps)) if settling_steps else None,
                'mean_homogeneity': (
                    float(np.mean([p['mean_homogeneity'] for p in platforms.values()]))
                    if platforms else 0.0
                ),
                'fraction_with_disruptions': (
                    sum(1 for p in platforms.values() if p['n_disruptions'] > 0)
                    / len(platforms)
                    if platforms else 0.0
                ),
            }
        }

    output_path = config_dir / "per_iter_enclave_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    return results
