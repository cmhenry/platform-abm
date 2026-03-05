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


# ---------------------------------------------------------------------------
# Phase 2: Per-config aggregation
# ---------------------------------------------------------------------------

def run_phase2_convergence(config_dir: Path) -> dict:
    """Task 2.1: Aggregate convergence across iterations."""
    with open(config_dir / "per_iter_convergence.json") as f:
        per_iter = json.load(f)

    patterns = [v['pattern'] for v in per_iter.values()]
    settling = [v['settling_step'] for v in per_iter.values() if v.get('settling_step') is not None]
    finals = [v['final_utility'] for v in per_iter.values()]
    gains = [v['second_half_gain'] for v in per_iter.values()
             if v.get('second_half_gain') is not None]

    pattern_counts: dict[str, int] = {}
    for p in patterns:
        pattern_counts[p] = pattern_counts.get(p, 0) + 1

    # Per-governance aggregation
    gov_patterns: dict[str, dict[str, int]] = {}
    for v in per_iter.values():
        for gov, gdata in v.get('per_governance', {}).items():
            if gov not in gov_patterns:
                gov_patterns[gov] = {}
            p = gdata['pattern']
            gov_patterns[gov][p] = gov_patterns[gov].get(p, 0) + 1

    result = {
        'n_iterations': len(per_iter),
        'pattern_counts': pattern_counts,
        'settling_step_mean': float(np.mean(settling)) if settling else None,
        'settling_step_sd': float(np.std(settling, ddof=1)) if len(settling) > 1 else None,
        'final_utility_mean': float(np.mean(finals)),
        'final_utility_sd': float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0,
        'second_half_gain_mean': float(np.mean(gains)) if gains else None,
        'second_half_gain_sd': float(np.std(gains, ddof=1)) if len(gains) > 1 else None,
        'per_governance_pattern_counts': gov_patterns,
    }

    with open(config_dir / "convergence_aggregate.json", 'w') as f:
        json.dump(result, f, indent=2)
    return result


def run_phase2_burst(config_dir: Path) -> dict:
    """Task 2.2: Aggregate burst statistics across iterations.

    CRITICAL: Only pool escalation slopes from platforms with n_bursts >= 3.
    Platforms with exactly 2 bursts have mechanical R²=1.0 (two-point regression).
    """
    with open(config_dir / "per_iter_burst_analysis.json") as f:
        per_iter = json.load(f)

    all_burst_sizes: list[float] = []
    all_intervals: list[float] = []
    all_slopes: list[float] = []  # ONLY from platforms with n_bursts >= 3
    classifications: list[str] = []
    n_platform_iters = 0

    for iter_data in per_iter.values():
        for pid, pdata in iter_data.items():
            n_platform_iters += 1
            classifications.append(pdata.get('classification', 'unknown'))
            if pdata.get('has_bursts'):
                all_burst_sizes.extend(pdata.get('burst_sizes', []))
                all_intervals.extend(pdata.get('burst_intervals', []))
                # CRITICAL FILTER: only n_bursts >= 3 for escalation slopes
                if pdata.get('n_bursts', 0) >= 3:
                    slope = pdata.get('escalation_slope')
                    if slope is not None and not (isinstance(slope, float) and math.isnan(slope)):
                        all_slopes.append(slope)

    # Classification proportions
    class_counts: dict[str, int] = {}
    for c in classifications:
        class_counts[c] = class_counts.get(c, 0) + 1
    class_props = (
        {k: v / n_platform_iters for k, v in class_counts.items()}
        if n_platform_iters > 0 else {}
    )

    # Escalation t-test
    from scipy.stats import ttest_1samp
    if len(all_slopes) >= 2:
        t_stat, p_value = ttest_1samp(all_slopes, 0)
        frac_positive = sum(1 for s in all_slopes if s > 0) / len(all_slopes)
    else:
        t_stat, p_value, frac_positive = float('nan'), float('nan'), float('nan')

    burst_rate = (
        sum(1 for c in classifications if c not in ('quiet',)) / n_platform_iters
        if n_platform_iters > 0 else 0.0
    )

    result = {
        'n_iterations': len(per_iter),
        'n_platform_iterations': n_platform_iters,
        'classification_proportions': class_props,
        'burst_size_mean': float(np.mean(all_burst_sizes)) if all_burst_sizes else None,
        'burst_size_median': float(np.median(all_burst_sizes)) if all_burst_sizes else None,
        'burst_size_sd': (
            float(np.std(all_burst_sizes, ddof=1)) if len(all_burst_sizes) > 1 else None
        ),
        'interval_mean': float(np.mean(all_intervals)) if all_intervals else None,
        'interval_median': float(np.median(all_intervals)) if all_intervals else None,
        'interval_sd': (
            float(np.std(all_intervals, ddof=1)) if len(all_intervals) > 1 else None
        ),
        'escalation_n_slopes': len(all_slopes),
        'escalation_mean_slope': float(np.mean(all_slopes)) if all_slopes else None,
        'escalation_sd': (
            float(np.std(all_slopes, ddof=1)) if len(all_slopes) > 1 else None
        ),
        'escalation_t_stat': _nan_to_none(t_stat),
        'escalation_p_value': _nan_to_none(p_value),
        'escalation_fraction_positive': _nan_to_none(frac_positive),
        'burst_rate': burst_rate,
    }

    with open(config_dir / "burst_aggregate.json", 'w') as f:
        json.dump(result, f, indent=2)
    return result


def run_phase2_displacement(config_dir: Path) -> dict:
    """Task 2.3: Aggregate displacement across iterations."""
    with open(config_dir / "per_iter_displacement.json") as f:
        per_iter = json.load(f)

    all_main_deltas: list[float] = []
    all_dest_algo: list[float] = []
    all_dest_coal: list[float] = []
    all_corrs: list[float] = []
    epoch_curves: dict[str, dict[str, list[float]]] = {}

    for iter_data in per_iter.values():
        if iter_data.get('n_events', 0) == 0:
            continue
        flow = iter_data.get('flow_analysis', {})
        if 'error' in flow:
            continue

        if 'mainstream_util_delta_mean' in flow:
            all_main_deltas.append(flow['mainstream_util_delta_mean'])
        if 'fraction_to_algorithmic' in flow:
            all_dest_algo.append(flow['fraction_to_algorithmic'])
        if 'fraction_to_coalition' in flow:
            all_dest_coal.append(flow['fraction_to_coalition'])
        if 'burst_displacement_correlation' in flow:
            corr = flow['burst_displacement_correlation']
            if corr is not None:
                all_corrs.append(corr)

        # Collect superposed epoch curves for averaging
        epoch = iter_data.get('superposed_epoch', {})
        if 'relative_steps' in epoch:
            steps = epoch['relative_steps']
            for metric_key in epoch:
                if metric_key in ('relative_steps', 'n_events_per_step', 'error'):
                    continue
                values = epoch[metric_key]
                if not isinstance(values, list):
                    continue
                for rs, val in zip(steps, values):
                    rs_key = str(rs)
                    if rs_key not in epoch_curves:
                        epoch_curves[rs_key] = {}
                    if metric_key not in epoch_curves[rs_key]:
                        epoch_curves[rs_key][metric_key] = []
                    epoch_curves[rs_key][metric_key].append(val)

    n_iters_with_events = len(all_main_deltas)
    total_events = sum(d.get('n_events', 0) for d in per_iter.values())

    # Average superposed epoch with SE
    superposed: dict[str, Any] = {
        'relative_steps': sorted(int(k) for k in epoch_curves.keys())
    }
    for rs in superposed['relative_steps']:
        rs_key = str(rs)
        for metric_key, vals in epoch_curves.get(rs_key, {}).items():
            if metric_key not in superposed:
                superposed[metric_key] = []
            arr = np.array(vals)
            superposed[metric_key].append(float(np.mean(arr)))
            # Add SE for _mean metrics
            se_key = metric_key.replace('_mean', '_se')
            if '_mean' in metric_key and se_key != metric_key:
                if se_key not in superposed:
                    superposed[se_key] = []
                se = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
                superposed[se_key].append(se)

    result = {
        'n_iterations': len(per_iter),
        'n_iterations_with_events': n_iters_with_events,
        'total_events': total_events,
        'destination_algorithmic_mean': float(np.mean(all_dest_algo)) if all_dest_algo else None,
        'destination_coalition_mean': float(np.mean(all_dest_coal)) if all_dest_coal else None,
        'mainstream_util_delta_mean': (
            float(np.mean(all_main_deltas)) if all_main_deltas else None
        ),
        'mainstream_util_delta_sd': (
            float(np.std(all_main_deltas, ddof=1)) if len(all_main_deltas) > 1 else None
        ),
        'mainstream_util_delta_negative_frac': (
            float(np.mean([d < 0 for d in all_main_deltas])) if all_main_deltas else None
        ),
        'burst_displacement_corr_mean': float(np.mean(all_corrs)) if all_corrs else None,
        'superposed_epoch': superposed,
    }

    with open(config_dir / "displacement_aggregate.json", 'w') as f:
        json.dump(_clean_for_json(result), f, indent=2)
    return result


def run_phase2_enclaves(config_dir: Path) -> dict:
    """Task 2.4: Aggregate enclaves across iterations."""
    with open(config_dir / "per_iter_enclave_analysis.json") as f:
        per_iter = json.load(f)

    settling_steps: list[float] = []
    mean_homogeneities: list[float] = []
    frac_disrupted: list[float] = []
    recovery_times: list[float] = []

    for iter_data in per_iter.values():
        system = iter_data.get('system', {})
        if system.get('mean_settling_step') is not None:
            settling_steps.append(system['mean_settling_step'])
        if system.get('mean_homogeneity') is not None:
            mean_homogeneities.append(system['mean_homogeneity'])
        frac_disrupted.append(system.get('fraction_with_disruptions', 0.0))
        for pdata in iter_data.get('platforms', {}).values():
            if pdata.get('mean_recovery_steps') is not None:
                recovery_times.append(pdata['mean_recovery_steps'])

    result = {
        'n_iterations': len(per_iter),
        'settling_step_mean': float(np.mean(settling_steps)) if settling_steps else None,
        'settling_step_sd': (
            float(np.std(settling_steps, ddof=1)) if len(settling_steps) > 1 else None
        ),
        'mean_homogeneity': float(np.mean(mean_homogeneities)) if mean_homogeneities else None,
        'mean_homogeneity_sd': (
            float(np.std(mean_homogeneities, ddof=1)) if len(mean_homogeneities) > 1 else None
        ),
        'fraction_disrupted': float(np.mean(frac_disrupted)) if frac_disrupted else None,
        'mean_recovery_steps': float(np.mean(recovery_times)) if recovery_times else None,
    }

    with open(config_dir / "enclave_aggregate.json", 'w') as f:
        json.dump(result, f, indent=2)
    return result


def run_phase2_summary_update(config_dir: Path) -> None:
    """Task 2.5: Update config summary.csv with reporting pipeline metrics."""
    summary_path = config_dir / "summary.csv"
    if not summary_path.exists():
        return

    rows = []
    fieldnames: list[str] = []
    with open(summary_path) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(row)

    def add_measure(name: str, value: Any) -> None:
        row_dict: dict[str, str] = {fn: '' for fn in fieldnames}
        row_dict['Measure'] = name
        row_dict['Mean'] = f'{value:.6f}' if isinstance(value, (int, float)) and value is not None else ''
        rows.append(row_dict)

    # Load aggregates and add key metrics
    for agg_file, metrics in [
        ('convergence_aggregate.json', [
            ('convergence_settling_step', 'settling_step_mean'),
        ]),
        ('burst_aggregate.json', [
            ('burst_rate', 'burst_rate'),
            ('burst_size_median', 'burst_size_median'),
            ('escalation_mean_slope', 'escalation_mean_slope'),
            ('escalation_p_value', 'escalation_p_value'),
        ]),
        ('displacement_aggregate.json', [
            ('displacement_util_delta_mean', 'mainstream_util_delta_mean'),
            ('displacement_frac_negative', 'mainstream_util_delta_negative_frac'),
        ]),
        ('enclave_aggregate.json', [
            ('enclave_mean_homogeneity', 'mean_homogeneity'),
            ('enclave_settling_step', 'settling_step_mean'),
        ]),
    ]:
        agg_path = config_dir / agg_file
        if agg_path.exists():
            with open(agg_path) as f:
                agg = json.load(f)
            for new_name, key in metrics:
                val = agg.get(key)
                add_measure(new_name, val)

    # Convergence pattern mode
    conv_path = config_dir / "convergence_aggregate.json"
    if conv_path.exists():
        with open(conv_path) as f:
            conv = json.load(f)
        pattern_counts = conv.get('pattern_counts', {})
        if pattern_counts:
            mode = max(pattern_counts, key=pattern_counts.get)
            row_dict = {fn: '' for fn in fieldnames}
            row_dict['Measure'] = 'convergence_pattern_mode'
            row_dict['Mean'] = mode
            rows.append(row_dict)

    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Phase 3: Cross-config master summaries
# ---------------------------------------------------------------------------

def run_phase3_master_summary(exp_dir: Path, experiment: str) -> None:
    """Task 3.1: Master summary CSV combining all configs with reporting metrics."""
    summary_path = exp_dir / "summary.csv"
    if not summary_path.exists():
        logger.warning("No experiment summary.csv found at %s", summary_path)
        return

    rows = []
    with open(summary_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            config_name = row.get('config_name', '')
            config_dir = exp_dir / config_name

            # Enrich with aggregate data
            for agg_file, key_map in [
                ('convergence_aggregate.json', {
                    'convergence_pattern_mode': lambda d: max(
                        d.get('pattern_counts', {}),
                        key=d.get('pattern_counts', {}).get, default='',
                    ),
                    'convergence_settling_step': lambda d: d.get('settling_step_mean'),
                }),
                ('burst_aggregate.json', {
                    'burst_rate': lambda d: d.get('burst_rate'),
                    'burst_size_median': lambda d: d.get('burst_size_median'),
                    'burst_interval_median': lambda d: d.get('interval_median'),
                    'escalation_mean_slope': lambda d: d.get('escalation_mean_slope'),
                    'escalation_p_value': lambda d: d.get('escalation_p_value'),
                }),
                ('displacement_aggregate.json', {
                    'displacement_util_delta_mean': lambda d: d.get('mainstream_util_delta_mean'),
                    'displacement_frac_negative': lambda d: d.get(
                        'mainstream_util_delta_negative_frac',
                    ),
                }),
                ('enclave_aggregate.json', {
                    'enclave_mean_homogeneity': lambda d: d.get('mean_homogeneity'),
                    'enclave_settling_step': lambda d: d.get('settling_step_mean'),
                }),
            ]:
                agg_path = config_dir / agg_file
                if agg_path.exists():
                    with open(agg_path) as af:
                        agg = json.load(af)
                    for col_name, extractor in key_map.items():
                        val = extractor(agg)
                        if isinstance(val, (int, float)) and val is not None:
                            row[col_name] = f'{val:.6f}'
                        elif val:
                            row[col_name] = val
                        else:
                            row[col_name] = ''

            rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        for r in rows[1:]:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)

        output_name = f"{experiment}_master_summary.csv"
        with open(exp_dir / output_name, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def run_phase3_burst_master(exp_dir: Path) -> None:
    """Task 3.2: Burst master CSV for all exp2 configs."""
    rows = []
    for config_dir in sorted(exp_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        agg_path = config_dir / "burst_aggregate.json"
        cfg_path = config_dir / "config.json"
        if not agg_path.exists() or not cfg_path.exists():
            continue
        with open(agg_path) as f:
            agg = json.load(f)
        with open(cfg_path) as f:
            cfg = json.load(f)

        row: dict[str, Any] = {
            'config_name': cfg.get('name', config_dir.name),
            'n_platforms': cfg.get('n_platforms'),
            'rho_e': cfg.get('rho_extremist'),
            'alpha': cfg.get('alpha'),
        }
        for key in ('burst_rate', 'burst_size_mean', 'burst_size_median', 'burst_size_sd',
                     'interval_mean', 'interval_median', 'escalation_mean_slope',
                     'escalation_p_value', 'escalation_fraction_positive',
                     'n_platform_iterations'):
            row[key] = agg.get(key, '')
        for cls, prop in agg.get('classification_proportions', {}).items():
            row[f'class_{cls}'] = prop
        rows.append(row)

    if rows:
        _write_master_csv(exp_dir / "exp2_burst_master.csv", rows)


def run_phase3_displacement_master(exp_dir: Path) -> None:
    """Task 3.3: Displacement master CSV for all exp2 configs."""
    rows = []
    for config_dir in sorted(exp_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        agg_path = config_dir / "displacement_aggregate.json"
        cfg_path = config_dir / "config.json"
        if not agg_path.exists() or not cfg_path.exists():
            continue
        with open(agg_path) as f:
            agg = json.load(f)
        with open(cfg_path) as f:
            cfg = json.load(f)

        row: dict[str, Any] = {
            'config_name': cfg.get('name', config_dir.name),
            'n_platforms': cfg.get('n_platforms'),
            'rho_e': cfg.get('rho_extremist'),
            'alpha': cfg.get('alpha'),
        }
        for key in ('n_iterations_with_events', 'total_events',
                     'destination_algorithmic_mean', 'destination_coalition_mean',
                     'mainstream_util_delta_mean', 'mainstream_util_delta_sd',
                     'mainstream_util_delta_negative_frac',
                     'burst_displacement_corr_mean'):
            row[key] = agg.get(key, '')
        rows.append(row)

    if rows:
        _write_master_csv(exp_dir / "exp2_displacement_master.csv", rows)


def run_phase3_enclave_master(exp_dir: Path) -> None:
    """Task 3.4: Enclave master CSV for all exp2 configs."""
    rows = []
    for config_dir in sorted(exp_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        agg_path = config_dir / "enclave_aggregate.json"
        cfg_path = config_dir / "config.json"
        if not agg_path.exists() or not cfg_path.exists():
            continue
        with open(agg_path) as f:
            agg = json.load(f)
        with open(cfg_path) as f:
            cfg = json.load(f)

        row: dict[str, Any] = {
            'config_name': cfg.get('name', config_dir.name),
            'n_platforms': cfg.get('n_platforms'),
            'rho_e': cfg.get('rho_extremist'),
            'alpha': cfg.get('alpha'),
        }
        for key in ('settling_step_mean', 'settling_step_sd',
                     'mean_homogeneity', 'mean_homogeneity_sd',
                     'fraction_disrupted', 'mean_recovery_steps'):
            row[key] = agg.get(key, '')
        rows.append(row)

    if rows:
        _write_master_csv(exp_dir / "exp2_enclave_master.csv", rows)


def run_phase3_factorial_tables(exp_dir: Path) -> None:
    """Task 3.5: Factorial tables (N_p x alpha) at each rho_e level."""
    master_path = exp_dir / "exp2_master_summary.csv"
    if not master_path.exists():
        return

    df = pd.read_csv(master_path)
    required = {'rho_extremist', 'n_platforms', 'alpha'}
    if not required.issubset(df.columns):
        return

    tables_dir = exp_dir / "factorial_tables"
    tables_dir.mkdir(exist_ok=True)

    for metric in ('burst_rate', 'escalation_mean_slope', 'displacement_util_delta_mean',
                    'enclave_mean_homogeneity', 'convergence_settling_step'):
        if metric not in df.columns:
            continue
        for rho_val in sorted(df['rho_extremist'].unique()):
            sub = df[df['rho_extremist'] == rho_val]
            try:
                pivot = sub.pivot_table(
                    values=metric, index='n_platforms', columns='alpha', aggfunc='first',
                )
                rho_str = str(rho_val).replace('.', '')
                pivot.to_csv(tables_dir / f"{metric}_rho{rho_str}.csv")
            except Exception:
                pass


def run_phase3_anova(exp_dir: Path) -> None:
    """Task 3.6: Two-way ANOVA on iteration-level mainstream utility."""
    rows = []
    for config_dir in sorted(exp_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        raw_path = config_dir / "raw.csv"
        cfg_path = config_dir / "config.json"
        if not raw_path.exists() or not cfg_path.exists():
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        rho = cfg.get('rho_extremist', 0)
        n_p = cfg.get('n_platforms')
        alpha = cfg.get('alpha')

        with open(raw_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = row.get('norm_utility_mainstream')
                if val:
                    rows.append({
                        'rho_e': rho, 'n_platforms': n_p, 'alpha': alpha,
                        'norm_utility_mainstream': float(val),
                    })

    if not rows:
        return

    df = pd.DataFrame(rows)
    results = {}

    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols

        for rho_val in sorted(df['rho_e'].unique()):
            sub = df[df['rho_e'] == rho_val]
            if len(sub['n_platforms'].unique()) < 2 or len(sub['alpha'].unique()) < 2:
                continue
            model = ols('norm_utility_mainstream ~ C(n_platforms) * C(alpha)', data=sub).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

            rho_key = f"rho_{str(rho_val).replace('.', '')}"
            results[rho_key] = {
                'N_p_F': float(anova_table.loc['C(n_platforms)', 'F']),
                'N_p_p': float(anova_table.loc['C(n_platforms)', 'PR(>F)']),
                'alpha_F': float(anova_table.loc['C(alpha)', 'F']),
                'alpha_p': float(anova_table.loc['C(alpha)', 'PR(>F)']),
                'interaction_F': float(anova_table.loc['C(n_platforms):C(alpha)', 'F']),
                'interaction_p': float(anova_table.loc['C(n_platforms):C(alpha)', 'PR(>F)']),
            }
    except ImportError:
        logger.warning("statsmodels not available; skipping ANOVA")
        return

    if results:
        with open(exp_dir / "exp2_anova_results.json", 'w') as f:
            json.dump(results, f, indent=2)
