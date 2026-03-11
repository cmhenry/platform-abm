"""
Burst-based raiding cycle analysis for extremist outflow series.

Replaces ACF-based cycle detection with burst statistics that better capture
the bursty, aperiodic raiding pattern observed in simulation data.

Usage:
    # From existing raiding.json:
    import json
    with open('raiding.json') as f:
        raiding_data = json.load(f)
    
    results = analyze_all_platforms(raiding_data, burst_threshold=10)
    print_burst_report(results)

    # Or from a MovementAnalyzer outflow series directly:
    from burst_analysis import analyze_bursts
    result = analyze_bursts(outflow_series, burst_threshold=10)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def analyze_bursts(
    outflow_series: list[float] | NDArray[np.float64],
    burst_threshold: float = 10.0,
) -> dict[str, Any]:
    """Analyze burst patterns in an extremist outflow series.

    Args:
        outflow_series: Per-step count of extremists leaving a platform.
        burst_threshold: Minimum outflow to count as a burst event.
            Default 10 filters out noise from individual community movements.

    Returns:
        Dict with burst statistics:
            n_bursts: number of burst events
            burst_sizes: list of burst magnitudes
            burst_steps: list of steps where bursts occurred
            burst_intervals: list of inter-burst intervals (steps between bursts)
            mean_burst_size: average burst magnitude (NaN if no bursts)
            max_burst_size: largest single burst
            median_burst_size: median burst magnitude
            mean_interval: average steps between bursts (NaN if < 2 bursts)
            median_interval: median inter-burst interval
            total_outflow: sum of all outflow (burst and non-burst)
            burst_outflow: sum of outflow in burst events only
            burst_fraction: proportion of total outflow occurring in bursts
            escalation_slope: OLS slope of burst size on burst index
                Positive = bursts getting larger over time.
                NaN if < 2 bursts.
            escalation_r2: R² of the escalation regression
            has_bursts: True if at least one burst detected
            has_escalation: True if escalation_slope > 0 and R² > 0.1
    """
    series = np.asarray(outflow_series, dtype=np.float64)
    n_steps = len(series)

    # Identify burst events
    burst_mask = series >= burst_threshold
    burst_steps = np.where(burst_mask)[0].tolist()
    burst_sizes = series[burst_mask].tolist()
    n_bursts = len(burst_steps)

    # Inter-burst intervals
    if n_bursts >= 2:
        intervals = [
            burst_steps[i + 1] - burst_steps[i]
            for i in range(n_bursts - 1)
        ]
    else:
        intervals = []

    # Total and burst outflow
    total_outflow = float(np.sum(series))
    burst_outflow = float(np.sum(series[burst_mask])) if n_bursts > 0 else 0.0
    burst_fraction = burst_outflow / total_outflow if total_outflow > 0 else 0.0

    # Escalation: regress burst size on burst index
    if n_bursts >= 2:
        x = np.arange(n_bursts, dtype=np.float64)
        y = np.array(burst_sizes)
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        slope = float('nan')
        intercept = float('nan')
        r2 = float('nan')

    return {
        'n_bursts': n_bursts,
        'burst_sizes': burst_sizes,
        'burst_steps': burst_steps,
        'burst_intervals': intervals,
        'mean_burst_size': float(np.mean(burst_sizes)) if n_bursts > 0 else float('nan'),
        'max_burst_size': float(np.max(burst_sizes)) if n_bursts > 0 else float('nan'),
        'median_burst_size': float(np.median(burst_sizes)) if n_bursts > 0 else float('nan'),
        'mean_interval': float(np.mean(intervals)) if intervals else float('nan'),
        'median_interval': float(np.median(intervals)) if intervals else float('nan'),
        'min_interval': float(np.min(intervals)) if intervals else float('nan'),
        'max_interval': float(np.max(intervals)) if intervals else float('nan'),
        'total_outflow': total_outflow,
        'burst_outflow': burst_outflow,
        'burst_fraction': burst_fraction,
        'escalation_slope': float(slope),
        'escalation_r2': float(r2),
        'has_bursts': n_bursts > 0,
        'has_escalation': n_bursts >= 2 and slope > 0 and r2 > 0.1,
        'n_steps': n_steps,
    }


def classify_platform(
    burst_result: dict[str, Any],
    governance_type: str | None = None,
) -> str:
    """Classify a platform's raiding behavior based on burst analysis.

    Returns one of:
        'raiding_base': direct platform with bursty outflow and escalation
        'raiding_stable': direct platform with bursts but no escalation
        'enclave': coalition platform with minimal outflow (settled extremists)
        'absorber': algorithmic platform (typically receives raids, low outflow)
        'quiet': minimal extremist activity
        'active': non-bursty but sustained outflow
    """
    if burst_result['total_outflow'] < 5:
        return 'quiet'

    if not burst_result['has_bursts']:
        if burst_result['total_outflow'] > 20:
            return 'active'
        return 'quiet'

    if burst_result['has_escalation']:
        return 'raiding_base'

    if burst_result['n_bursts'] >= 3:
        return 'raiding_stable'

    if burst_result['burst_fraction'] > 0.8:
        # Most outflow is in a small number of bursts — looks like enclave disruption
        return 'enclave'

    return 'active'


def analyze_all_platforms(
    raiding_data: dict[str, dict[str, Any]],
    burst_threshold: float = 10.0,
) -> dict[str, dict[str, Any]]:
    """Analyze all platforms from a raiding.json file.

    Args:
        raiding_data: Dict keyed by platform ID string, each containing
            'outflow_series' (list of floats).
        burst_threshold: Minimum outflow to count as a burst.

    Returns:
        Dict keyed by platform ID with burst analysis results plus classification.
    """
    results = {}
    for pid, pdata in raiding_data.items():
        series = pdata.get('outflow_series', [])
        if not series:
            continue
        burst_result = analyze_bursts(series, burst_threshold=burst_threshold)
        burst_result['classification'] = classify_platform(burst_result)
        burst_result['platform_id'] = pid
        # Preserve original ACF data if present
        if 'acf' in pdata:
            burst_result['acf_original'] = pdata['acf']
        if 'has_cycle' in pdata:
            burst_result['acf_had_cycle'] = pdata['has_cycle']
        results[pid] = burst_result
    return results


def summarize_system(
    platform_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Produce system-level summary statistics from per-platform burst analyses.

    Returns:
        Dict with system-level metrics for the manuscript.
    """
    classifications = [r['classification'] for r in platform_results.values()]
    has_bursts = [r for r in platform_results.values() if r['has_bursts']]
    has_escalation = [r for r in platform_results.values() if r.get('has_escalation', False)]

    all_burst_sizes = []
    all_intervals = []
    all_escalation_slopes = []
    for r in has_bursts:
        all_burst_sizes.extend(r['burst_sizes'])
        all_intervals.extend(r['burst_intervals'])
        if r.get('has_escalation', False):
            all_escalation_slopes.append(r['escalation_slope'])

    return {
        'n_platforms': len(platform_results),
        'n_with_bursts': len(has_bursts),
        'n_with_escalation': len(has_escalation),
        'burst_rate': len(has_bursts) / len(platform_results) if platform_results else 0,
        'classification_counts': {
            c: classifications.count(c)
            for c in set(classifications)
        },
        # Burst size statistics across all platforms
        'mean_burst_size': float(np.mean(all_burst_sizes)) if all_burst_sizes else float('nan'),
        'median_burst_size': float(np.median(all_burst_sizes)) if all_burst_sizes else float('nan'),
        'max_burst_size': float(np.max(all_burst_sizes)) if all_burst_sizes else float('nan'),
        # Inter-burst interval statistics
        'mean_interval': float(np.mean(all_intervals)) if all_intervals else float('nan'),
        'median_interval': float(np.median(all_intervals)) if all_intervals else float('nan'),
        # Escalation statistics
        'mean_escalation_slope': float(np.mean(all_escalation_slopes)) if all_escalation_slopes else float('nan'),
        # ACF comparison (how many did the old method catch?)
        'n_acf_detected': sum(
            1 for r in platform_results.values()
            if r.get('acf_had_cycle', False)
        ),
    }


def print_burst_report(
    platform_results: dict[str, dict[str, Any]],
    file=None,
) -> None:
    """Print a human-readable burst analysis report."""
    out = file or sys.stdout

    summary = summarize_system(platform_results)

    print(f"\n{'='*70}", file=out)
    print(f"  BURST RAIDING ANALYSIS", file=out)
    print(f"{'='*70}", file=out)
    print(f"  Platforms analyzed:    {summary['n_platforms']}", file=out)
    print(f"  With burst activity:   {summary['n_with_bursts']} ({summary['burst_rate']:.0%})", file=out)
    print(f"  With escalation:       {summary['n_with_escalation']}", file=out)
    print(f"  ACF method detected:   {summary['n_acf_detected']} (comparison)", file=out)
    print(f"\n  Classifications:", file=out)
    for cls, count in sorted(summary['classification_counts'].items()):
        print(f"    {cls:20s}: {count}", file=out)

    if summary['n_with_bursts'] > 0:
        print(f"\n  System-wide burst statistics:", file=out)
        print(f"    Mean burst size:     {summary['mean_burst_size']:.1f}", file=out)
        print(f"    Median burst size:   {summary['median_burst_size']:.1f}", file=out)
        print(f"    Max burst size:      {summary['max_burst_size']:.0f}", file=out)
        print(f"    Mean inter-burst:    {summary['mean_interval']:.1f} steps", file=out)
        print(f"    Median inter-burst:  {summary['median_interval']:.1f} steps", file=out)

    print(f"\n{'='*70}", file=out)
    print(f"  PER-PLATFORM DETAILS", file=out)
    print(f"{'='*70}", file=out)

    # Sort by total outflow descending
    sorted_platforms = sorted(
        platform_results.items(),
        key=lambda x: x[1]['total_outflow'],
        reverse=True,
    )

    for pid, r in sorted_platforms:
        print(f"\n  Platform {pid} [{r['classification'].upper()}]", file=out)
        print(f"    Total outflow:     {r['total_outflow']:.0f}", file=out)
        print(f"    Burst events:      {r['n_bursts']}", file=out)

        if r['n_bursts'] > 0:
            print(f"    Burst sizes:       {[int(s) for s in r['burst_sizes']]}", file=out)
            print(f"    Burst steps:       {r['burst_steps']}", file=out)
            print(f"    Mean burst size:   {r['mean_burst_size']:.1f}", file=out)
            print(f"    Max burst size:    {r['max_burst_size']:.0f}", file=out)
            print(f"    Burst fraction:    {r['burst_fraction']:.1%} of total outflow", file=out)

            if r['n_bursts'] >= 2:
                print(f"    Mean interval:     {r['mean_interval']:.1f} steps", file=out)
                print(f"    Escalation slope:  {r['escalation_slope']:.2f} (R²={r['escalation_r2']:.3f})", file=out)
                if r['has_escalation']:
                    print(f"    ** ESCALATING **", file=out)

        acf_status = "detected" if r.get('acf_had_cycle', False) else "missed"
        print(f"    ACF method:        {acf_status}", file=out)


def main():
    """CLI entry point: analyze raiding.json files."""
    if len(sys.argv) < 2:
        print("Usage: python burst_analysis.py <raiding.json> [more files...]")
        print("       python burst_analysis.py <directory>/")
        sys.exit(1)

    files = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            files.extend(sorted(p.glob('*raiding*.json')))
        elif p.is_file():
            files.append(p)

    threshold = 10.0
    # Check for --threshold flag
    for i, arg in enumerate(sys.argv):
        if arg == '--threshold' and i + 1 < len(sys.argv):
            threshold = float(sys.argv[i + 1])

    for filepath in files:
        print(f"\n{'#'*70}")
        print(f"  File: {filepath}")
        print(f"  Burst threshold: {threshold}")
        print(f"{'#'*70}")

        with open(filepath) as f:
            raiding_data = json.load(f)

        results = analyze_all_platforms(raiding_data, burst_threshold=threshold)
        print_burst_report(results)

        # Save results as JSON
        output_path = filepath.with_name(
            filepath.stem.replace('raiding', 'burst_analysis') + '.json'
        )
        # Convert for JSON serialization
        serializable = {}
        for pid, r in results.items():
            sr = {k: v for k, v in r.items()}
            # Remove numpy arrays and large lists from summary output
            sr.pop('burst_sizes', None)
            sr.pop('burst_steps', None)
            sr.pop('burst_intervals', None)
            sr.pop('acf_original', None)
            # Convert any remaining numpy types
            for k, v in sr.items():
                if hasattr(v, 'item'):
                    sr[k] = v.item()
            serializable[pid] = sr

        serializable['_system_summary'] = summarize_system(results)
        # Clean summary too
        for k, v in serializable['_system_summary'].items():
            if hasattr(v, 'item'):
                serializable['_system_summary'][k] = v.item()

        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"\n  Results saved: {output_path}")


if __name__ == '__main__':
    main()