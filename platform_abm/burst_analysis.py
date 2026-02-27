"""
Burst-based raiding cycle analysis for extremist outflow series.

Replaces ACF-based cycle detection with burst statistics that better capture
the bursty, aperiodic raiding pattern observed in simulation data.
"""

from __future__ import annotations

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
            escalation_r2: R-squared of the escalation regression
            has_bursts: True if at least one burst detected
            has_escalation: True if escalation_slope > 0 and R-squared > 0.1
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
        return 'enclave'

    return 'active'
