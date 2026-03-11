"""
Mainstream Displacement Diagnostic

Analyzes whether mainstream communities are displaced from platforms
following extremist raid arrivals, using existing per-step tracking data.

Approach:
  1. Load burst events from burst_analysis.json (outflow from direct platforms)
  2. Load per-step community counts and utilities from stepwise CSV
  3. For each burst event, construct an event window (±5 steps)
  4. Measure: did mainstream count drop and extremist count rise on the
     likely destination platform(s) in the steps following the burst?
  5. Infer flow direction from net changes across governance types

Usage:
    python displacement_diagnostic.py <stepwise.csv> <burst_analysis.json>

    Or from within the pipeline:
        from displacement_diagnostic import run_displacement_analysis
        results = run_displacement_analysis(stepwise_df, burst_data)

Input requirements:
    stepwise.csv must contain columns:
        step, avg_utility, n_relocations,
        per_governance_community_count_direct,
        per_governance_community_count_coalition,
        per_governance_community_count_algorithmic,
        per_governance_utilities_direct,
        per_governance_utilities_coalition,
        per_governance_utilities_algorithmic,
        per_type_utility_mainstream, per_type_utility_extremist
        (and optionally per_type_relocations_mainstream, per_type_relocations_extremist)

    burst_analysis.json: output from burst_analysis.py with burst_steps and
        burst_sizes per platform, plus classification field.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def extract_raid_events(
    burst_data: dict[str, Any],
    min_burst_size: int = 10,
) -> list[dict[str, Any]]:
    """Extract raid departure events from burst analysis output.

    A raid event is a burst departure from a platform classified as
    raiding_stable or raiding_base. Each event records the step,
    burst size, and source platform.

    Returns list of dicts with keys: step, size, source_platform, source_class.
    """
    events = []
    for pid, pdata in burst_data.items():
        if pid.startswith('_'):
            continue
        classification = pdata.get('classification', '')
        if classification not in ('raiding_stable', 'raiding_base', 'active'):
            continue

        # Reconstruct burst steps and sizes from the raw data
        # burst_analysis.json strips these in the summary; if present, use them
        burst_steps = pdata.get('burst_steps', [])
        burst_sizes = pdata.get('burst_sizes', [])

        if not burst_steps:
            # If burst_steps not in the JSON, we can't locate events in time
            continue

        for step, size in zip(burst_steps, burst_sizes):
            if size >= min_burst_size:
                events.append({
                    'step': int(step),
                    'size': float(size),
                    'source_platform': pid,
                    'source_class': classification,
                })

    # Sort by step
    events.sort(key=lambda e: e['step'])
    return events


def build_event_windows(
    stepwise_df: pd.DataFrame,
    raid_events: list[dict[str, Any]],
    window_before: int = 5,
    window_after: int = 5,
) -> list[dict[str, Any]]:
    """For each raid event, extract a window of stepwise metrics around it.

    Returns enriched event dicts with pre/post metrics for each governance type.
    """
    max_step = stepwise_df['step'].max()
    enriched = []

    for event in raid_events:
        t = event['step']
        t_pre_start = max(0, t - window_before)
        t_post_end = min(max_step, t + window_after)

        # Pre-raid window: steps [t - window_before, t - 1]
        pre = stepwise_df[
            (stepwise_df['step'] >= t_pre_start) &
            (stepwise_df['step'] < t)
        ]
        # Post-raid window: steps [t, t + window_after]
        post = stepwise_df[
            (stepwise_df['step'] >= t) &
            (stepwise_df['step'] <= t_post_end)
        ]

        if pre.empty or post.empty:
            continue

        result = dict(event)

        # For each governance type, compute pre/post means
        for gov in ['direct', 'coalition', 'algorithmic']:
            count_col = f'per_governance_community_count_{gov}'
            util_col = f'per_governance_utilities_{gov}'

            if count_col in stepwise_df.columns:
                result[f'{gov}_count_pre'] = pre[count_col].mean()
                result[f'{gov}_count_post'] = post[count_col].mean()
                result[f'{gov}_count_delta'] = (
                    result[f'{gov}_count_post'] - result[f'{gov}_count_pre']
                )

            if util_col in stepwise_df.columns:
                result[f'{gov}_util_pre'] = pre[util_col].mean()
                result[f'{gov}_util_post'] = post[util_col].mean()
                result[f'{gov}_util_delta'] = (
                    result[f'{gov}_util_post'] - result[f'{gov}_util_pre']
                )

        # Mainstream utility change
        if 'per_type_utility_mainstream' in stepwise_df.columns:
            result['mainstream_util_pre'] = pre['per_type_utility_mainstream'].mean()
            result['mainstream_util_post'] = post['per_type_utility_mainstream'].mean()
            result['mainstream_util_delta'] = (
                result['mainstream_util_post'] - result['mainstream_util_pre']
            )

        enriched.append(result)

    return enriched


def infer_flows(event_windows: list[dict[str, Any]]) -> dict[str, Any]:
    """Infer likely flow directions from net changes in governance counts.

    For each raid event (extremists leaving direct platforms), check:
    - Which governance type gained communities in the post window?
    - Did mainstream communities leave the destination?

    Returns aggregate flow statistics.
    """
    if not event_windows:
        return {'n_events': 0, 'error': 'no raid events found'}

    n_events = len(event_windows)

    # Track inferred destination and displacement for each event
    destinations = []
    displacements = []

    for ew in event_windows:
        # Where did communities arrive? Look for the governance type
        # with the largest positive count delta (excluding direct, which
        # is the source)
        deltas = {}
        for gov in ['coalition', 'algorithmic']:
            key = f'{gov}_count_delta'
            if key in ew:
                deltas[gov] = ew[key]

        if not deltas:
            continue

        # Primary destination: governance type with largest gain
        dest = max(deltas, key=deltas.get)
        destinations.append(dest)

        # Displacement signal: did the destination's mainstream utility drop?
        util_key = f'{dest}_util_delta'
        if util_key in ew:
            displacements.append({
                'destination': dest,
                'dest_count_delta': deltas[dest],
                'dest_util_delta': ew.get(util_key, float('nan')),
                'mainstream_util_delta': ew.get('mainstream_util_delta', float('nan')),
                'direct_count_delta': ew.get('direct_count_delta', float('nan')),
                'burst_size': ew['size'],
                'step': ew['step'],
            })

    if not displacements:
        return {'n_events': n_events, 'error': 'could not infer flows'}

    disp_df = pd.DataFrame(displacements)

    # Aggregate statistics
    dest_counts = disp_df['destination'].value_counts().to_dict()
    dest_arriving_algo = disp_df[disp_df['destination'] == 'algorithmic']
    dest_arriving_coal = disp_df[disp_df['destination'] == 'coalition']

    return {
        'n_events': n_events,
        'n_analyzed': len(displacements),

        # Where do raids land?
        'destination_counts': dest_counts,
        'fraction_to_algorithmic': dest_counts.get('algorithmic', 0) / len(displacements),
        'fraction_to_coalition': dest_counts.get('coalition', 0) / len(displacements),

        # Displacement signal: mainstream utility change after raids
        'mainstream_util_delta_mean': float(disp_df['mainstream_util_delta'].mean()),
        'mainstream_util_delta_median': float(disp_df['mainstream_util_delta'].median()),
        'mainstream_util_delta_negative_fraction': float(
            (disp_df['mainstream_util_delta'] < 0).mean()
        ),

        # Direct platform count change (should be negative — source of raids)
        'direct_count_delta_mean': float(disp_df['direct_count_delta'].mean()),

        # Destination utility change (should drop if mainstream displaced)
        'dest_util_delta_mean': float(disp_df['dest_util_delta'].mean()),

        # Burst size vs displacement correlation
        'burst_size_mean': float(disp_df['burst_size'].mean()),
        'burst_displacement_correlation': float(
            disp_df['burst_size'].corr(disp_df['mainstream_util_delta'])
        ) if len(disp_df) > 2 else float('nan'),

        # Per-event details for visualization
        'event_details': displacements,
    }


def run_displacement_analysis(
    stepwise_df: pd.DataFrame,
    burst_data: dict[str, Any],
    window_before: int = 5,
    window_after: int = 5,
    min_burst_size: int = 10,
) -> dict[str, Any]:
    """Full displacement analysis pipeline.

    Args:
        stepwise_df: DataFrame with per-step metrics (from stepwise.csv)
        burst_data: Dict from burst_analysis.json
        window_before: Steps before raid to include in pre-window
        window_after: Steps after raid to include in post-window
        min_burst_size: Minimum burst size to analyze

    Returns:
        Dict with raid events, event windows, inferred flows, and summary.
    """
    # Step 1: Extract raid events
    events = extract_raid_events(burst_data, min_burst_size=min_burst_size)

    if not events:
        return {
            'n_events': 0,
            'error': 'no raid events found in burst data',
        }

    # Step 2: Build event windows
    windows = build_event_windows(
        stepwise_df, events,
        window_before=window_before,
        window_after=window_after,
    )

    # Step 3: Infer flows and displacement
    flows = infer_flows(windows)

    # Step 4: Time-series for visualization
    # Build a "superposed epoch" — average the step metrics aligned to
    # raid events (t=0 at raid step)
    epoch_data = build_superposed_epoch(stepwise_df, events, window_before=8, window_after=8)

    return {
        'n_events': len(events),
        'n_windows': len(windows),
        'flow_analysis': flows,
        'superposed_epoch': epoch_data,
        'events': events,
    }


def build_superposed_epoch(
    stepwise_df: pd.DataFrame,
    raid_events: list[dict[str, Any]],
    window_before: int = 8,
    window_after: int = 8,
) -> dict[str, Any]:
    """Build superposed epoch analysis aligned to raid events.

    For each raid event, shift the time axis so t=0 is the raid step.
    Then average across all events to get the "typical" trajectory of
    governance counts and utilities around a raid.

    This is the data source for the primary displacement visualization.
    """
    max_step = stepwise_df['step'].max()
    epochs = []

    for event in raid_events:
        t0 = event['step']
        start = max(0, t0 - window_before)
        end = min(max_step, t0 + window_after)

        window = stepwise_df[
            (stepwise_df['step'] >= start) &
            (stepwise_df['step'] <= end)
        ].copy()

        if window.empty:
            continue

        window['relative_step'] = window['step'] - t0
        window['burst_size'] = event['size']
        epochs.append(window)

    if not epochs:
        return {'error': 'no valid epochs'}

    combined = pd.concat(epochs, ignore_index=True)

    # Average by relative step
    cols_to_avg = [c for c in combined.columns if c not in ('step', 'relative_step', 'burst_size')]
    epoch_avg = combined.groupby('relative_step')[cols_to_avg].mean()

    # Also compute std for confidence bands
    epoch_std = combined.groupby('relative_step')[cols_to_avg].std()

    # Count events per relative step (edges have fewer)
    epoch_n = combined.groupby('relative_step').size()

    result = {
        'relative_steps': epoch_avg.index.tolist(),
        'n_events_per_step': epoch_n.to_dict(),
    }

    # Extract the key series
    for gov in ['direct', 'coalition', 'algorithmic']:
        count_col = f'per_governance_community_count_{gov}'
        util_col = f'per_governance_utilities_{gov}'

        if count_col in epoch_avg.columns:
            result[f'{gov}_count_mean'] = epoch_avg[count_col].tolist()
            result[f'{gov}_count_std'] = epoch_std[count_col].tolist()
        if util_col in epoch_avg.columns:
            result[f'{gov}_util_mean'] = epoch_avg[util_col].tolist()
            result[f'{gov}_util_std'] = epoch_std[util_col].tolist()

    if 'per_type_utility_mainstream' in epoch_avg.columns:
        result['mainstream_util_mean'] = epoch_avg['per_type_utility_mainstream'].tolist()
        result['mainstream_util_std'] = epoch_std['per_type_utility_mainstream'].tolist()

    if 'per_type_utility_extremist' in epoch_avg.columns:
        result['extremist_util_mean'] = epoch_avg['per_type_utility_extremist'].tolist()
        result['extremist_util_std'] = epoch_std['per_type_utility_extremist'].tolist()

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_displacement_report(results: dict[str, Any], file=None) -> None:
    """Print human-readable displacement analysis report."""
    out = file or sys.stdout

    print(f"\n{'='*70}", file=out)
    print(f"  MAINSTREAM DISPLACEMENT DIAGNOSTIC", file=out)
    print(f"{'='*70}", file=out)

    if 'error' in results:
        print(f"  Error: {results['error']}", file=out)
        return

    print(f"  Raid events analyzed:  {results['n_events']}", file=out)
    print(f"  Valid event windows:   {results['n_windows']}", file=out)

    flows = results.get('flow_analysis', {})
    if 'error' in flows:
        print(f"  Flow analysis error: {flows['error']}", file=out)
        return

    print(f"\n  --- Raid Destinations ---", file=out)
    print(f"  Events with inferred destination:  {flows['n_analyzed']}", file=out)
    print(f"  Fraction landing on algorithmic:   {flows['fraction_to_algorithmic']:.1%}", file=out)
    print(f"  Fraction landing on coalition:     {flows['fraction_to_coalition']:.1%}", file=out)

    print(f"\n  --- Displacement Signal ---", file=out)
    print(f"  Mean mainstream utility Δ:         {flows['mainstream_util_delta_mean']:+.4f}", file=out)
    print(f"  Median mainstream utility Δ:       {flows['mainstream_util_delta_median']:+.4f}", file=out)
    print(f"  Fraction with negative Δ:          {flows['mainstream_util_delta_negative_fraction']:.1%}", file=out)

    print(f"\n  --- Source Platform ---", file=out)
    print(f"  Mean direct count Δ:               {flows['direct_count_delta_mean']:+.1f}", file=out)

    print(f"\n  --- Destination Impact ---", file=out)
    print(f"  Mean dest utility Δ:               {flows['dest_util_delta_mean']:+.4f}", file=out)

    if not np.isnan(flows.get('burst_displacement_correlation', float('nan'))):
        print(f"  Burst size ↔ displacement corr:   {flows['burst_displacement_correlation']:+.3f}", file=out)

    # Superposed epoch summary
    epoch = results.get('superposed_epoch', {})
    if 'relative_steps' in epoch and 'mainstream_util_mean' in epoch:
        steps = epoch['relative_steps']
        mu = epoch['mainstream_util_mean']
        pre_idx = [i for i, s in enumerate(steps) if s < 0]
        post_idx = [i for i, s in enumerate(steps) if s > 0]
        if pre_idx and post_idx:
            pre_mean = np.mean([mu[i] for i in pre_idx])
            post_mean = np.mean([mu[i] for i in post_idx])
            print(f"\n  --- Superposed Epoch ---", file=out)
            print(f"  Mean mainstream util (pre-raid):   {pre_mean:.4f}", file=out)
            print(f"  Mean mainstream util (post-raid):  {post_mean:.4f}", file=out)
            print(f"  Epoch displacement:                {post_mean - pre_mean:+.4f}", file=out)

    print(f"\n{'='*70}", file=out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print("Usage: python displacement_diagnostic.py <stepwise.csv> <burst_analysis.json>")
        print("\n  stepwise.csv: per-step metrics from simulation")
        print("  burst_analysis.json: output from burst_analysis.py")
        sys.exit(1)

    stepwise_path = Path(sys.argv[1])
    burst_path = Path(sys.argv[2])

    # Load data
    stepwise_df = pd.read_csv(stepwise_path)
    with open(burst_path) as f:
        burst_data = json.load(f)

    # Run analysis
    results = run_displacement_analysis(stepwise_df, burst_data)

    # Report
    print_displacement_report(results)

    # Save results
    output_path = stepwise_path.parent / 'displacement_analysis.json'

    # Clean for JSON serialization
    serializable = {
        'n_events': results['n_events'],
        'n_windows': results['n_windows'],
        'flow_analysis': {
            k: v for k, v in results['flow_analysis'].items()
            if k != 'event_details'
        },
        'event_details': results['flow_analysis'].get('event_details', []),
    }
    # Add epoch data if present
    epoch = results.get('superposed_epoch', {})
    if 'error' not in epoch:
        serializable['superposed_epoch'] = {
            k: v for k, v in epoch.items()
        }

    # Convert numpy types
    def clean_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: clean_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean_numpy(v) for v in obj]
        return obj

    serializable = clean_numpy(serializable)

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n  Results saved: {output_path}")


if __name__ == '__main__':
    main()