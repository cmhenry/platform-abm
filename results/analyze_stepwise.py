"""
Analyze per-step simulation metrics from Experiment 1 JSON files.

Produces:
1. Convergence diagnostics: is utility still climbing at t_max?
2. Per-governance utility trajectories (mean across iterations)
3. Relocation rate over time (are communities still churning?)
4. Per-governance community count over time (if available)

Usage:
    python analyze_stepwise.py path/to/exp1_direct_np9_stepwise.json [more files...]
    python analyze_stepwise.py path/to/stepwise_dir/  # processes all JSON in directory

Output:
    - Console summary of convergence diagnostics
    - PNG plots saved alongside each input file
    - CSV of per-step averages for each config
"""

import json
import sys
import os
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not available — skipping plots, producing CSV and console output only.")


def load_stepwise(filepath):
    """Load stepwise JSON. Keys are iteration indices (as strings), values are lists of step dicts."""
    with open(filepath) as f:
        data = json.load(f)
    return data


def extract_trajectories(data):
    """
    Extract per-step metrics averaged across iterations.
    
    Returns dict with:
        steps: array of step numbers
        avg_utility: mean utility per step (averaged across iterations)
        avg_utility_ci: 95% CI half-width per step
        n_relocations: mean relocations per step
        n_relocations_ci: 95% CI half-width
        per_gov_utility: {gov_type: mean utility array}
        per_gov_utility_ci: {gov_type: CI half-width array}
        n_iterations: number of iterations
    """
    # Determine number of steps from first iteration
    first_key = list(data.keys())[0]
    n_steps = len(data[first_key])
    n_iterations = len(data)
    
    # Preallocate arrays
    utility_matrix = np.zeros((n_iterations, n_steps))
    relocation_matrix = np.zeros((n_iterations, n_steps))
    
    # Discover governance types from first iteration's first step
    gov_types = list(data[first_key][0].get('per_governance_utilities', {}).keys())
    gov_matrices = {g: np.zeros((n_iterations, n_steps)) for g in gov_types}
    
    for i, (iter_key, steps) in enumerate(data.items()):
        for j, step_data in enumerate(steps):
            if j >= n_steps:
                break
            utility_matrix[i, j] = step_data['avg_utility']
            relocation_matrix[i, j] = step_data['n_relocations']
            for g in gov_types:
                gov_utils = step_data.get('per_governance_utilities', {})
                if g in gov_utils:
                    gov_matrices[g][i, j] = gov_utils[g]
                else:
                    gov_matrices[g][i, j] = np.nan
    
    steps = np.arange(1, n_steps + 1)
    
    def mean_and_ci(matrix):
        mean = np.nanmean(matrix, axis=0)
        se = np.nanstd(matrix, axis=0, ddof=1) / np.sqrt(n_iterations)
        ci = 1.96 * se
        return mean, ci
    
    avg_utility, avg_utility_ci = mean_and_ci(utility_matrix)
    n_relocations, n_relocations_ci = mean_and_ci(relocation_matrix)
    
    per_gov_utility = {}
    per_gov_utility_ci = {}
    for g in gov_types:
        per_gov_utility[g], per_gov_utility_ci[g] = mean_and_ci(gov_matrices[g])
    
    return {
        'steps': steps,
        'avg_utility': avg_utility,
        'avg_utility_ci': avg_utility_ci,
        'n_relocations': n_relocations,
        'n_relocations_ci': n_relocations_ci,
        'per_gov_utility': per_gov_utility,
        'per_gov_utility_ci': per_gov_utility_ci,
        'n_iterations': n_iterations,
        'n_steps': n_steps,
        # Raw matrices for additional analysis
        '_utility_matrix': utility_matrix,
        '_relocation_matrix': relocation_matrix,
    }


def convergence_diagnostics(trajectories):
    """
    Determine whether the system is converging, oscillating, or flat.
    
    Returns a dict with diagnostic measures.
    """
    util = trajectories['avg_utility']
    reloc = trajectories['n_relocations']
    n = len(util)
    
    # Slope of utility over last 20% of steps
    tail_start = int(n * 0.8)
    tail_util = util[tail_start:]
    tail_steps = np.arange(len(tail_util))
    
    if len(tail_util) > 1:
        slope = np.polyfit(tail_steps, tail_util, 1)[0]
    else:
        slope = 0.0
    
    # Utility at step 1, midpoint, and final step
    util_start = util[0]
    util_mid = util[n // 2]
    util_end = util[-1]
    
    # Relocations at final step (are communities still moving?)
    reloc_end = reloc[-1]
    reloc_start = reloc[0]
    
    # Oscillation: standard deviation of utility in last 20% relative to mean
    tail_cv = np.std(tail_util) / np.mean(tail_util) if np.mean(tail_util) > 0 else 0
    
    # Autocorrelation of utility differences (lag-1) in tail
    # Negative autocorrelation suggests oscillation
    if len(tail_util) > 2:
        diffs = np.diff(tail_util)
        if np.std(diffs) > 0:
            autocorr = np.corrcoef(diffs[:-1], diffs[1:])[0, 1]
        else:
            autocorr = 0.0
    else:
        autocorr = 0.0
    
    # Classification
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
    
    return {
        'pattern': pattern,
        'tail_slope': slope,
        'tail_cv': tail_cv,
        'tail_autocorr': autocorr,
        'util_start': util_start,
        'util_mid': util_mid,
        'util_end': util_end,
        'util_gain_total': util_end - util_start,
        'util_gain_second_half': util_end - util_mid,
        'reloc_start': reloc_start,
        'reloc_end': reloc_end,
        'reloc_reduction_pct': (1 - reloc_end / reloc_start) * 100 if reloc_start > 0 else 0,
    }


def print_diagnostics(config_name, diag, trajectories):
    """Print convergence diagnostics to console."""
    print(f"\n{'='*60}")
    print(f"  {config_name}")
    print(f"{'='*60}")
    print(f"  Pattern:              {diag['pattern']}")
    print(f"  Utility trajectory:   {diag['util_start']:.3f} → {diag['util_mid']:.3f} → {diag['util_end']:.3f}")
    print(f"  Total gain:           +{diag['util_gain_total']:.3f}")
    print(f"  Second-half gain:     +{diag['util_gain_second_half']:.3f}")
    print(f"  Tail slope:           {diag['tail_slope']:.5f} util/step")
    print(f"  Tail CV:              {diag['tail_cv']:.4f}")
    print(f"  Tail autocorr(Δu):   {diag['tail_autocorr']:.3f}")
    print(f"  Relocations:          {diag['reloc_start']:.0f} → {diag['reloc_end']:.0f} ({diag['reloc_reduction_pct']:.1f}% reduction)")
    print(f"  Iterations:           {trajectories['n_iterations']}")
    print(f"  Steps:                {trajectories['n_steps']}")
    
    # Per-governance summary at final step
    print(f"\n  Per-governance utility at final step:")
    for g, util_arr in trajectories['per_gov_utility'].items():
        ci = trajectories['per_gov_utility_ci'][g]
        print(f"    {g:15s}: {util_arr[-1]:.3f} ± {ci[-1]:.3f}")


def plot_trajectories(config_name, trajectories, output_dir):
    """Generate diagnostic plots."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{config_name}', fontsize=14, fontweight='bold')
    
    steps = trajectories['steps']
    
    # Plot 1: Average utility over time
    ax = axes[0, 0]
    ax.plot(steps, trajectories['avg_utility'], 'k-', linewidth=1.5, label='System avg')
    ax.fill_between(steps,
                     trajectories['avg_utility'] - trajectories['avg_utility_ci'],
                     trajectories['avg_utility'] + trajectories['avg_utility_ci'],
                     alpha=0.2, color='gray')
    ax.set_xlabel('Step')
    ax.set_ylabel('Average utility')
    ax.set_title('System-wide utility')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Per-governance utility over time
    ax = axes[0, 1]
    colors = {'direct': '#e41a1c', 'coalition': '#377eb8', 'algorithmic': '#4daf4a'}
    for g, util_arr in trajectories['per_gov_utility'].items():
        ci = trajectories['per_gov_utility_ci'][g]
        color = colors.get(g, 'gray')
        ax.plot(steps, util_arr, '-', color=color, linewidth=1.5, label=g)
        ax.fill_between(steps, util_arr - ci, util_arr + ci, alpha=0.15, color=color)
    ax.set_xlabel('Step')
    ax.set_ylabel('Average utility')
    ax.set_title('Utility by governance type')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Relocations over time
    ax = axes[1, 0]
    ax.plot(steps, trajectories['n_relocations'], 'k-', linewidth=1.5)
    ax.fill_between(steps,
                     trajectories['n_relocations'] - trajectories['n_relocations_ci'],
                     trajectories['n_relocations'] + trajectories['n_relocations_ci'],
                     alpha=0.2, color='gray')
    ax.set_xlabel('Step')
    ax.set_ylabel('Relocations')
    ax.set_title('Community relocations per step')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Utility gain rate (derivative)
    ax = axes[1, 1]
    if len(trajectories['avg_utility']) > 1:
        delta_util = np.diff(trajectories['avg_utility'])
        ax.plot(steps[1:], delta_util, 'k-', linewidth=0.8, alpha=0.5)
        # Smoothed (rolling mean, window=5)
        if len(delta_util) > 5:
            kernel = np.ones(5) / 5
            smoothed = np.convolve(delta_util, kernel, mode='valid')
            ax.plot(steps[3:3+len(smoothed)], smoothed, 'r-', linewidth=1.5, label='Smoothed (5-step)')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Δ Utility')
        ax.set_title('Utility change per step')
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{config_name}_diagnostics.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {output_path}")


def save_csv(config_name, trajectories, output_dir):
    """Save per-step averages as CSV."""
    steps = trajectories['steps']
    
    rows = []
    header = ['step', 'avg_utility', 'avg_utility_ci', 'n_relocations', 'n_relocations_ci']
    gov_types = sorted(trajectories['per_gov_utility'].keys())
    for g in gov_types:
        header.append(f'utility_{g}')
        header.append(f'utility_{g}_ci')
    
    for i, step in enumerate(steps):
        row = [
            step,
            f"{trajectories['avg_utility'][i]:.6f}",
            f"{trajectories['avg_utility_ci'][i]:.6f}",
            f"{trajectories['n_relocations'][i]:.2f}",
            f"{trajectories['n_relocations_ci'][i]:.2f}",
        ]
        for g in gov_types:
            row.append(f"{trajectories['per_gov_utility'][g][i]:.6f}")
            row.append(f"{trajectories['per_gov_utility_ci'][g][i]:.6f}")
        rows.append(row)
    
    output_path = os.path.join(output_dir, f'{config_name}_stepwise.csv')
    with open(output_path, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(str(x) for x in row) + '\n')
    print(f"  CSV saved: {output_path}")


def process_file(filepath):
    """Process a single stepwise JSON file."""
    filepath = Path(filepath)
    config_name = filepath.stem.replace('_stepwise', '').replace('_step_metrics', '')
    output_dir = str(filepath.parent)
    
    print(f"\nLoading {filepath}...")
    data = load_stepwise(filepath)
    
    trajectories = extract_trajectories(data)
    diag = convergence_diagnostics(trajectories)
    
    print_diagnostics(config_name, diag, trajectories)
    save_csv(config_name, trajectories, output_dir)
    plot_trajectories(config_name, trajectories, output_dir)
    
    return config_name, diag, trajectories


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_stepwise.py <file_or_directory> [more files...]")
        sys.exit(1)
    
    files = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            files.extend(sorted(p.glob('step_metrics.json')))
        elif p.is_file() and p.suffix == '.json':
            files.append(p)
        else:
            print(f"Skipping {arg} (not a JSON file or directory)")
    
    if not files:
        print("No JSON files found.")
        sys.exit(1)
    
    results = []
    for f in files:
        try:
            results.append(process_file(f))
        except Exception as e:
            print(f"ERROR processing {f}: {e}")
    
    # Summary comparison across configs
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"  SUMMARY ACROSS CONFIGURATIONS")
        print(f"{'='*60}")
        print(f"  {'Config':<30s} {'Pattern':<18s} {'Final util':>10s} {'Tail slope':>12s} {'Final reloc':>12s}")
        print(f"  {'-'*82}")
        for name, diag, traj in results:
            print(f"  {name:<30s} {diag['pattern']:<18s} {diag['util_end']:>10.3f} {diag['tail_slope']:>12.5f} {diag['reloc_end']:>12.0f}")


if __name__ == '__main__':
    main()