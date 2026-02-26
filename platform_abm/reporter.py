"""Simulation reporting: per-iteration measures, aggregation, and export."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from platform_abm.model import MiniTiebout
    from platform_abm.tracker import StepRecord


@dataclass
class IterationResult:
    """Extracted results from a single simulation run."""

    n_comms: int
    n_plats: int
    p_space: int
    institution: str
    alpha: float
    steps: int
    has_extremists: bool
    community_utilities: list[float]
    community_types: list[str]
    community_governance_types: list[str]
    community_moves: list[int]
    community_last_move_steps: list[int]
    platform_institutions: list[str]
    platform_community_counts: list[int]
    tracker_log: dict[int, StepRecord] | None = None
    step_log: list[dict] | None = None
    step_series: dict | None = None


@dataclass
class AggregatedMeasure:
    """A single aggregated statistic across iterations."""

    name: str
    mean: float
    sd: float
    ci_lower: float
    ci_upper: float
    median: float
    min: float
    max: float
    n_iterations: int


class SimulationReporter:
    """Aggregates measures across multiple simulation iterations."""

    def __init__(self) -> None:
        self._iterations: list[IterationResult] = []
        self._summary: list[AggregatedMeasure] | None = None

    @staticmethod
    def from_model(model: MiniTiebout) -> IterationResult:
        """Extract an IterationResult from a completed MiniTiebout model."""
        communities = list(model.communities)
        platforms = list(model.platforms)

        community_governance = []
        for c in communities:
            if hasattr(c, "platform") and hasattr(c.platform, "institution"):
                community_governance.append(c.platform.institution)
            else:
                community_governance.append("")

        tracker_log = None
        if hasattr(model, "tracker") and model.tracker is not None:
            tracker_log = model.tracker.get_log()

        step_log = getattr(model, "step_log", None)
        step_series = getattr(model, "step_series", None)

        return IterationResult(
            n_comms=model.p.n_comms,
            n_plats=model.p.n_plats,
            p_space=model.p.p_space,
            institution=model.p.institution,
            alpha=getattr(model.p, "alpha", 1.0),
            steps=model.p.steps,
            has_extremists=model.p.extremists == "yes",
            community_utilities=[c.current_utility for c in communities],
            community_types=[c.type for c in communities],
            community_governance_types=community_governance,
            community_moves=[c.moves for c in communities],
            community_last_move_steps=[c.last_move_step for c in communities],
            platform_institutions=[p.institution for p in platforms],
            platform_community_counts=[len(p.communities) for p in platforms],
            tracker_log=tracker_log,
            step_log=step_log,
            step_series=step_series,
        )

    def add_iteration(self, result: IterationResult) -> None:
        """Add a completed iteration result."""
        self._iterations.append(result)
        self._summary = None  # invalidate cache

    def compute_summary(self) -> list[AggregatedMeasure]:
        """Compute all measures per iteration, then aggregate across iterations."""
        if not self._iterations:
            raise ValueError("No iterations to summarize")

        if self._summary is not None:
            return self._summary

        # Collect per-iteration measure values
        measure_values: dict[str, list[float]] = {}

        for it in self._iterations:
            measures = self._compute_iteration_measures(it)
            for name, value in measures.items():
                measure_values.setdefault(name, []).append(value)

        # Aggregate each measure
        self._summary = [
            self._aggregate(name, values)
            for name, values in measure_values.items()
        ]
        return self._summary

    def _compute_iteration_measures(self, it: IterationResult) -> dict[str, float]:
        """Compute the full set of measures for a single iteration."""
        measures: dict[str, float] = {}

        utilities = np.array(it.community_utilities)
        types = it.community_types
        gov_types = it.community_governance_types
        moves = it.community_moves
        last_steps = it.community_last_move_steps

        # 1. System-wide average utility
        measures["avg_utility"] = float(np.mean(utilities))

        # 2. By community type
        type_set = set(types)
        for ctype in sorted(type_set):
            mask = [t == ctype for t in types]
            if any(mask):
                measures[f"avg_utility_{ctype}"] = float(np.mean(utilities[mask]))

        # 3. By governance type
        gov_set = set(gov_types) - {""}
        for gtype in sorted(gov_set):
            mask = [g == gtype for g in gov_types]
            if any(mask):
                measures[f"avg_utility_gov_{gtype}"] = float(np.mean(utilities[mask]))

        # 4. Full cross: ctype x gtype
        for ctype in sorted(type_set):
            for gtype in sorted(gov_set):
                mask = [t == ctype and g == gtype for t, g in zip(types, gov_types)]
                if any(mask):
                    measures[f"avg_utility_{ctype}_{gtype}"] = float(
                        np.mean(utilities[mask])
                    )

        # 5. Normalized utility
        p_space = it.p_space
        alpha = it.alpha
        for ctype in sorted(type_set):
            mask = [t == ctype for t in types]
            if any(mask):
                mean_u = float(np.mean(utilities[mask]))
                if ctype == "extremist":
                    denom = p_space + alpha
                else:
                    denom = p_space
                measures[f"norm_utility_{ctype}"] = mean_u / denom if denom > 0 else 0.0

        # 6. Total relocations (moves - 1, clamped to 0)
        relocations = [max(0, m - 1) for m in moves]
        total_reloc = sum(relocations)
        measures["total_relocations"] = float(total_reloc)

        # 7. Average relocations per community
        measures["avg_relocations_per_community"] = total_reloc / it.n_comms

        # 8. Settling time 90th percentile
        sorted_steps = sorted(last_steps)
        n = len(sorted_steps)
        idx = math.ceil(0.9 * n) - 1
        idx = max(0, min(idx, n - 1))
        measures["settling_time_90pct"] = float(sorted_steps[idx])

        # 9 & 10. Final counts and proportions by governance type
        for gtype in sorted(gov_set):
            count = sum(1 for g in gov_types if g == gtype)
            measures[f"final_count_{gtype}"] = float(count)
            measures[f"final_proportion_{gtype}"] = count / it.n_comms

        # 11. Cross counts: ctype x gtype
        for ctype in sorted(type_set):
            for gtype in sorted(gov_set):
                count = sum(
                    1 for t, g in zip(types, gov_types) if t == ctype and g == gtype
                )
                measures[f"final_count_{ctype}_{gtype}"] = float(count)

        return measures

    @staticmethod
    def _aggregate(name: str, values: list[float]) -> AggregatedMeasure:
        """Aggregate a list of per-iteration values into summary statistics."""
        arr = np.array(values, dtype=np.float64)
        n = len(arr)
        mean = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        median = float(np.median(arr))
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        margin = 1.96 * (sd / math.sqrt(n)) if n > 0 else 0.0
        return AggregatedMeasure(
            name=name,
            mean=mean,
            sd=sd,
            ci_lower=mean - margin,
            ci_upper=mean + margin,
            median=median,
            min=vmin,
            max=vmax,
            n_iterations=n,
        )

    def to_csv(self, filepath: str) -> None:
        """Export summary to CSV file."""
        summary = self.compute_summary()
        fieldnames = [
            "Measure",
            "Mean",
            "SD",
            "CI_Lower",
            "CI_Upper",
            "Median",
            "Min",
            "Max",
            "N",
        ]
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in summary:
                writer.writerow(
                    {
                        "Measure": m.name,
                        "Mean": f"{m.mean:.6f}",
                        "SD": f"{m.sd:.6f}",
                        "CI_Lower": f"{m.ci_lower:.6f}",
                        "CI_Upper": f"{m.ci_upper:.6f}",
                        "Median": f"{m.median:.6f}",
                        "Min": f"{m.min:.6f}",
                        "Max": f"{m.max:.6f}",
                        "N": m.n_iterations,
                    }
                )

    def to_latex(self, filepath: str) -> None:
        """Export summary to LaTeX booktabs table."""
        summary = self.compute_summary()
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\begin{tabular}{lrrrrrrr}",
            r"\toprule",
            r"Measure & Mean & SD & 95\% CI & Median & Min & Max & N \\",
            r"\midrule",
        ]
        for m in summary:
            ci = f"[{m.ci_lower:.3f}, {m.ci_upper:.3f}]"
            lines.append(
                f"{m.name} & {m.mean:.3f} & {m.sd:.3f} & {ci} "
                f"& {m.median:.3f} & {m.min:.3f} & {m.max:.3f} & {m.n_iterations} \\\\"
            )
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{Simulation Summary Statistics}",
                r"\end{table}",
            ]
        )
        with open(filepath, "w") as f:
            f.write("\n".join(lines) + "\n")

    def compute_step_summary(self) -> dict[str, dict[str, np.ndarray]]:
        """Aggregate step_series across iterations into mean and 95% CI arrays.

        Returns a dict mapping metric names to {'mean': array, 'ci': array}.
        Nested dicts (per_governance_utility, etc.) produce flattened keys
        like 'utility_direct', 'community_count_algorithmic', etc.
        """
        series_list = [
            it.step_series for it in self._iterations if it.step_series is not None
        ]
        if not series_list:
            raise ValueError("No step_series data available")

        n_iters = len(series_list)
        n_steps = len(series_list[0]['steps'])
        result: dict[str, dict[str, np.ndarray]] = {}

        # Simple (flat) metrics
        for key in ('avg_utility', 'n_relocations'):
            matrix = np.zeros((n_iters, n_steps))
            for i, ss in enumerate(series_list):
                vals = ss.get(key, [])
                length = min(len(vals), n_steps)
                matrix[i, :length] = vals[:length]
            mean = np.mean(matrix, axis=0)
            se = np.std(matrix, axis=0, ddof=1) / math.sqrt(n_iters) if n_iters > 1 else np.zeros(n_steps)
            result[key] = {'mean': mean, 'ci': 1.96 * se}

        # Nested dict metrics
        nested_keys = [
            ('per_governance_utility', 'utility'),
            ('per_governance_community_count', 'community_count'),
        ]
        # Conditionally add per-type metrics
        if 'per_type_utility' in series_list[0]:
            nested_keys.append(('per_type_utility', 'type_utility'))
        if 'per_type_relocations' in series_list[0]:
            nested_keys.append(('per_type_relocations', 'type_relocations'))

        for series_key, prefix in nested_keys:
            sub_keys = sorted(series_list[0].get(series_key, {}).keys())
            for sk in sub_keys:
                matrix = np.zeros((n_iters, n_steps))
                for i, ss in enumerate(series_list):
                    vals = ss.get(series_key, {}).get(sk, [])
                    length = min(len(vals), n_steps)
                    matrix[i, :length] = vals[:length]
                mean = np.mean(matrix, axis=0)
                se = np.std(matrix, axis=0, ddof=1) / math.sqrt(n_iters) if n_iters > 1 else np.zeros(n_steps)
                result[f'{prefix}_{sk}'] = {'mean': mean, 'ci': 1.96 * se}

        # Include step numbers
        result['steps'] = {'mean': np.array(series_list[0]['steps'], dtype=float), 'ci': np.zeros(n_steps)}

        return result

    def step_summary_to_csv(self, filepath: str) -> None:
        """Write per-step aggregated time series to CSV (one row per step)."""
        summary = self.compute_step_summary()
        steps = summary.pop('steps')['mean']
        n_steps = len(steps)

        # Build columns: step, then mean/ci for each metric
        fieldnames = ['step']
        metric_names = sorted(summary.keys())
        for name in metric_names:
            fieldnames.append(f'{name}_mean')
            fieldnames.append(f'{name}_ci')

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for j in range(n_steps):
                row: dict[str, str] = {'step': str(int(steps[j]))}
                for name in metric_names:
                    row[f'{name}_mean'] = f"{summary[name]['mean'][j]:.6f}"
                    row[f'{name}_ci'] = f"{summary[name]['ci'][j]:.6f}"
                writer.writerow(row)

    def compute_convergence_diagnostics(self) -> dict:
        """Compute convergence diagnostics from aggregated step trajectories.

        Mirrors results/analyze_stepwise.py:convergence_diagnostics().
        """
        summary = self.compute_step_summary()
        util = summary['avg_utility']['mean']
        reloc = summary['n_relocations']['mean']
        n = len(util)

        # Slope of utility over last 20% of steps
        tail_start = int(n * 0.8)
        tail_util = util[tail_start:]
        tail_steps = np.arange(len(tail_util))

        if len(tail_util) > 1:
            slope = float(np.polyfit(tail_steps, tail_util, 1)[0])
        else:
            slope = 0.0

        util_start = float(util[0])
        util_mid = float(util[n // 2])
        util_end = float(util[-1])

        reloc_end = float(reloc[-1])
        reloc_start = float(reloc[0])

        # Coefficient of variation in tail
        tail_cv = float(np.std(tail_util) / np.mean(tail_util)) if np.mean(tail_util) > 0 else 0.0

        # Autocorrelation of utility differences (lag-1) in tail
        if len(tail_util) > 2:
            diffs = np.diff(tail_util)
            if len(diffs) > 1 and np.std(diffs[:-1]) > 0 and np.std(diffs[1:]) > 0:
                autocorr = float(np.corrcoef(diffs[:-1], diffs[1:])[0, 1])
            else:
                autocorr = 0.0
        else:
            autocorr = 0.0

        # Classification (same thresholds as analyze_stepwise.py)
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
            'reloc_reduction_pct': (1 - reloc_end / reloc_start) * 100 if reloc_start > 0 else 0.0,
        }
