"""Movement analysis: flow matrices, residence times, raiding cycles, enclaves."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from platform_abm.tracker import RelocationTracker

if TYPE_CHECKING:
    pass


def _acf_numpy(series: NDArray[np.float64], nlags: int) -> NDArray[np.float64]:
    """Compute autocorrelation via numpy (fallback when statsmodels unavailable)."""
    n = len(series)
    mean = np.mean(series)
    var = np.sum((series - mean) ** 2)
    if var == 0:
        result = np.zeros(nlags + 1)
        result[0] = 1.0
        return result

    result = np.empty(nlags + 1)
    for k in range(nlags + 1):
        cov = np.sum((series[: n - k] - mean) * (series[k:] - mean))
        result[k] = cov / var
    return result


def _acf(series: NDArray[np.float64], nlags: int) -> NDArray[np.float64]:
    """Compute ACF using statsmodels if available, else numpy fallback."""
    try:
        from statsmodels.tsa.stattools import acf

        return acf(series, nlags=nlags, fft=True)  # type: ignore[return-value]
    except ImportError:
        return _acf_numpy(series, nlags)


class MovementAnalyzer:
    """Analyzes movement patterns from a RelocationTracker log."""

    def __init__(self, tracker: RelocationTracker, platform_ids: list[int]) -> None:
        self.tracker = tracker
        self.platform_ids = sorted(platform_ids)
        self._id_to_idx = {pid: i for i, pid in enumerate(self.platform_ids)}
        self.n_platforms = len(platform_ids)

    def compute_flow_matrices(self) -> dict[int, NDArray[np.int_]]:
        """Per-step (n_platforms x n_platforms) transition count matrices.

        Returns empty dict when no relocations occurred.
        """
        log = self.tracker.get_log()
        matrices: dict[int, NDArray[np.int_]] = {}

        for step, record in sorted(log.items()):
            if not record.relocations:
                continue
            mat = np.zeros((self.n_platforms, self.n_platforms), dtype=int)
            for event in record.relocations:
                from_idx = self._id_to_idx.get(event.from_platform_id)
                to_idx = self._id_to_idx.get(event.to_platform_id)
                if from_idx is not None and to_idx is not None:
                    mat[from_idx, to_idx] += 1
            matrices[step] = mat

        return matrices

    def compute_residence_times(
        self,
        community_ids: list[int],
        initial_assignments: dict[int, int],
        total_steps: int,
    ) -> dict[int, dict[int, float]]:
        """Compute fraction of time each community spent on each platform.

        Args:
            community_ids: All community IDs.
            initial_assignments: community_id → initial platform_id.
            total_steps: Total simulation steps (time slices).

        Returns:
            community_id → {platform_id: proportion} where proportions sum to 1.0.
        """
        if total_steps <= 0:
            return {cid: {initial_assignments.get(cid, 0): 1.0} for cid in community_ids}

        # Track current platform per community
        current: dict[int, int] = dict(initial_assignments)
        # Count time slices per platform per community
        counts: dict[int, dict[int, int]] = {
            cid: {} for cid in community_ids
        }

        log = self.tracker.get_log()

        for step in range(1, total_steps + 1):
            # Record current platform for this time slice
            for cid in community_ids:
                pid = current[cid]
                counts[cid][pid] = counts[cid].get(pid, 0) + 1

            # Apply relocations at this step
            if step in log:
                for event in log[step].relocations:
                    current[event.community_id] = event.to_platform_id

        # Normalize to proportions
        result: dict[int, dict[int, float]] = {}
        for cid in community_ids:
            total_count = sum(counts[cid].values())
            if total_count == 0:
                result[cid] = {initial_assignments.get(cid, 0): 1.0}
            else:
                result[cid] = {
                    pid: count / total_count
                    for pid, count in counts[cid].items()
                }
        return result

    def detect_raiding_cycles(
        self,
        total_steps: int,
        nlags: int = 10,
        significance_threshold: float = 1.96,
    ) -> dict[int, dict[str, Any]]:
        """Detect periodic raiding by extremists leaving platforms.

        Builds per-platform extremist outflow time series, computes ACF,
        and identifies significant periodic lags.

        Returns:
            platform_id → {outflow_series, acf, significant_lags, has_cycle}
        """
        log = self.tracker.get_log()

        # Build outflow time series per platform
        outflow: dict[int, NDArray[np.float64]] = {
            pid: np.zeros(total_steps, dtype=np.float64) for pid in self.platform_ids
        }

        for step, record in log.items():
            step_idx = step - 1  # Convert 1-based step to 0-based index
            if step_idx < 0 or step_idx >= total_steps:
                continue
            for event in record.relocations:
                if (
                    event.community_type == "extremist"
                    and event.from_platform_id in outflow
                ):
                    outflow[event.from_platform_id][step_idx] += 1

        results: dict[int, dict[str, Any]] = {}
        for pid in self.platform_ids:
            series = outflow[pid]
            actual_nlags = min(nlags, len(series) - 1)
            if actual_nlags < 1:
                results[pid] = {
                    "outflow_series": series,
                    "acf": np.array([1.0]),
                    "significant_lags": [],
                    "has_cycle": False,
                }
                continue

            acf_values = _acf(series, actual_nlags)
            n = len(series)
            threshold = significance_threshold / math.sqrt(n) if n > 0 else float("inf")

            significant_lags = [
                lag
                for lag in range(1, len(acf_values))
                if abs(acf_values[lag]) > threshold
            ]

            results[pid] = {
                "outflow_series": series,
                "acf": acf_values,
                "significant_lags": significant_lags,
                "has_cycle": len(significant_lags) > 0,
            }

        return results

    def detect_enclaves(
        self,
        community_types: dict[int, str],
        enclave_threshold: float = 0.9,
    ) -> dict[int, dict[str, Any]]:
        """Detect extremist enclaves in coalition platforms.

        For coalition platforms, at each step compute homogeneity among
        winning-coalition voters using governance snapshots.

        Returns:
            platform_id → {homogeneity_series, mean_homogeneity, fraction_enclaved}
        """
        log = self.tracker.get_log()
        results: dict[int, dict[str, Any]] = {}

        # Collect all coalition platform IDs from governance snapshots
        coalition_plats: set[int] = set()
        for record in log.values():
            for snap in record.governance:
                if snap.institution == "coalition":
                    coalition_plats.add(snap.platform_id)

        for pid in coalition_plats:
            homogeneity_series: list[float] = []

            for step in sorted(log):
                record = log[step]
                for snap in record.governance:
                    if snap.platform_id != pid or snap.institution != "coalition":
                        continue
                    if not snap.community_order or snap.winning_coalition_index is None:
                        continue

                    # Find winning coalition voters
                    winning_idx = snap.winning_coalition_index
                    winning_voters = [
                        cid
                        for cid, vote in zip(snap.community_order, snap.coalition_votes)
                        if vote == winning_idx
                    ]

                    if not winning_voters:
                        continue

                    n_main = sum(
                        1
                        for cid in winning_voters
                        if community_types.get(cid) == "mainstream"
                    )
                    n_ext = len(winning_voters) - n_main
                    total = len(winning_voters)
                    homogeneity = max(n_main, n_ext) / total
                    homogeneity_series.append(homogeneity)

            if homogeneity_series:
                arr = np.array(homogeneity_series)
                results[pid] = {
                    "homogeneity_series": arr,
                    "mean_homogeneity": float(np.mean(arr)),
                    "fraction_enclaved": float(np.mean(arr > enclave_threshold)),
                }
            else:
                results[pid] = {
                    "homogeneity_series": np.array([]),
                    "mean_homogeneity": 0.0,
                    "fraction_enclaved": 0.0,
                }

        return results
