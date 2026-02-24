"""Tests for MovementAnalyzer."""

import numpy as np
import pytest

from platform_abm.analyzer import MovementAnalyzer, _acf_numpy
from platform_abm.tracker import RelocationEvent, RelocationTracker, StepRecord
from tests.conftest import make_model


def _make_tracker_with_events(
    events_by_step: dict[int, list[tuple[int, str, int, int, str, str]]],
) -> RelocationTracker:
    """Build a tracker from synthetic event tuples.

    Each tuple: (community_id, community_type, from_plat, to_plat, from_inst, to_inst)
    """
    tracker = RelocationTracker(enabled=True)
    for step, events in events_by_step.items():
        tuples = []
        for cid, ctype, fp, tp, fi, ti in events:

            class _FakeComm:
                pass

            class _FakePlat:
                pass

            comm = _FakeComm()
            comm.id = cid
            comm.type = ctype
            old_plat = _FakePlat()
            old_plat.id = fp
            old_plat.institution = fi
            new_plat = _FakePlat()
            new_plat.id = tp
            new_plat.institution = ti
            tuples.append((comm, old_plat, new_plat))
        tracker.record_step(step, tuples)
    return tracker


class TestFlowMatrices:
    def test_correct_shape(self):
        tracker = _make_tracker_with_events(
            {1: [(1, "mainstream", 10, 20, "direct", "direct")]}
        )
        analyzer = MovementAnalyzer(tracker, [10, 20])
        matrices = analyzer.compute_flow_matrices()
        assert 1 in matrices
        assert matrices[1].shape == (2, 2)

    def test_zero_diagonal_single_move(self):
        tracker = _make_tracker_with_events(
            {1: [(1, "mainstream", 10, 20, "direct", "direct")]}
        )
        analyzer = MovementAnalyzer(tracker, [10, 20])
        matrices = analyzer.compute_flow_matrices()
        assert matrices[1][0, 0] == 0  # no self-move
        assert matrices[1][0, 1] == 1  # 10→20

    def test_empty_when_no_moves(self):
        tracker = RelocationTracker(enabled=True)
        analyzer = MovementAnalyzer(tracker, [10, 20])
        matrices = analyzer.compute_flow_matrices()
        assert matrices == {}

    def test_sum_matches_relocation_count(self):
        tracker = _make_tracker_with_events(
            {
                1: [
                    (1, "mainstream", 10, 20, "direct", "direct"),
                    (2, "extremist", 20, 10, "direct", "direct"),
                ],
                2: [(3, "mainstream", 10, 20, "direct", "direct")],
            }
        )
        analyzer = MovementAnalyzer(tracker, [10, 20])
        matrices = analyzer.compute_flow_matrices()
        total = sum(m.sum() for m in matrices.values())
        assert total == 3


class TestResidenceTimes:
    def test_proportions_sum_to_one(self):
        tracker = _make_tracker_with_events(
            {2: [(1, "mainstream", 10, 20, "direct", "direct")]}
        )
        analyzer = MovementAnalyzer(tracker, [10, 20])
        result = analyzer.compute_residence_times([1], {1: 10}, total_steps=5)
        total = sum(result[1].values())
        assert abs(total - 1.0) < 1e-9

    def test_no_move_stays_on_initial(self):
        tracker = RelocationTracker(enabled=True)
        analyzer = MovementAnalyzer(tracker, [10, 20])
        result = analyzer.compute_residence_times([1], {1: 10}, total_steps=5)
        assert result[1] == {10: 1.0}

    def test_split_residence(self):
        """Community moves at step 3 of 4 total.

        Record-then-move: steps 1,2,3 on plat 10, step 4 on plat 20 → 75%/25%.
        """
        tracker = _make_tracker_with_events(
            {3: [(1, "mainstream", 10, 20, "direct", "direct")]}
        )
        analyzer = MovementAnalyzer(tracker, [10, 20])
        result = analyzer.compute_residence_times([1], {1: 10}, total_steps=4)
        assert result[1][10] == pytest.approx(0.75)
        assert result[1][20] == pytest.approx(0.25)


class TestRaidingCycles:
    def test_constant_series_no_cycle(self):
        """No extremist moves → no cycles detected."""
        tracker = RelocationTracker(enabled=True)
        # Record empty steps
        for s in range(1, 21):
            tracker.record_step(s, [])
        analyzer = MovementAnalyzer(tracker, [10])
        result = analyzer.detect_raiding_cycles(total_steps=20, nlags=5)
        assert not result[10]["has_cycle"]

    def test_acf_lag_zero_is_one(self):
        series = np.random.default_rng(42).random(50)
        acf_vals = _acf_numpy(series, 10)
        assert acf_vals[0] == pytest.approx(1.0)

    def test_synthetic_period_4(self):
        """Period-4 signal: extremist leaves plat 10 every 4 steps."""
        events: dict[int, list[tuple[int, str, int, int, str, str]]] = {}
        for s in range(1, 41):
            if s % 4 == 0:
                events[s] = [(1, "extremist", 10, 20, "direct", "direct")]
            else:
                events[s] = []

        tracker = _make_tracker_with_events(events)
        # Fill governance for steps without events
        for s in range(1, 41):
            if s not in tracker._log:
                tracker._log[s] = StepRecord(step=s)

        analyzer = MovementAnalyzer(tracker, [10, 20])
        result = analyzer.detect_raiding_cycles(total_steps=40, nlags=10)
        # Lag 4 should be significant for platform 10
        plat_result = result[10]
        assert 4 in plat_result["significant_lags"]
        assert plat_result["has_cycle"]


class TestEnclaves:
    def test_homogeneity_in_range(self):
        model = make_model(
            {"institution": "coalition", "steps": 3, "extremists": "yes", "percent_extremists": 30}
        )
        tracker = RelocationTracker(enabled=True)
        model.tracker = tracker
        model.run()

        community_types = {c.id: c.type for c in model.communities}
        platform_ids = [p.id for p in model.platforms]
        analyzer = MovementAnalyzer(tracker, platform_ids)
        result = analyzer.detect_enclaves(community_types)

        for pid, data in result.items():
            if len(data["homogeneity_series"]) > 0:
                assert all(0.0 <= h <= 1.0 for h in data["homogeneity_series"])
                assert 0.0 <= data["mean_homogeneity"] <= 1.0
                assert 0.0 <= data["fraction_enclaved"] <= 1.0


class TestACFFallback:
    def test_zero_variance_returns_ones_at_lag_zero(self):
        series = np.ones(20)
        result = _acf_numpy(series, 5)
        assert result[0] == 1.0
        # All other lags should be 0
        assert all(result[k] == 0.0 for k in range(1, 6))
