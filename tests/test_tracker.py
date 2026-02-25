"""Tests for RelocationTracker."""

from platform_abm.tracker import (
    GovernanceSnapshot,
    RelocationEvent,
    RelocationTracker,
    StepRecord,
)
from tests.conftest import make_model


class TestTrackerDisabled:
    def test_disabled_records_nothing(self):
        tracker = RelocationTracker(enabled=False)
        tracker.record_step(1, [])
        tracker.record_governance_state(1, [])
        assert tracker.get_log() == {}
        assert tracker.get_all_relocations() == []


class TestTrackerDataclasses:
    def test_relocation_event_fields(self):
        event = RelocationEvent(
            community_id=1,
            community_type="mainstream",
            from_platform_id=10,
            to_platform_id=20,
            from_institution="direct",
            to_institution="coalition",
        )
        assert event.community_id == 1
        assert event.from_institution == "direct"

    def test_governance_snapshot_defaults(self):
        snap = GovernanceSnapshot(platform_id=1, institution="direct")
        assert snap.coalition_votes == []
        assert snap.winning_coalition_index is None
        assert snap.community_order == []
        assert snap.group_membership == {}

    def test_step_record_defaults(self):
        rec = StepRecord(step=1)
        assert rec.relocations == []
        assert rec.governance == []


class TestTrackerWithModel:
    def test_tracker_none_doesnt_crash(self):
        """Model runs fine without a tracker attached."""
        model = make_model({"steps": 3})
        assert model.tracker is None
        model.run()
        assert "average_moves" in model.reporters

    def test_tracker_attached_records_events(self):
        model = make_model({"steps": 5})
        tracker = RelocationTracker(enabled=True)
        model.tracker = tracker
        model.run()

        log = tracker.get_log()
        # Should have entries for steps where relocations or governance happened
        assert isinstance(log, dict)
        all_events = tracker.get_all_relocations()
        assert isinstance(all_events, list)
        for event in all_events:
            assert isinstance(event, RelocationEvent)

    def test_coalition_governance_captured(self):
        model = make_model({"institution": "coalition", "steps": 3})
        tracker = RelocationTracker(enabled=True)
        model.tracker = tracker
        model.run()

        log = tracker.get_log()
        found_coalition = False
        for record in log.values():
            for snap in record.governance:
                if snap.institution == "coalition":
                    found_coalition = True
                    assert isinstance(snap.coalition_votes, list)
                    assert isinstance(snap.community_order, list)
        assert found_coalition

    def test_algorithmic_governance_captured(self):
        model = make_model({"institution": "algorithmic", "steps": 3})
        tracker = RelocationTracker(enabled=True)
        model.tracker = tracker
        model.run()

        log = tracker.get_log()
        found_algo = False
        for record in log.values():
            for snap in record.governance:
                if snap.institution == "algorithmic":
                    found_algo = True
                    assert isinstance(snap.group_membership, dict)
        assert found_algo

    def test_model_results_unchanged_with_tracker(self):
        """Tracker should not alter simulation results."""
        model_no_tracker = make_model({"steps": 3, "seed": 99})
        model_no_tracker.run()

        model_with_tracker = make_model({"steps": 3, "seed": 99})
        model_with_tracker.tracker = RelocationTracker(enabled=True)
        model_with_tracker.run()

        assert (
            model_no_tracker.reporters["average_moves"]
            == model_with_tracker.reporters["average_moves"]
        )
        assert (
            model_no_tracker.reporters["average_utility"]
            == model_with_tracker.reporters["average_utility"]
        )


class TestStepMetrics:
    def test_step_metrics_recorded(self):
        """Every step should have avg_utility and n_relocations populated."""
        model = make_model({"steps": 5})
        tracker = RelocationTracker(enabled=True)
        model.tracker = tracker
        model.run()

        log = tracker.get_log()
        for record in log.values():
            assert record.avg_utility is not None
            assert record.avg_utility >= 0
            assert record.n_relocations is not None
            assert record.n_relocations >= 0

    def test_step_metrics_accessor(self):
        """get_step_metrics() returns correct count, sorted order, and expected keys."""
        steps = 4
        model = make_model({"steps": steps})
        tracker = RelocationTracker(enabled=True)
        model.tracker = tracker
        model.run()

        metrics = tracker.get_step_metrics()
        assert len(metrics) == steps
        # Sorted by step
        step_nums = [m["step"] for m in metrics]
        assert step_nums == sorted(step_nums)
        # Expected keys
        for m in metrics:
            assert "step" in m
            assert "avg_utility" in m
            assert "n_relocations" in m
            assert "per_governance_utilities" in m

    def test_disabled_records_no_metrics(self):
        """Disabled tracker produces empty metrics list."""
        model = make_model({"steps": 3})
        tracker = RelocationTracker(enabled=False)
        model.tracker = tracker
        model.run()

        assert tracker.get_step_metrics() == []

    def test_mixed_governance_utilities(self):
        """Mixed institution model records per-type utility breakdown."""
        model = make_model({"institution": "mixed", "n_plats": 3, "steps": 3})
        tracker = RelocationTracker(enabled=True)
        model.tracker = tracker
        model.run()

        metrics = tracker.get_step_metrics()
        assert len(metrics) > 0
        valid_institutions = {"direct", "coalition", "algorithmic"}
        for m in metrics:
            per_gov = m["per_governance_utilities"]
            assert len(per_gov) > 0
            for inst, util in per_gov.items():
                assert inst in valid_institutions
                assert isinstance(util, float)

    def test_relocation_count_matches_events(self):
        """n_relocations should equal len(relocations) for every step."""
        model = make_model({"steps": 5})
        tracker = RelocationTracker(enabled=True)
        model.tracker = tracker
        model.run()

        log = tracker.get_log()
        for record in log.values():
            assert record.n_relocations == len(record.relocations)
