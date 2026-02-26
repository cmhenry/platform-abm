"""Tests for SimulationReporter."""

import csv
import os
import tempfile

import numpy as np
import pytest

from platform_abm.reporter import AggregatedMeasure, IterationResult, SimulationReporter
from tests.conftest import make_model


def _make_iteration(
    n_comms: int = 20,
    p_space: int = 5,
    alpha: float = 1.0,
    utilities: list[float] | None = None,
    types: list[str] | None = None,
    gov_types: list[str] | None = None,
    moves: list[int] | None = None,
    last_steps: list[int] | None = None,
    institution: str = "direct",
    has_extremists: bool = False,
) -> IterationResult:
    """Factory for synthetic IterationResults."""
    if utilities is None:
        utilities = [3.0] * n_comms
    if types is None:
        types = ["mainstream"] * n_comms
    if gov_types is None:
        gov_types = [institution] * n_comms
    if moves is None:
        moves = [1] * n_comms  # initial join only
    if last_steps is None:
        last_steps = [0] * n_comms

    return IterationResult(
        n_comms=n_comms,
        n_plats=2,
        p_space=p_space,
        institution=institution,
        alpha=alpha,
        steps=10,
        has_extremists=has_extremists,
        community_utilities=utilities,
        community_types=types,
        community_governance_types=gov_types,
        community_moves=moves,
        community_last_move_steps=last_steps,
        platform_institutions=["direct", "direct"],
        platform_community_counts=[n_comms // 2, n_comms - n_comms // 2],
    )


class TestSingleIteration:
    def test_avg_utility(self):
        reporter = SimulationReporter()
        reporter.add_iteration(_make_iteration(utilities=[2.0, 4.0, 6.0], n_comms=3))
        summary = reporter.compute_summary()
        measures = {m.name: m for m in summary}
        assert measures["avg_utility"].mean == pytest.approx(4.0)

    def test_single_iteration_sd_is_zero(self):
        reporter = SimulationReporter()
        reporter.add_iteration(_make_iteration())
        summary = reporter.compute_summary()
        for m in summary:
            assert m.sd == 0.0


class TestMultipleIterations:
    def test_sd_with_ddof_1(self):
        reporter = SimulationReporter()
        reporter.add_iteration(_make_iteration(utilities=[2.0] * 10, n_comms=10))
        reporter.add_iteration(_make_iteration(utilities=[4.0] * 10, n_comms=10))
        summary = reporter.compute_summary()
        measures = {m.name: m for m in summary}
        # mean of [2.0, 4.0] = 3.0, std(ddof=1) = sqrt(2)
        assert measures["avg_utility"].mean == pytest.approx(3.0)
        assert measures["avg_utility"].sd == pytest.approx(np.std([2.0, 4.0], ddof=1))

    def test_ci_formula(self):
        reporter = SimulationReporter()
        reporter.add_iteration(_make_iteration(utilities=[2.0] * 10, n_comms=10))
        reporter.add_iteration(_make_iteration(utilities=[4.0] * 10, n_comms=10))
        summary = reporter.compute_summary()
        measures = {m.name: m for m in summary}
        m = measures["avg_utility"]
        expected_margin = 1.96 * (m.sd / np.sqrt(2))
        assert m.ci_lower == pytest.approx(m.mean - expected_margin)
        assert m.ci_upper == pytest.approx(m.mean + expected_margin)


class TestNoIterationsRaises:
    def test_empty_raises(self):
        reporter = SimulationReporter()
        with pytest.raises(ValueError, match="No iterations"):
            reporter.compute_summary()


class TestNormalizedUtility:
    def test_mainstream_normalized_by_p_space(self):
        reporter = SimulationReporter()
        reporter.add_iteration(
            _make_iteration(
                utilities=[4.0] * 5,
                types=["mainstream"] * 5,
                n_comms=5,
                p_space=5,
            )
        )
        summary = reporter.compute_summary()
        measures = {m.name: m for m in summary}
        assert measures["norm_utility_mainstream"].mean == pytest.approx(4.0 / 5.0)

    def test_extremist_normalized_by_p_space_plus_alpha(self):
        reporter = SimulationReporter()
        reporter.add_iteration(
            _make_iteration(
                utilities=[5.0] * 5,
                types=["extremist"] * 5,
                n_comms=5,
                p_space=5,
                alpha=2.0,
                has_extremists=True,
            )
        )
        summary = reporter.compute_summary()
        measures = {m.name: m for m in summary}
        # 5.0 / (5 + 2.0) = 5/7
        assert measures["norm_utility_extremist"].mean == pytest.approx(5.0 / 7.0)


class TestRelocations:
    def test_total_relocations(self):
        reporter = SimulationReporter()
        reporter.add_iteration(
            _make_iteration(
                moves=[1, 3, 2, 1, 5],  # relocations: 0, 2, 1, 0, 4 = 7
                n_comms=5,
            )
        )
        summary = reporter.compute_summary()
        measures = {m.name: m for m in summary}
        assert measures["total_relocations"].mean == 7.0
        assert measures["avg_relocations_per_community"].mean == pytest.approx(7.0 / 5)

    def test_zero_moves_zero_relocations(self):
        reporter = SimulationReporter()
        reporter.add_iteration(_make_iteration(moves=[1, 1, 1], n_comms=3))
        summary = reporter.compute_summary()
        measures = {m.name: m for m in summary}
        assert measures["total_relocations"].mean == 0.0


class TestSettlingTime:
    def test_90th_percentile(self):
        reporter = SimulationReporter()
        # 10 communities: last_move_steps = [0,1,2,3,4,5,6,7,8,9]
        # 90th percentile: ceil(0.9*10)-1 = 8 → value 8
        reporter.add_iteration(
            _make_iteration(
                last_steps=list(range(10)),
                n_comms=10,
            )
        )
        summary = reporter.compute_summary()
        measures = {m.name: m for m in summary}
        assert measures["settling_time_90pct"].mean == 8.0


class TestGovernanceCounts:
    def test_final_counts_and_proportions(self):
        reporter = SimulationReporter()
        reporter.add_iteration(
            _make_iteration(
                gov_types=["direct"] * 12 + ["coalition"] * 8,
                n_comms=20,
            )
        )
        summary = reporter.compute_summary()
        measures = {m.name: m for m in summary}
        assert measures["final_count_direct"].mean == 12.0
        assert measures["final_count_coalition"].mean == 8.0
        assert measures["final_proportion_direct"].mean == pytest.approx(0.6)
        assert measures["final_proportion_coalition"].mean == pytest.approx(0.4)

    def test_cross_counts(self):
        reporter = SimulationReporter()
        reporter.add_iteration(
            _make_iteration(
                types=["mainstream"] * 10 + ["extremist"] * 10,
                gov_types=["direct"] * 10 + ["direct"] * 10,
                n_comms=20,
                has_extremists=True,
            )
        )
        summary = reporter.compute_summary()
        measures = {m.name: m for m in summary}
        assert measures["final_count_mainstream_direct"].mean == 10.0
        assert measures["final_count_extremist_direct"].mean == 10.0


class TestNoExtremists:
    def test_no_extremist_measures_when_all_mainstream(self):
        reporter = SimulationReporter()
        reporter.add_iteration(_make_iteration())
        summary = reporter.compute_summary()
        names = {m.name for m in summary}
        assert "norm_utility_extremist" not in names
        assert "avg_utility_extremist" not in names
        assert "norm_utility_mainstream" in names


class TestCSVExport:
    def test_csv_structure(self):
        reporter = SimulationReporter()
        reporter.add_iteration(_make_iteration())
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            filepath = f.name
        try:
            reporter.to_csv(filepath)
            with open(filepath) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) > 0
            assert "Measure" in rows[0]
            assert "Mean" in rows[0]
            assert "SD" in rows[0]
        finally:
            os.unlink(filepath)


class TestLatexExport:
    def test_latex_structure(self):
        reporter = SimulationReporter()
        reporter.add_iteration(_make_iteration())
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            filepath = f.name
        try:
            reporter.to_latex(filepath)
            with open(filepath) as f:
                content = f.read()
            assert r"\begin{table}" in content
            assert r"\toprule" in content
            assert r"\bottomrule" in content
            assert r"\end{table}" in content
            assert "avg_utility" in content
        finally:
            os.unlink(filepath)


class TestFromModel:
    def test_extracts_correct_data(self):
        model = make_model({"steps": 3, "seed": 42})
        model.run()
        result = SimulationReporter.from_model(model)
        assert result.n_comms == model.p.n_comms
        assert result.p_space == model.p.p_space
        assert len(result.community_utilities) == model.p.n_comms
        assert len(result.community_types) == model.p.n_comms
        assert len(result.community_moves) == model.p.n_comms
        assert len(result.community_last_move_steps) == model.p.n_comms

    def test_from_model_with_tracker(self):
        from platform_abm.tracker import RelocationTracker

        model = make_model({"steps": 3, "seed": 42})
        model.tracker = RelocationTracker(enabled=True)
        model.run()
        result = SimulationReporter.from_model(model)
        assert result.tracker_log is not None

    def test_from_model_feeds_reporter(self):
        model = make_model({"steps": 3, "seed": 42})
        model.run()
        result = SimulationReporter.from_model(model)
        reporter = SimulationReporter()
        reporter.add_iteration(result)
        summary = reporter.compute_summary()
        assert len(summary) > 0
        measures = {m.name: m for m in summary}
        assert "avg_utility" in measures

    def test_step_log_extracted(self):
        model = make_model({"steps": 3, "seed": 42})
        model.run()
        result = SimulationReporter.from_model(model)
        assert result.step_log is not None
        assert len(result.step_log) == 3
        for entry in result.step_log:
            assert "step" in entry
            assert "avg_utility" in entry
            assert "n_relocations" in entry
            assert "per_governance_utilities" in entry

    def test_extremist_model_extraction(self):
        model = make_model(
            {"steps": 3, "seed": 42, "extremists": "yes", "percent_extremists": 20}
        )
        model.run()
        result = SimulationReporter.from_model(model)
        assert result.has_extremists
        assert "extremist" in result.community_types
