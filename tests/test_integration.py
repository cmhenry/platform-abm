"""Integration tests: full simulation runs."""

import pytest

from platform_abm.model import MiniTiebout


def _run_model(institution: str, extremists: str = "no", **extra) -> MiniTiebout:
    """Helper to run a model to completion."""
    params = {
        "n_comms": 30,
        "n_plats": 3,
        "p_space": 5,
        "p_type": "binary",
        "steps": 5,
        "institution": institution,
        "extremists": extremists,
        "percent_extremists": 10,
        "coalitions": 3,
        "mutations": 2,
        "search_steps": 5,
        "svd_groups": 2,
        "stop_condition": "steps",
        "seed": 42,
    }
    params.update(extra)
    model = MiniTiebout(params)
    model.run()
    return model


class TestFullSimulationRuns:
    def test_direct_completes(self):
        model = _run_model("direct")
        assert "average_moves" in model.reporters
        assert "average_utility" in model.reporters

    def test_coalition_completes(self):
        model = _run_model("coalition")
        assert "average_moves" in model.reporters

    def test_algorithmic_completes(self):
        model = _run_model("algorithmic")
        assert "average_moves" in model.reporters

    def test_mixed_completes(self):
        model = _run_model("mixed")
        assert "n_direct_comms" in model.reporters
        assert "n_coalition_comms" in model.reporters
        assert "n_algo_comms" in model.reporters

    def test_extremists_completes(self):
        model = _run_model("direct", extremists="yes")
        assert "average_extremist_utility" in model.reporters
        assert "average_mainstream_utility" in model.reporters

    def test_mixed_extremists_completes(self):
        model = _run_model("mixed", extremists="yes")
        assert "average_extremist_utility" in model.reporters
        assert "n_direct_comms" in model.reporters


class TestCommunityConservation:
    """Total communities should be conserved across platforms."""

    @pytest.mark.parametrize("institution", ["direct", "coalition", "algorithmic", "mixed"])
    def test_community_count_conserved(self, institution):
        model = _run_model(institution)
        total_on_platforms = sum(len(p.communities) for p in model.platforms)
        assert total_on_platforms == model.p.n_comms


class TestReporterKeys:
    def test_direct_reporters(self):
        model = _run_model("direct")
        assert set(model.reporters.keys()) >= {"seed", "average_moves", "average_utility"}

    def test_mixed_reporters(self):
        model = _run_model("mixed")
        expected = {
            "seed",
            "average_moves",
            "average_utility",
            "n_direct_comms",
            "n_coalition_comms",
            "n_algo_comms",
            "ratio_direct",
            "ratio_coalition",
            "ratio_algo",
            "util_direct",
            "util_coalition",
            "util_algo",
            "avg_utility_direct",
            "avg_utility_coalition",
            "avg_utility_algo",
        }
        assert set(model.reporters.keys()) >= expected

    def test_extremist_reporters(self):
        model = _run_model("direct", extremists="yes")
        expected = {
            "seed",
            "average_moves",
            "average_utility",
            "average_extremist_utility",
            "average_mainstream_utility",
        }
        assert set(model.reporters.keys()) >= expected


class TestSeedReproducibility:
    """Running with the same seed should produce identical results."""

    @pytest.mark.parametrize("institution", ["direct", "coalition", "algorithmic"])
    def test_seed_reproducibility(self, institution):
        model1 = _run_model(institution, seed=1234)
        model2 = _run_model(institution, seed=1234)
        assert model1.reporters["average_moves"] == model2.reporters["average_moves"]
        assert model1.reporters["average_utility"] == model2.reporters["average_utility"]


