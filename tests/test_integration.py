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


from platform_abm.config import CommunityType
from tests.conftest import make_model


class TestExtremistSubtypeSplit:
    def test_all_ideologues_when_frac_griefer_zero(self):
        params = {
            "n_comms": 40, "n_plats": 1,
            "extremists": "yes", "percent_extremists": 50,
            "alpha": 3.0,
            "alpha_ideologue": 2.0, "alpha_griefer": 10.0,
            "frac_griefer": 0.0,
        }
        model = make_model(params)
        ext = [c for c in model.communities
               if c.type == CommunityType.EXTREMIST.value]
        assert ext
        assert all(c.subtype == "ideologue" for c in ext)
        assert all(c.alpha == 2.0 for c in ext)

    def test_all_griefers_when_frac_griefer_one(self):
        params = {
            "n_comms": 40, "n_plats": 1,
            "extremists": "yes", "percent_extremists": 50,
            "alpha": 3.0,
            "alpha_ideologue": 2.0, "alpha_griefer": 10.0,
            "frac_griefer": 1.0,
        }
        model = make_model(params)
        ext = [c for c in model.communities
               if c.type == CommunityType.EXTREMIST.value]
        assert ext
        assert all(c.subtype == "griefer" for c in ext)
        assert all(c.alpha == 10.0 for c in ext)

    def test_half_half_split(self):
        params = {
            "n_comms": 100, "n_plats": 1,
            "extremists": "yes", "percent_extremists": 40,
            "alpha": 3.0,
            "alpha_ideologue": 2.0, "alpha_griefer": 10.0,
            "frac_griefer": 0.5,
        }
        model = make_model(params)
        ext = [c for c in model.communities
               if c.type == CommunityType.EXTREMIST.value]
        n_griefer = sum(1 for c in ext if c.subtype == "griefer")
        n_ideologue = sum(1 for c in ext if c.subtype == "ideologue")
        assert n_griefer + n_ideologue == len(ext)
        # 40% of 100 = 40 extremists; 50% griefer = 20; allow ±1 for rounding.
        assert abs(n_griefer - 20) <= 1
        assert abs(n_ideologue - 20) <= 1

    def test_legacy_params_default_to_ideologue(self):
        """When alpha_* and frac_griefer are absent, all extremists become ideologues."""
        params = {
            "n_comms": 40, "n_plats": 1,
            "extremists": "yes", "percent_extremists": 50,
            "alpha": 3.0,
        }
        model = make_model(params)
        ext = [c for c in model.communities
               if c.type == CommunityType.EXTREMIST.value]
        assert ext
        assert all(c.subtype == "ideologue" for c in ext)
        # alpha fallback: community.alpha stays at self.p.alpha (current behavior)
        assert all(c.alpha == 3.0 for c in ext)


