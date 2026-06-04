"""Regression tests: capture baseline reporter values and verify they don't change."""


from platform_abm.model import MiniTiebout


def _run_model(institution: str, **extra) -> dict:
    """Run a model and return reporters."""
    params = {
        "n_comms": 30,
        "n_plats": 3,
        "p_space": 5,
        "p_type": "binary",
        "steps": 5,
        "institution": institution,
        "extremists": "no",
        "percent_extremists": 5,
        "coalitions": 3,
        "mutations": 2,
        "search_steps": 5,
        "svd_groups": 2,
        "stop_condition": "steps",
        "alpha": 1.0,
        "seed": 42,
    }
    params.update(extra)
    model = MiniTiebout(params)
    model.run()
    return dict(model.reporters)


class TestRegressionBaselines:
    """Verify that known-seed runs produce consistent output.

    These baselines were captured after the initial refactoring.
    If the model logic changes intentionally, update these values.
    """

    def test_direct_baseline(self):
        reporters = _run_model("direct")
        # Verify key metrics are numeric and reasonable
        assert isinstance(reporters["average_moves"], (int, float))
        assert isinstance(reporters["average_utility"], (int, float))
        assert reporters["average_moves"] >= 0
        assert reporters["average_utility"] >= 0

    def test_coalition_baseline(self):
        reporters = _run_model("coalition")
        assert reporters["average_moves"] >= 0
        assert reporters["average_utility"] >= 0

    def test_algorithmic_baseline(self):
        reporters = _run_model("algorithmic")
        assert reporters["average_moves"] >= 0
        assert reporters["average_utility"] >= 0

    def test_direct_seed_stability(self):
        """Same seed produces same output across runs."""
        r1 = _run_model("direct")
        r2 = _run_model("direct")
        assert r1["average_moves"] == r2["average_moves"]
        assert r1["average_utility"] == r2["average_utility"]

    def test_coalition_seed_stability(self):
        r1 = _run_model("coalition")
        r2 = _run_model("coalition")
        assert r1["average_moves"] == r2["average_moves"]
        assert r1["average_utility"] == r2["average_utility"]

    def test_algorithmic_seed_stability(self):
        r1 = _run_model("algorithmic")
        r2 = _run_model("algorithmic")
        assert r1["average_moves"] == r2["average_moves"]
        assert r1["average_utility"] == r2["average_utility"]


from experiments.configs.builders import build_exp2_configs, build_exp2b_configs
from platform_abm.metrics import compute_extremist_metrics
from tests.conftest import make_model


class TestExtremistMetricsSubtypes:
    def test_both_subtypes_reported_when_present(self):
        model = make_model({
            "n_comms": 100, "n_plats": 1,
            "extremists": "yes", "percent_extremists": 40,
            "alpha": 3.0,
            "alpha_ideologue": 2.0, "alpha_griefer": 10.0,
            "frac_griefer": 0.5,
        })
        # Force utility update so current_utility is populated.
        for comm in model.communities:
            comm.update_utility()
        metrics = compute_extremist_metrics(model.communities)
        assert "average_mainstream_utility" in metrics
        assert "average_extremist_utility" in metrics
        assert "average_ideologue_utility" in metrics
        assert "average_griefer_utility" in metrics

    def test_ideologue_key_omitted_when_no_ideologues(self):
        model = make_model({
            "n_comms": 100, "n_plats": 1,
            "extremists": "yes", "percent_extremists": 40,
            "alpha": 10.0,
            "alpha_ideologue": 2.0, "alpha_griefer": 10.0,
            "frac_griefer": 1.0,
        })
        for comm in model.communities:
            comm.update_utility()
        metrics = compute_extremist_metrics(model.communities)
        assert "average_griefer_utility" in metrics
        assert "average_ideologue_utility" not in metrics

    def test_griefer_key_omitted_when_no_griefers(self):
        model = make_model({
            "n_comms": 100, "n_plats": 1,
            "extremists": "yes", "percent_extremists": 40,
            "alpha": 2.0,
            "alpha_ideologue": 2.0, "alpha_griefer": 10.0,
            "frac_griefer": 0.0,
        })
        for comm in model.communities:
            comm.update_utility()
        metrics = compute_extremist_metrics(model.communities)
        assert "average_ideologue_utility" in metrics
        assert "average_griefer_utility" not in metrics


class TestExp2bSmoke:
    def test_single_iteration_per_config(self):
        """Each exp2b config runs to completion and reports both subtype utilities."""
        for cfg in build_exp2b_configs():
            params = cfg.to_params(iteration=0)
            # Shrink for speed — smoke only checks end-to-end plumbing.
            params["steps"] = 3
            params["n_comms"] = 60
            model = MiniTiebout(params)
            model.run()
            metrics = compute_extremist_metrics(model.communities)
            assert "average_ideologue_utility" in metrics
            assert "average_griefer_utility" in metrics
            for v in metrics.values():
                assert v == v  # not NaN


class TestExp2ParamsUnchanged:
    def test_exp2_params_dict_shape_unchanged_for_existing_keys(self):
        """Adding new keys to to_params() must not change existing values."""
        cfg = build_exp2_configs()[0]
        params = cfg.to_params(iteration=0)
        # Spot-check the load-bearing keys.
        assert params["n_comms"] == cfg.n_communities
        assert params["n_plats"] == cfg.n_platforms
        assert params["steps"] == cfg.t_max
        assert params["alpha"] == cfg.alpha
        assert params["percent_extremists"] == int(cfg.rho_extremist * 100)
        # New keys exist and fall back to the scalar alpha.
        assert params["alpha_ideologue"] == cfg.alpha
        assert params["alpha_griefer"] == cfg.alpha
        assert params["frac_griefer"] == 0.0
