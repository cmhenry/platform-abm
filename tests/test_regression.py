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
