"""Shared test fixtures."""

import pytest

from platform_abm.model import MiniTiebout


def make_model(overrides: dict | None = None) -> MiniTiebout:
    """Create a MiniTiebout model with default small parameters."""
    params = {
        "n_comms": 20,
        "n_plats": 2,
        "p_space": 5,
        "p_type": "binary",
        "steps": 3,
        "institution": "direct",
        "extremists": "no",
        "percent_extremists": 5,
        "coalitions": 3,
        "mutations": 2,
        "search_steps": 5,
        "svd_groups": 2,
        "stop_condition": "steps",
        "seed": 42,
    }
    if overrides:
        params.update(overrides)
    model = MiniTiebout(params)
    model.sim_setup(steps=params["steps"], seed=params.get("seed"))
    return model


@pytest.fixture
def direct_model() -> MiniTiebout:
    """A set-up model with direct institution."""
    return make_model({"institution": "direct"})


@pytest.fixture
def coalition_model() -> MiniTiebout:
    """A set-up model with coalition institution."""
    return make_model({"institution": "coalition"})


@pytest.fixture
def algorithmic_model() -> MiniTiebout:
    """A set-up model with algorithmic institution."""
    return make_model({"institution": "algorithmic"})


@pytest.fixture
def mixed_model() -> MiniTiebout:
    """A set-up model with mixed institutions."""
    return make_model({"institution": "mixed", "n_plats": 3})


@pytest.fixture
def extremist_model() -> MiniTiebout:
    """A set-up model with extremists enabled."""
    return make_model({"extremists": "yes", "percent_extremists": 20})
