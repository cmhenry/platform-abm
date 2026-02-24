"""Tests for the equal initial distribution feature."""

from tests.conftest import make_model


def test_equal_distribution_balanced():
    """Equal distribution with N_c=30, N_p=3 gives exactly 10 per platform."""
    model = make_model({
        "n_comms": 30, "n_plats": 3, "initial_distribution": "equal",
    })
    counts = [len(p.communities) for p in model.platforms]
    assert counts == [10, 10, 10]


def test_equal_distribution_uneven():
    """Equal distribution with N_c=31, N_p=3: max - min <= 1."""
    model = make_model({
        "n_comms": 31, "n_plats": 3, "initial_distribution": "equal",
    })
    counts = [len(p.communities) for p in model.platforms]
    assert max(counts) - min(counts) <= 1
    assert sum(counts) == 31


def test_default_distribution_is_random():
    """Default distribution (no param) uses random assignment."""
    model = make_model({"n_comms": 30, "n_plats": 3})
    counts = [len(p.communities) for p in model.platforms]
    # Random assignment is unlikely to be perfectly balanced; just check all assigned
    assert sum(counts) == 30


def test_equal_distribution_single_platform():
    """Equal distribution with N_p=1 puts all communities on one platform."""
    model = make_model({
        "n_comms": 20, "n_plats": 1, "initial_distribution": "equal",
    })
    counts = [len(p.communities) for p in model.platforms]
    assert counts == [20]
