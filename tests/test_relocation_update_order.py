"""Tests for simultaneous vs staggered relocation update order."""

from __future__ import annotations

from types import MethodType

from platform_abm.config import Strategy
from tests.conftest import make_model


def _put_all_communities_on_first_platform(model):
    """Reset platform membership so every community starts on platform 0."""
    source, destination = list(model.platforms)
    for platform in model.platforms:
        platform.communities.clear()
    for community in model.communities:
        community.platform = source
        source.add_community(community)
    return source, destination


def test_staggered_relocation_recomputes_and_applies_moves_immediately():
    """Later agents in staggered order should observe earlier relocations."""
    model = make_model({
        "n_comms": 2,
        "n_plats": 2,
        "relocation_update_order": "staggered",
    })
    source, destination = _put_all_communities_on_first_platform(model)
    observed_source_sizes = []

    for community in model.communities:
        def update_utility(self):
            return None

        def set_strategy(self):
            observed_source_sizes.append(len(source.communities))
            self.strategy = Strategy.MOVE.value
            self._search_destination = destination

        community.update_utility = MethodType(update_utility, community)
        community.set_strategy = MethodType(set_strategy, community)

    model._step_relocation()

    assert observed_source_sizes == [2, 1]
    assert len(source.communities) == 0
    assert len(destination.communities) == 2
    assert model._last_n_relocations == 2


def test_simultaneous_relocation_uses_precomputed_move_set():
    """Default relocation should preserve the existing two-phase batch semantics."""
    model = make_model({"n_comms": 2, "n_plats": 2})
    source, destination = _put_all_communities_on_first_platform(model)

    for community in model.communities:
        community.strategy = Strategy.MOVE.value
        community._search_destination = destination

    model._step_relocation()

    assert len(source.communities) == 0
    assert len(destination.communities) == 2
    assert model._last_n_relocations == 2
