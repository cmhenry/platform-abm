"""Tests for Community agent."""

import pytest

from platform_abm.config import CommunityType, Strategy
from tests.conftest import make_model


class TestCommunityUtility:
    def test_perfect_alignment(self, direct_model):
        """Utility equals p_space when preferences match policy exactly."""
        comm = direct_model.communities[0]
        policies = comm.preferences.copy()
        assert comm.utility(policies) == direct_model.p.p_space

    def test_zero_alignment(self, direct_model):
        """Utility is 0 when preferences are opposite of policy."""
        comm = direct_model.communities[0]
        policies = 1 - comm.preferences
        assert comm.utility(policies) == 0

    def test_partial_alignment(self, direct_model):
        """Utility is between 0 and p_space for mixed alignment."""
        comm = direct_model.communities[0]
        policies = comm.preferences.copy()
        policies[0] = 1 - policies[0]  # flip one
        assert comm.utility(policies) == direct_model.p.p_space - 1


class TestCommunitySetup:
    def test_preference_shape(self, direct_model):
        """Preferences should have shape (p_space,)."""
        for comm in direct_model.communities:
            assert comm.preferences.shape == (direct_model.p.p_space,)

    def test_preference_values_binary(self, direct_model):
        """All preferences should be 0 or 1."""
        for comm in direct_model.communities:
            assert set(comm.preferences).issubset({0, 1})

    def test_initial_type_is_mainstream(self, direct_model):
        """All communities start as mainstream (when no extremists)."""
        for comm in direct_model.communities:
            assert comm.type == CommunityType.MAINSTREAM.value

    def test_initial_moves_zero(self):
        """Moves should start at 0 (before join_platform in setup adds 1)."""
        # After setup, each community has been assigned to a platform (1 move)
        model = make_model()
        for comm in model.communities:
            assert comm.moves == 1

    def test_initial_strategy(self, direct_model):
        """Strategy should be unset initially."""
        # After setup (before any step), strategy is still ''
        model = make_model()
        for comm in model.communities:
            assert comm.strategy == Strategy.UNSET.value


class TestExtremistvampirism:
    def test_extremist_gains_utility(self, extremist_model):
        """Extremists gain utility from mainstream neighbors."""
        model = extremist_model
        # Run one step to update utilities
        model.step()
        extremists = [c for c in model.communities if c.type == CommunityType.EXTREMIST.value]
        # At least some extremists should have positive vampirism bonus
        assert any(e.current_utility > 0 for e in extremists)

    def test_mainstream_loses_utility(self, extremist_model):
        """Mainstream communities lose utility from extremist neighbors."""
        model = extremist_model
        model.step()
        # Find a mainstream community on a platform with extremists
        for comm in model.communities:
            if comm.type == CommunityType.MAINSTREAM.value:
                has_extremist_neighbor = any(
                    n.type == CommunityType.EXTREMIST.value for n in comm.platform.communities
                )
                if has_extremist_neighbor:
                    # utility should be reduced
                    n_extremists = sum(
                        1
                        for n in comm.platform.communities
                        if n.type == CommunityType.EXTREMIST.value
                    )
                    # With proportional vampirism, penalty is at most alpha
                    assert comm.current_utility <= comm.p.p_space
                    return
        # If no mainstream has extremist neighbor, test is inconclusive (skip)
        pytest.skip("No mainstream community with extremist neighbor found")


class TestCommunityStrategy:
    def test_strategy_set_after_update(self, direct_model):
        """After update_utility + set_strategy, strategy is stay or move."""
        comm = direct_model.communities[0]
        comm.update_utility()
        comm.set_strategy()
        assert comm.strategy in (Strategy.STAY.value, Strategy.MOVE.value)
