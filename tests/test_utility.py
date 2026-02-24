"""Tests for utility computation with proportional vampirism."""

import numpy as np
import pytest

from platform_abm.config import CommunityType
from platform_abm.utility import compute_base_utility, compute_utility
from tests.conftest import make_model


class TestBaseUtility:
    def test_perfect_alignment(self):
        """Base utility equals p_space when preferences match policy."""
        model = make_model({"n_comms": 1, "n_plats": 1})
        comm = model.communities[0]
        plat = model.platforms[0]
        plat.policies = comm.preferences.copy()
        assert compute_base_utility(comm, plat) == model.p.p_space

    def test_zero_alignment(self):
        """Base utility is 0 when preferences are opposite of policy."""
        model = make_model({"n_comms": 1, "n_plats": 1})
        comm = model.communities[0]
        plat = model.platforms[0]
        plat.policies = 1 - comm.preferences
        assert compute_base_utility(comm, plat) == 0

    def test_partial_alignment(self):
        """Base utility is between 0 and p_space for mixed alignment."""
        model = make_model({"n_comms": 1, "n_plats": 1})
        comm = model.communities[0]
        plat = model.platforms[0]
        plat.policies = comm.preferences.copy()
        plat.policies[0] = 1 - plat.policies[0]
        assert compute_base_utility(comm, plat) == model.p.p_space - 1

    def test_algorithmic_group_policy(self):
        """Algorithmic platforms use group-specific policy."""
        model = make_model({"institution": "algorithmic", "n_comms": 4, "n_plats": 1})
        plat = model.platforms[0]
        comm = plat.communities[0]
        group = comm.group
        expected_policy = plat.group_policies[group][1]
        assert compute_base_utility(comm, plat) == comm.utility(expected_policy)


class TestMainstreamUtility:
    def test_no_extremists_equals_base(self):
        """With 0 extremists, mainstream utility equals base utility."""
        model = make_model({"n_comms": 5, "n_plats": 1})
        plat = model.platforms[0]
        comm = plat.communities[0]
        base = compute_base_utility(comm, plat)
        full = compute_utility(comm, plat)
        assert full == base

    def test_all_extremist_neighbors(self):
        """With all extremist neighbors, penalty is alpha."""
        model = make_model({"n_comms": 4, "n_plats": 1, "extremists": "yes", "percent_extremists": 100})
        plat = model.platforms[0]
        # Make one mainstream, rest extremist
        model.communities[0].type = CommunityType.MAINSTREAM.value
        for c in model.communities[1:]:
            c.type = CommunityType.EXTREMIST.value
        comm = model.communities[0]
        base = compute_base_utility(comm, plat)
        full = compute_utility(comm, plat)
        assert full == pytest.approx(base - comm.alpha)

    def test_half_extremist_neighbors(self):
        """With 50/50 neighbors, penalty is alpha/2."""
        model = make_model({"n_comms": 5, "n_plats": 1, "alpha": 2.0})
        plat = model.platforms[0]
        # First community is mainstream, 2 mainstream neighbors, 2 extremist neighbors
        comms = list(plat.communities)
        comms[0].type = CommunityType.MAINSTREAM.value
        comms[1].type = CommunityType.MAINSTREAM.value
        comms[2].type = CommunityType.MAINSTREAM.value
        comms[3].type = CommunityType.EXTREMIST.value
        comms[4].type = CommunityType.EXTREMIST.value
        comm = comms[0]
        base = compute_base_utility(comm, plat)
        full = compute_utility(comm, plat)
        # n_ext=2, n_main=2, total=4, penalty = 2.0 * (2/4) = 1.0
        assert full == pytest.approx(base - 1.0)

    def test_solo_mainstream_equals_base(self):
        """Solo community has no neighbors, utility equals base."""
        model = make_model({"n_comms": 1, "n_plats": 1})
        plat = model.platforms[0]
        comm = plat.communities[0]
        base = compute_base_utility(comm, plat)
        full = compute_utility(comm, plat)
        assert full == base


class TestExtremistUtility:
    def test_all_mainstream_neighbors(self):
        """With all mainstream neighbors, bonus is alpha."""
        model = make_model({"n_comms": 4, "n_plats": 1})
        plat = model.platforms[0]
        model.communities[0].type = CommunityType.EXTREMIST.value
        for c in model.communities[1:]:
            c.type = CommunityType.MAINSTREAM.value
        comm = model.communities[0]
        base = compute_base_utility(comm, plat)
        full = compute_utility(comm, plat)
        assert full == pytest.approx(base + comm.alpha)

    def test_all_extremist_neighbors(self):
        """With all extremist neighbors, bonus is 0."""
        model = make_model({"n_comms": 4, "n_plats": 1})
        plat = model.platforms[0]
        for c in model.communities:
            c.type = CommunityType.EXTREMIST.value
        comm = model.communities[0]
        base = compute_base_utility(comm, plat)
        full = compute_utility(comm, plat)
        assert full == pytest.approx(base)

    def test_half_mainstream_neighbors(self):
        """With 50/50 neighbors, bonus is alpha/2."""
        model = make_model({"n_comms": 5, "n_plats": 1, "alpha": 4.0})
        plat = model.platforms[0]
        comms = list(plat.communities)
        comms[0].type = CommunityType.EXTREMIST.value
        comms[1].type = CommunityType.MAINSTREAM.value
        comms[2].type = CommunityType.MAINSTREAM.value
        comms[3].type = CommunityType.EXTREMIST.value
        comms[4].type = CommunityType.EXTREMIST.value
        comm = comms[0]
        base = compute_base_utility(comm, plat)
        full = compute_utility(comm, plat)
        # n_main=2, n_ext=2, total=4, bonus = 4.0 * (2/4) = 2.0
        assert full == pytest.approx(base + 2.0)

    def test_solo_extremist_equals_base(self):
        """Solo extremist has no neighbors, utility equals base."""
        model = make_model({"n_comms": 1, "n_plats": 1})
        plat = model.platforms[0]
        comm = plat.communities[0]
        comm.type = CommunityType.EXTREMIST.value
        base = compute_base_utility(comm, plat)
        full = compute_utility(comm, plat)
        assert full == base


class TestUtilityBounds:
    def test_mainstream_lower_bound(self):
        """Mainstream utility is at least base - alpha."""
        model = make_model({"n_comms": 10, "n_plats": 1, "extremists": "yes", "percent_extremists": 50})
        plat = model.platforms[0]
        for comm in plat.communities:
            if comm.type == CommunityType.MAINSTREAM.value:
                base = compute_base_utility(comm, plat)
                full = compute_utility(comm, plat)
                assert full >= base - comm.alpha

    def test_extremist_upper_bound(self):
        """Extremist utility is at most base + alpha."""
        model = make_model({"n_comms": 10, "n_plats": 1, "extremists": "yes", "percent_extremists": 50})
        plat = model.platforms[0]
        for comm in plat.communities:
            if comm.type == CommunityType.EXTREMIST.value:
                base = compute_base_utility(comm, plat)
                full = compute_utility(comm, plat)
                assert full <= base + comm.alpha


class TestDivisionByZero:
    def test_no_crash_solo(self):
        """No crash when community is alone on platform."""
        model = make_model({"n_comms": 1, "n_plats": 1})
        comm = model.communities[0]
        plat = model.platforms[0]
        # Should not raise
        result = compute_utility(comm, plat)
        assert isinstance(result, float)

    def test_no_crash_extremist_solo(self):
        """No crash when extremist is alone."""
        model = make_model({"n_comms": 1, "n_plats": 1})
        comm = model.communities[0]
        comm.type = CommunityType.EXTREMIST.value
        plat = model.platforms[0]
        result = compute_utility(comm, plat)
        assert isinstance(result, float)


class TestAlphaValues:
    @pytest.mark.parametrize("alpha", [2.0, 5.0, 10.0])
    def test_different_alpha_scales_penalty(self, alpha):
        """Proportional penalty scales with alpha."""
        model = make_model({"n_comms": 3, "n_plats": 1, "alpha": alpha})
        plat = model.platforms[0]
        comms = list(plat.communities)
        comms[0].type = CommunityType.MAINSTREAM.value
        comms[1].type = CommunityType.EXTREMIST.value
        comms[2].type = CommunityType.MAINSTREAM.value
        comm = comms[0]
        comm.alpha = alpha
        base = compute_base_utility(comm, plat)
        full = compute_utility(comm, plat)
        # n_ext=1, n_main=1, total=2, penalty = alpha * (1/2) = alpha/2
        assert full == pytest.approx(base - alpha / 2)
