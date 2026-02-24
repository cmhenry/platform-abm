"""Tests for institution strategies."""

import numpy as np

from platform_abm.institutions.algorithmic import AlgorithmicInstitution
from platform_abm.institutions.coalition import CoalitionInstitution
from platform_abm.institutions.direct import DirectInstitution
from tests.conftest import make_model


class TestDirectInstitution:
    def test_poll_counts(self):
        """Direct poll returns one vote count per policy."""
        model = make_model({"institution": "direct"})
        inst = DirectInstitution()
        plat = model.platforms[0]
        if len(plat.communities) > 0:
            votes = inst.direct_poll(plat)
            assert len(votes) == len(plat.policies)

    def test_majority_flip(self):
        """Policy should flip when majority of communities prefer opposite."""
        model = make_model({"institution": "direct", "n_comms": 50, "n_plats": 1, "p_space": 3})
        plat = model.platforms[0]
        # Force all communities to prefer [1, 1, 1]
        for comm in plat.communities:
            comm.preferences = np.array([1, 1, 1], dtype=int)
        # Force policy to [0, 0, 0]
        plat.policies = np.array([0, 0, 0], dtype=int)
        inst = DirectInstitution()
        inst.run_election(plat)
        # All policies should flip to 1 (majority prefers 1, vote count for 0 is 0 < threshold)
        # Wait - direct_poll counts votes for 0. If all prefer 1, votes for 0 = 0.
        # threshold = floor(0.5 * 50) = 25. 0 < 25 => flip. policies[i] = 0^1 = 1
        np.testing.assert_array_equal(plat.policies, np.array([1, 1, 1]))

    def test_minority_keep(self):
        """Policy stays when minority wants change."""
        model = make_model({"institution": "direct", "n_comms": 50, "n_plats": 1, "p_space": 3})
        plat = model.platforms[0]
        # Force all communities to prefer [0, 0, 0]
        for comm in plat.communities:
            comm.preferences = np.array([0, 0, 0], dtype=int)
        # Force policy to [0, 0, 0]
        plat.policies = np.array([0, 0, 0], dtype=int)
        inst = DirectInstitution()
        inst.run_election(plat)
        # votes for 0 = 50 >= threshold 25, so no flip
        np.testing.assert_array_equal(plat.policies, np.array([0, 0, 0]))


class TestCoalitionInstitution:
    def test_creation_shape(self):
        """Created coalitions have correct shape."""
        model = make_model({"institution": "coalition"})
        inst = CoalitionInstitution()
        plat = model.platforms[0]
        inst.create_coalitions(plat)
        assert plat.coalitions.shape == (model.p.coalitions, model.p.p_space)

    def test_fitness_is_sum_of_utilities(self):
        """Fitness equals sum of community utilities for a coalition."""
        model = make_model({"institution": "coalition", "n_plats": 1, "p_space": 5})
        inst = CoalitionInstitution()
        plat = model.platforms[0]
        coalition = np.array([1, 1, 1, 1, 1], dtype=int)
        fitness = inst.fitness(plat, coalition)
        expected = sum(c.utility(coalition) for c in plat.communities)
        assert fitness == expected

    def test_mutate_does_not_worsen(self):
        """Mutation should not produce worse fitness than input."""
        model = make_model({"institution": "coalition", "n_plats": 1, "p_space": 5})
        inst = CoalitionInstitution()
        plat = model.platforms[0]
        coalition = np.array([0, 1, 0, 1, 0], dtype=int)
        original_fitness = inst.fitness(plat, coalition)
        mutated = inst.coalition_mutate(plat, coalition)
        mutated_fitness = inst.fitness(plat, mutated)
        assert mutated_fitness >= original_fitness

    def test_poll_selects_indices(self):
        """Poll returns valid coalition indices."""
        model = make_model({"institution": "coalition"})
        inst = CoalitionInstitution()
        plat = model.platforms[0]
        if len(plat.communities) > 0:
            inst.create_coalitions(plat)
            votes = inst.coalition_poll(plat)
            for v in votes:
                assert v is None or 0 <= v < len(plat.coalitions)

    def test_election_produces_valid_policy(self):
        """After election, policies have correct shape."""
        model = make_model({"institution": "coalition"})
        plat = model.platforms[0]
        if len(plat.communities) > 0:
            plat.election()
            assert len(plat.policies) == model.p.p_space


class TestAlgorithmicInstitution:
    def test_group_assignment_complete(self):
        """Every community is assigned to exactly one group."""
        model = make_model({"institution": "algorithmic"})
        inst = AlgorithmicInstitution()
        plat = model.platforms[0]
        if len(plat.communities) > 0:
            inst.group_communities(plat)
            all_grouped = []
            for group in plat.grouped_communities:
                all_grouped.extend(group)
            assert len(all_grouped) == len(plat.communities)

    def test_cold_start_shape(self):
        """Cold start policies have correct shape."""
        model = make_model({"institution": "algorithmic"})
        inst = AlgorithmicInstitution()
        plat = model.platforms[0]
        policies = inst.cold_start_policies(plat)
        assert policies.shape == (5, model.p.p_space)  # COLD_START_BUNDLE_COUNT = 5

    def test_rating_produces_entries(self):
        """Rating produces at least one entry per community."""
        model = make_model({"institution": "algorithmic"})
        inst = AlgorithmicInstitution()
        plat = model.platforms[0]
        if len(plat.communities) > 0:
            inst.group_communities(plat)
            plat.policies = inst.cold_start_policies(plat)
            inst.rate_policies(plat)
            assert len(plat.ui_array) > 0

    def test_one_policy_per_group(self):
        """set_group_policies produces one entry per group."""
        model = make_model({"institution": "algorithmic"})
        inst = AlgorithmicInstitution()
        plat = model.platforms[0]
        if len(plat.communities) > 0:
            inst.group_communities(plat)
            plat.policies = inst.cold_start_policies(plat)
            inst.rate_policies(plat)
            inst.set_group_policies(plat)
            n_groups = len(plat.grouped_communities)
            assert len(plat.group_policies) == n_groups
