"""Tests for Platform agent."""

import numpy as np


class TestPlatformCommunityManagement:
    def test_add_community(self, direct_model):
        """Platform tracks communities that join."""
        plat = direct_model.platforms[0]
        initial_count = len(plat.communities)
        assert initial_count > 0  # communities assigned during setup

    def test_rm_community(self, direct_model):
        """Removing a community decreases count."""
        plat = direct_model.platforms[0]
        if len(plat.communities) > 0:
            comm = plat.communities[0]
            initial = len(plat.communities)
            plat.rm_community(comm)
            assert len(plat.communities) == initial - 1


class TestPlatformAggregatePreferences:
    def test_shape(self, direct_model):
        """Aggregate preferences shape should be (n_communities, p_space)."""
        plat = direct_model.platforms[0]
        plat.aggregate_preferences()
        n_comms = len(plat.communities)
        assert plat.community_preferences.shape == (n_comms, direct_model.p.p_space)

    def test_values_match(self, direct_model):
        """Aggregated preferences should match community preferences."""
        plat = direct_model.platforms[0]
        plat.aggregate_preferences()
        for idx, comm in enumerate(plat.communities):
            np.testing.assert_array_equal(plat.community_preferences[idx], comm.preferences)


class TestPlatformInstitution:
    def test_institution_set(self, direct_model):
        """All platforms have institution set."""
        for plat in direct_model.platforms:
            assert plat.institution == "direct"
            assert plat.institution_strategy is not None

    def test_mixed_institutions(self, mixed_model):
        """Mixed model has multiple institution types."""
        types = {plat.institution for plat in mixed_model.platforms}
        assert len(types) > 1

    def test_election_runs(self, direct_model):
        """Election can run without error."""
        plat = direct_model.platforms[0]
        if len(plat.communities) > 0:
            plat.election()
