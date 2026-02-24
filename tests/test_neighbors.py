"""Tests for governance-aware neighbor resolution."""

import pytest

from platform_abm.config import CommunityType, InstitutionType
from platform_abm.neighbors import get_neighbor_counts, get_neighbors
from tests.conftest import make_model


class TestDirectNeighbors:
    def test_all_others_are_neighbors(self, direct_model):
        """Every community on a platform sees all others as neighbors."""
        for plat in direct_model.platforms:
            for comm in plat.communities:
                neighbors = get_neighbors(comm, plat)
                assert len(neighbors) == len(plat.communities) - 1

    def test_self_exclusion(self, direct_model):
        """A community is never its own neighbor."""
        for plat in direct_model.platforms:
            for comm in plat.communities:
                neighbors = get_neighbors(comm, plat)
                assert comm not in neighbors

    def test_solo_community(self):
        """A solo community on a platform has no neighbors."""
        model = make_model({"n_comms": 1, "n_plats": 1})
        comm = model.communities[0]
        plat = model.platforms[0]
        neighbors = get_neighbors(comm, plat)
        assert neighbors == []


class TestCoalitionNeighbors:
    def test_winner_sees_only_coalition_mates(self):
        """A community that voted for the winning coalition sees only fellow winners."""
        model = make_model({"institution": "coalition", "n_comms": 6, "n_plats": 1})
        plat = model.platforms[0]
        # Run an election to populate coalition_votes and winning_coalition_index
        plat.election()

        assert plat.coalition_votes
        assert plat.winning_coalition_index is not None

        winner_idx = plat.winning_coalition_index
        winner_communities = [
            plat.communities[i]
            for i, v in enumerate(plat.coalition_votes)
            if v == winner_idx
        ]

        for comm in winner_communities:
            neighbors = get_neighbors(comm, plat)
            # All neighbors should also be winner voters
            for n in neighbors:
                n_idx = plat.communities.index(n)
                assert plat.coalition_votes[n_idx] == winner_idx
            # Should not include self
            assert comm not in neighbors

    def test_non_winner_sees_all(self):
        """A community not in the winning coalition sees all others."""
        model = make_model({"institution": "coalition", "n_comms": 10, "n_plats": 1})
        plat = model.platforms[0]
        plat.election()

        winner_idx = plat.winning_coalition_index
        if winner_idx is None:
            pytest.skip("No winner in this election")

        non_winner_comms = [
            plat.communities[i]
            for i, v in enumerate(plat.coalition_votes)
            if v != winner_idx
        ]
        if not non_winner_comms:
            pytest.skip("All communities voted for the winner")

        comm = non_winner_comms[0]
        neighbors = get_neighbors(comm, plat)
        assert len(neighbors) == len(plat.communities) - 1

    def test_no_election_fallback(self):
        """Before any election, coalition neighbors fall back to direct behavior."""
        model = make_model({"institution": "coalition", "n_comms": 4, "n_plats": 1})
        plat = model.platforms[0]
        # Don't run election — coalition_votes is empty
        comm = plat.communities[0]
        neighbors = get_neighbors(comm, plat)
        assert len(neighbors) == len(plat.communities) - 1


class TestAlgorithmicNeighbors:
    def test_same_group_only(self):
        """Algorithmic neighbors are group mates only."""
        model = make_model({"institution": "algorithmic", "n_comms": 10, "n_plats": 1, "svd_groups": 2})
        plat = model.platforms[0]

        for comm in plat.communities:
            neighbors = get_neighbors(comm, plat)
            group = comm.group
            # All neighbors should be in the same group
            for n in neighbors:
                assert n.group == group
            assert comm not in neighbors

    def test_solo_in_group(self):
        """A community alone in its group has no neighbors."""
        model = make_model({"institution": "algorithmic", "n_comms": 2, "n_plats": 1, "svd_groups": 2})
        plat = model.platforms[0]

        for group_list in plat.grouped_communities:
            if len(group_list) == 1:
                comm = group_list[0]
                neighbors = get_neighbors(comm, plat)
                assert neighbors == []
                return
        pytest.skip("No solo group found with this seed")

    def test_no_group_fallback(self):
        """If no group assignment, fall back to direct behavior."""
        model = make_model({"institution": "algorithmic", "n_comms": 4, "n_plats": 1})
        plat = model.platforms[0]
        comm = plat.communities[0]
        # Reset group to unassigned
        original_group = comm.group
        comm.group = ""
        neighbors = get_neighbors(comm, plat)
        assert len(neighbors) == len(plat.communities) - 1
        comm.group = original_group  # restore


class TestNeighborCounts:
    def test_correct_partition(self):
        """Counts correctly partition neighbors into mainstream and extremist."""
        model = make_model({"extremists": "yes", "percent_extremists": 30, "n_comms": 20, "n_plats": 1})
        plat = model.platforms[0]
        for comm in plat.communities:
            counts = get_neighbor_counts(comm, plat)
            neighbors = get_neighbors(comm, plat)
            assert counts["n_mainstream"] + counts["n_extremist"] == len(neighbors)

    def test_zero_counts_when_solo(self):
        """Solo community has zero for both counts."""
        model = make_model({"n_comms": 1, "n_plats": 1})
        comm = model.communities[0]
        plat = model.platforms[0]
        counts = get_neighbor_counts(comm, plat)
        assert counts["n_mainstream"] == 0
        assert counts["n_extremist"] == 0

    def test_all_mainstream(self, direct_model):
        """With no extremists, all neighbors are mainstream."""
        plat = direct_model.platforms[0]
        if not plat.communities:
            pytest.skip("Empty platform")
        comm = plat.communities[0]
        counts = get_neighbor_counts(comm, plat)
        assert counts["n_extremist"] == 0
        assert counts["n_mainstream"] == len(plat.communities) - 1
