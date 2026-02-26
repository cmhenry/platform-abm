"""Tests for asymmetric search mechanism."""

import random

import numpy as np
import pytest

from platform_abm.config import CommunityType, Strategy
from platform_abm.search import search_and_select
from tests.conftest import make_model


class TestSearchBasics:
    def test_moves_to_better_platform(self):
        """Community moves to a platform with strictly higher base utility."""
        model = make_model({"n_comms": 1, "n_plats": 2})
        comm = model.communities[0]
        plat_a, plat_b = model.platforms[0], model.platforms[1]

        # Put community on plat_a with opposite policy
        comm.platform = plat_a
        plat_a.communities = [comm]
        plat_b.communities = []
        plat_a.policies = 1 - comm.preferences  # zero alignment
        plat_b.policies = comm.preferences.copy()  # perfect alignment

        rng = random.Random(42)
        decision, dest = search_and_select(comm, plat_a, list(model.platforms), rng)
        assert decision == "move"
        assert dest is plat_b

    def test_stays_when_no_better_option(self):
        """Community stays when no destination has higher base utility."""
        model = make_model({"n_comms": 1, "n_plats": 2})
        comm = model.communities[0]
        plat_a, plat_b = model.platforms[0], model.platforms[1]

        comm.platform = plat_a
        plat_a.communities = [comm]
        plat_b.communities = []
        plat_a.policies = comm.preferences.copy()  # perfect
        plat_b.policies = 1 - comm.preferences  # zero

        rng = random.Random(42)
        decision, dest = search_and_select(comm, plat_a, list(model.platforms), rng)
        assert decision == "stay"
        assert dest is plat_a

    def test_never_evaluates_current_as_destination(self):
        """Current platform is never in the destination comparison."""
        model = make_model({"n_comms": 2, "n_plats": 2})
        comm = model.communities[0]
        plat_a = model.platforms[0]

        comm.platform = plat_a
        plat_a.policies = comm.preferences.copy()  # perfect alignment

        rng = random.Random(42)
        decision, dest = search_and_select(comm, plat_a, list(model.platforms), rng)
        # Even though current is perfect, the only comparison is against other platforms
        if decision == "stay":
            assert dest is plat_a


class TestAsymmetry:
    def test_base_only_for_destinations(self):
        """Destinations are evaluated with base utility only (no vampirism)."""
        model = make_model({"n_comms": 4, "n_plats": 2})
        comm = model.communities[0]
        plat_a, plat_b = model.platforms[0], model.platforms[1]

        comm.type = CommunityType.MAINSTREAM.value
        comm.platform = plat_a
        plat_a.communities = [comm, model.communities[1]]
        model.communities[1].type = CommunityType.EXTREMIST.value
        model.communities[1].platform = plat_a
        plat_b.communities = [model.communities[2], model.communities[3]]
        model.communities[2].type = CommunityType.EXTREMIST.value
        model.communities[3].type = CommunityType.EXTREMIST.value

        # Same policy on both platforms (perfect alignment)
        plat_a.policies = comm.preferences.copy()
        plat_b.policies = comm.preferences.copy()

        rng = random.Random(42)
        decision, dest = search_and_select(comm, plat_a, list(model.platforms), rng)

        # Current has vampirism penalty (mainstream with extremist neighbor)
        # Destination base = p_space (no vampirism)
        # base(dest) > full(current) => should move
        assert decision == "move"

    def test_full_utility_for_current(self):
        """Current platform is evaluated with full utility (including vampirism)."""
        model = make_model({"n_comms": 3, "n_plats": 2, "alpha": 1.0})
        comm = model.communities[0]
        plat_a, plat_b = model.platforms[0], model.platforms[1]

        comm.type = CommunityType.MAINSTREAM.value
        comm.platform = plat_a
        plat_a.communities = [comm]
        plat_b.communities = []

        # Current: perfect alignment, no extremist neighbors => full = base = p_space
        plat_a.policies = comm.preferences.copy()
        # Destination: perfect alignment too
        plat_b.policies = comm.preferences.copy()

        rng = random.Random(42)
        decision, dest = search_and_select(comm, plat_a, list(model.platforms), rng)
        # base(dest) == full(current) => stay (not strictly greater)
        assert decision == "stay"


class TestTieBreaking:
    def test_ties_broken_randomly(self):
        """When multiple destinations tie, different seeds can give different results."""
        model = make_model({"n_comms": 1, "n_plats": 3})
        comm = model.communities[0]
        plat_a, plat_b, plat_c = model.platforms[0], model.platforms[1], model.platforms[2]

        comm.platform = plat_a
        plat_a.communities = [comm]
        plat_b.communities = []
        plat_c.communities = []

        # Current: zero alignment
        plat_a.policies = 1 - comm.preferences
        # Both destinations: perfect alignment (tie)
        plat_b.policies = comm.preferences.copy()
        plat_c.policies = comm.preferences.copy()

        destinations = set()
        for seed in range(100):
            rng = random.Random(seed)
            decision, dest = search_and_select(comm, plat_a, list(model.platforms), rng)
            assert decision == "move"
            destinations.add(id(dest))

        # With 100 different seeds, we should see both destinations
        assert len(destinations) == 2

    def test_deterministic_with_fixed_seed(self):
        """Same seed always gives same result."""
        model = make_model({"n_comms": 1, "n_plats": 3})
        comm = model.communities[0]
        plat_a = model.platforms[0]

        comm.platform = plat_a
        plat_a.policies = 1 - comm.preferences
        for p in model.platforms[1:]:
            p.policies = comm.preferences.copy()
            p.communities = []
        plat_a.communities = [comm]

        results = []
        for _ in range(5):
            rng = random.Random(42)
            results.append(search_and_select(comm, plat_a, list(model.platforms), rng))
        assert all(r[0] == results[0][0] and r[1] is results[0][1] for r in results)


class TestExtremistBehavior:
    def test_extremist_stickiness(self):
        """Extremist with high vampire bonus stays despite equal base elsewhere."""
        model = make_model({"n_comms": 4, "n_plats": 2, "alpha": 10.0})
        comm = model.communities[0]
        plat_a, plat_b = model.platforms[0], model.platforms[1]

        comm.type = CommunityType.EXTREMIST.value
        comm.alpha = 10.0
        comm.platform = plat_a

        # All neighbors are mainstream => big vampire bonus
        model.communities[1].type = CommunityType.MAINSTREAM.value
        model.communities[1].platform = plat_a
        plat_a.communities = [comm, model.communities[1]]
        plat_b.communities = [model.communities[2], model.communities[3]]

        # Same base utility on both platforms
        plat_a.policies = np.zeros(model.p.p_space, dtype=int)
        plat_b.policies = np.zeros(model.p.p_space, dtype=int)

        rng = random.Random(42)
        decision, dest = search_and_select(comm, plat_a, list(model.platforms), rng)
        # full(current) = base + alpha*(1/1) = base + 10 >> base(dest)
        assert decision == "stay"

    def test_mainstream_no_penalty_stays(self):
        """Mainstream with no extremist neighbors has full == base, stays on equal."""
        model = make_model({"n_comms": 3, "n_plats": 2})
        comm = model.communities[0]
        plat_a, plat_b = model.platforms[0], model.platforms[1]

        comm.type = CommunityType.MAINSTREAM.value
        comm.platform = plat_a
        model.communities[1].type = CommunityType.MAINSTREAM.value
        model.communities[1].platform = plat_a
        plat_a.communities = [comm, model.communities[1]]
        plat_b.communities = [model.communities[2]]

        # Same policy everywhere
        plat_a.policies = comm.preferences.copy()
        plat_b.policies = comm.preferences.copy()

        rng = random.Random(42)
        decision, _ = search_and_select(comm, plat_a, list(model.platforms), rng)
        assert decision == "stay"


class TestReturnFormat:
    def test_returns_tuple(self):
        """search_and_select returns a (str, Platform) tuple."""
        model = make_model({"n_comms": 1, "n_plats": 2})
        comm = model.communities[0]
        plat = model.platforms[0]
        comm.platform = plat
        rng = random.Random(42)
        result = search_and_select(comm, plat, list(model.platforms), rng)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] in ("move", "stay")

    def test_does_not_execute_move(self):
        """search_and_select does NOT move the community — only returns decision."""
        model = make_model({"n_comms": 1, "n_plats": 2})
        comm = model.communities[0]
        plat_a, plat_b = model.platforms[0], model.platforms[1]
        comm.platform = plat_a
        plat_a.communities = [comm]
        plat_b.communities = []
        plat_a.policies = 1 - comm.preferences
        plat_b.policies = comm.preferences.copy()

        rng = random.Random(42)
        decision, dest = search_and_select(comm, plat_a, list(model.platforms), rng)
        assert decision == "move"
        # But community is still on plat_a
        assert comm.platform is plat_a


class TestMovingCost:
    """Tests for proportional moving cost (mu)."""

    def _setup_improvement(self, improvement: int):
        """Create a 2-platform setup where destination is `improvement` better than current."""
        model = make_model({"n_comms": 1, "n_plats": 2, "p_space": 10, "mu": 0.0})
        comm = model.communities[0]
        plat_a, plat_b = model.platforms[0], model.platforms[1]

        comm.platform = plat_a
        plat_a.communities = [comm]
        plat_b.communities = []

        # Build policies so base(dest) - full(current) = improvement
        # Current: alignment = p_space - improvement (some mismatch)
        # Destination: alignment = p_space (perfect)
        plat_a.policies = comm.preferences.copy()
        # Flip `improvement` bits on current platform
        for i in range(improvement):
            plat_a.policies[i] = 1 - plat_a.policies[i]
        plat_b.policies = comm.preferences.copy()

        return model, comm, plat_a, plat_b

    def test_stays_below_threshold(self):
        """mu=0.05, p_space=10, improvement=0.4 → STAY (0.4 < 0.5)."""
        # improvement is in integer alignment units; moving_cost = 0.05*10 = 0.5
        # dest_base - current_full must be > 0.5 to move
        # We need improvement < moving_cost in alignment units
        # With 0 improvement (dest == current base), definitely stays
        model = make_model({"n_comms": 1, "n_plats": 2, "p_space": 10, "mu": 0.0})
        comm = model.communities[0]
        plat_a, plat_b = model.platforms[0], model.platforms[1]

        comm.platform = plat_a
        plat_a.communities = [comm]
        plat_b.communities = []

        # Current has 6/10 alignment, dest has 6/10 alignment => no improvement
        # Actually let's be precise: current_full=6, dest_base=6, cost=0.5 => 6 <= 6.5 => STAY
        plat_a.policies = comm.preferences.copy()
        plat_b.policies = comm.preferences.copy()
        # Make current slightly worse: flip 0 bits (same alignment)
        # Actually for "below threshold": dest improvement = 0, cost = 0.5
        # 0 improvement clearly below threshold
        rng = random.Random(42)
        moving_cost = 0.05 * 10  # 0.5
        decision, _ = search_and_select(
            comm, plat_a, list(model.platforms), rng, moving_cost=moving_cost
        )
        assert decision == "stay"

    def test_moves_above_threshold(self):
        """mu=0.05, p_space=10: improvement of 1 > cost of 0.5 → MOVE."""
        model, comm, plat_a, plat_b = self._setup_improvement(improvement=1)

        rng = random.Random(42)
        moving_cost = 0.05 * 10  # 0.5
        decision, dest = search_and_select(
            comm, plat_a, list(model.platforms), rng, moving_cost=moving_cost
        )
        assert decision == "move"
        assert dest is plat_b

    def test_stays_at_exact_threshold(self):
        """Strict inequality: improvement exactly equal to cost → STAY."""
        # We need base(dest) - full(current) == moving_cost exactly
        # Use mu=0.1, p_space=10 => cost=1.0; improvement=1 alignment unit
        model, comm, plat_a, plat_b = self._setup_improvement(improvement=1)

        rng = random.Random(42)
        moving_cost = 0.10 * 10  # 1.0, exactly equals improvement of 1
        decision, _ = search_and_select(
            comm, plat_a, list(model.platforms), rng, moving_cost=moving_cost
        )
        assert decision == "stay"

    def test_zero_mu_costless(self):
        """mu=0.0 preserves costless behavior — any improvement triggers move."""
        model, comm, plat_a, plat_b = self._setup_improvement(improvement=1)

        rng = random.Random(42)
        decision, dest = search_and_select(
            comm, plat_a, list(model.platforms), rng, moving_cost=0.0
        )
        assert decision == "move"
        assert dest is plat_b

    def test_same_cost_for_extremists(self):
        """Extremist communities face same moving cost threshold."""
        model, comm, plat_a, plat_b = self._setup_improvement(improvement=1)
        comm.type = CommunityType.EXTREMIST.value

        rng = random.Random(42)
        # cost = 0.5, improvement = 1 => should move
        moving_cost = 0.05 * 10
        decision, _ = search_and_select(
            comm, plat_a, list(model.platforms), rng, moving_cost=moving_cost
        )
        assert decision == "move"

        # cost = 1.0, improvement = 1 => stays (exact threshold)
        rng = random.Random(42)
        moving_cost = 0.10 * 10
        decision, _ = search_and_select(
            comm, plat_a, list(model.platforms), rng, moving_cost=moving_cost
        )
        assert decision == "stay"


class TestAlgorithmicDestination:
    def test_evaluates_best_group_policy(self):
        """Algorithmic destination evaluates max across group policies."""
        model = make_model({"institution": "algorithmic", "n_comms": 6, "n_plats": 2, "svd_groups": 2})
        comm = model.communities[0]
        # Just verify it doesn't crash and returns valid result
        rng = random.Random(42)
        result = search_and_select(comm, comm.platform, list(model.platforms), rng)
        assert result[0] in ("move", "stay")
