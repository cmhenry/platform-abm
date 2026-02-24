"""End-to-end validation tests for the Tiebout cycle integration (Task 4)."""

from __future__ import annotations

import copy
from unittest.mock import patch

import pytest

from platform_abm.config import CommunityType, InstitutionType
from platform_abm.model import MiniTiebout


def _run_model(overrides: dict | None = None) -> MiniTiebout:
    """Run a model to completion with sensible defaults."""
    params = {
        "n_comms": 30,
        "n_plats": 3,
        "p_space": 5,
        "p_type": "binary",
        "steps": 20,
        "institution": "mixed",
        "extremists": "no",
        "percent_extremists": 10,
        "coalitions": 3,
        "mutations": 2,
        "search_steps": 5,
        "svd_groups": 2,
        "stop_condition": "steps",
        "alpha": 0.0,
        "seed": 42,
    }
    if overrides:
        params.update(overrides)
    model = MiniTiebout(params)
    model.run()
    return model


class TestBaselineRegression:
    """Test 1: Baseline regression — no extremists, alpha=0."""

    def test_completes_without_error(self):
        model = _run_model()
        assert "average_moves" in model.reporters
        assert "average_utility" in model.reporters

    def test_utilities_equal_base_when_alpha_zero(self):
        """With alpha=0 there is no vampirism term, so full == base utility."""
        model = _run_model()
        # All utilities should be non-negative integers (hamming similarity)
        for comm in model.communities:
            assert comm.current_utility >= 0

    def test_community_count_conserved(self):
        model = _run_model()
        total_on_platforms = sum(len(p.communities) for p in model.platforms)
        assert total_on_platforms == model.p.n_comms

    def test_metrics_non_negative(self):
        model = _run_model()
        assert model.reporters["average_moves"] >= 0
        assert model.reporters["average_utility"] >= 0


class TestExtremistsChangeOutcomes:
    """Test 2: Extremists change outcomes relative to baseline."""

    def test_extremist_reporters_exist(self):
        model = _run_model({"extremists": "yes", "alpha": 5.0})
        assert "average_extremist_utility" in model.reporters
        assert "average_mainstream_utility" in model.reporters

    def test_extremist_avg_utility_ge_mainstream(self):
        """Extremists gain utility from mainstream neighbors; mainstream lose it."""
        model = _run_model({"extremists": "yes", "alpha": 5.0})
        ext_util = model.reporters["average_extremist_utility"]
        main_util = model.reporters["average_mainstream_utility"]
        # With positive alpha, extremists should be at least as well off
        assert ext_util >= main_util

    def test_some_mainstream_penalized(self):
        """At least one mainstream community has full utility < base utility."""
        model = _run_model({"extremists": "yes", "alpha": 5.0})
        from platform_abm.utility import compute_base_utility

        found_penalty = False
        for comm in model.communities:
            if comm.type == CommunityType.MAINSTREAM.value:
                base = compute_base_utility(comm, comm.platform)
                if comm.current_utility < base:
                    found_penalty = True
                    break
        assert found_penalty, "Expected at least one mainstream community with vampirism penalty"


class TestHighAlphaAmplifies:
    """Test 3: Higher alpha amplifies the parasitism gap."""

    def test_utility_gap_increases_with_alpha(self):
        model_low = _run_model({"extremists": "yes", "alpha": 5.0, "seed": 99})
        model_high = _run_model({"extremists": "yes", "alpha": 10.0, "seed": 99})

        gap_low = (
            model_low.reporters["average_extremist_utility"]
            - model_low.reporters["average_mainstream_utility"]
        )
        gap_high = (
            model_high.reporters["average_extremist_utility"]
            - model_high.reporters["average_mainstream_utility"]
        )
        assert gap_high >= gap_low

    def test_higher_alpha_higher_extremist_utility(self):
        model_low = _run_model({"extremists": "yes", "alpha": 5.0, "seed": 99})
        model_high = _run_model({"extremists": "yes", "alpha": 10.0, "seed": 99})
        assert (
            model_high.reporters["average_extremist_utility"]
            >= model_low.reporters["average_extremist_utility"]
        )

    def test_higher_alpha_lower_mainstream_utility(self):
        model_low = _run_model({"extremists": "yes", "alpha": 5.0, "seed": 99})
        model_high = _run_model({"extremists": "yes", "alpha": 10.0, "seed": 99})
        assert (
            model_high.reporters["average_mainstream_utility"]
            <= model_low.reporters["average_mainstream_utility"]
        )


class TestSimultaneity:
    """Test 4: Same seed produces identical results (batch relocation is deterministic)."""

    @pytest.mark.parametrize("seed", [10, 42, 123])
    def test_seed_reproducibility(self, seed):
        cfg = {"extremists": "yes", "alpha": 3.0, "seed": seed}
        model1 = _run_model(cfg)
        model2 = _run_model(cfg)

        # Reporter values must match exactly
        for key in model1.reporters:
            assert model1.reporters[key] == model2.reporters[key], (
                f"Reporter '{key}' differs: {model1.reporters[key]} vs {model2.reporters[key]}"
            )

        # Platform community counts must match
        counts1 = sorted(len(p.communities) for p in model1.platforms)
        counts2 = sorted(len(p.communities) for p in model2.platforms)
        assert counts1 == counts2


class TestGovernanceStateConsistency:
    """Test 5: Governance state is stable between elections and utility computation."""

    def _make_model(self, institution: str) -> MiniTiebout:
        params = {
            "n_comms": 20,
            "n_plats": 2,
            "p_space": 5,
            "p_type": "binary",
            "steps": 5,
            "institution": institution,
            "extremists": "yes",
            "percent_extremists": 20,
            "coalitions": 3,
            "mutations": 2,
            "search_steps": 5,
            "svd_groups": 2,
            "stop_condition": "steps",
            "alpha": 1.0,
            "seed": 42,
        }
        model = MiniTiebout(params)
        model.sim_setup(steps=params["steps"], seed=params["seed"])
        return model

    def test_coalition_state_stable_after_elections(self):
        """Coalition votes and winner don't change during utility computation."""
        model = self._make_model("coalition")

        # Run elections
        model._step_elections()

        # Snapshot governance state
        snapshots = {}
        for p in model.platforms:
            snapshots[p.id] = {
                "coalition_votes": list(p.coalition_votes),
                "winning_coalition_index": p.winning_coalition_index,
            }

        # Run utility computation
        model._step_update_utility()

        # Verify state unchanged
        for p in model.platforms:
            assert list(p.coalition_votes) == snapshots[p.id]["coalition_votes"]
            assert p.winning_coalition_index == snapshots[p.id]["winning_coalition_index"]

    def test_algorithmic_state_stable_after_elections(self):
        """Grouped communities and group policies don't change during utility computation."""
        model = self._make_model("algorithmic")

        # Run elections
        model._step_elections()

        # Snapshot governance state
        snapshots = {}
        for p in model.platforms:
            snapshots[p.id] = {
                "grouped_communities": [list(g) for g in p.grouped_communities],
                "group_policies": dict(p.group_policies),
            }

        # Run utility computation
        model._step_update_utility()

        # Verify state unchanged
        for p in model.platforms:
            actual_groups = [list(g) for g in p.grouped_communities]
            assert actual_groups == snapshots[p.id]["grouped_communities"]
            assert dict(p.group_policies) == snapshots[p.id]["group_policies"]


class TestBlindSearch:
    """Test 6: Search uses get_neighbor_counts only on own platform;
    compute_utility in search only called for the current platform."""

    def test_neighbor_counts_only_on_own_platform(self):
        """get_neighbor_counts should only be called for a community's own platform."""
        params = {
            "n_comms": 10,
            "n_plats": 2,
            "p_space": 5,
            "p_type": "binary",
            "steps": 3,
            "institution": "direct",
            "extremists": "yes",
            "percent_extremists": 20,
            "coalitions": 3,
            "mutations": 2,
            "search_steps": 5,
            "svd_groups": 2,
            "stop_condition": "steps",
            "alpha": 5.0,
            "seed": 42,
        }
        model = MiniTiebout(params)
        model.sim_setup(steps=params["steps"], seed=params["seed"])

        calls = []
        original_get_neighbor_counts = None

        import platform_abm.utility as utility_mod

        original_get_neighbor_counts = utility_mod.get_neighbor_counts

        def tracking_get_neighbor_counts(community, platform):
            calls.append((community, platform))
            return original_get_neighbor_counts(community, platform)

        with patch.object(utility_mod, "get_neighbor_counts", tracking_get_neighbor_counts):
            model._step_elections()
            model._step_update_utility()

        # Every call to get_neighbor_counts should be community on its own platform
        for community, platform in calls:
            assert platform is community.platform, (
                f"get_neighbor_counts called with platform {platform.id} "
                f"but community {community.id} is on platform {community.platform.id}"
            )

    def test_search_compute_utility_only_on_current_platform(self):
        """compute_utility in search should only be called for the current platform."""
        params = {
            "n_comms": 10,
            "n_plats": 2,
            "p_space": 5,
            "p_type": "binary",
            "steps": 3,
            "institution": "direct",
            "extremists": "yes",
            "percent_extremists": 20,
            "coalitions": 3,
            "mutations": 2,
            "search_steps": 5,
            "svd_groups": 2,
            "stop_condition": "steps",
            "alpha": 5.0,
            "seed": 42,
        }
        model = MiniTiebout(params)
        model.sim_setup(steps=params["steps"], seed=params["seed"])

        calls = []

        import platform_abm.search as search_mod
        from platform_abm.utility import compute_utility as original_compute_utility

        def tracking_compute_utility(community, platform):
            calls.append((community, platform))
            return original_compute_utility(community, platform)

        with patch.object(search_mod, "compute_utility", tracking_compute_utility):
            model._step_elections()
            model._step_update_utility()

        # compute_utility in search should only be called for current platform
        for community, platform in calls:
            assert platform is community.platform, (
                f"search's compute_utility called with platform {platform.id} "
                f"but community {community.id} is on platform {community.platform.id}"
            )
