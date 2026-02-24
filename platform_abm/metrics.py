"""Metrics computation functions extracted from Model.end()."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import agentpy as ap


def compute_average_moves(communities: ap.AgentList, n_comms: int) -> float:
    """Compute average number of moves per community."""
    return float(sum(communities.moves) / n_comms)


def compute_average_utility(communities: ap.AgentList, n_comms: int) -> float:
    """Compute average utility per community."""
    return float(sum(communities.current_utility) / n_comms)


def compute_mixed_institution_metrics(
    communities: ap.AgentList, platforms: ap.AgentList, n_comms: int
) -> dict[str, Any]:
    """Compute metrics broken down by institution type for mixed runs."""
    metrics: dict[str, Any] = {}

    n_direct_comms = len(communities.select(communities.platform.institution == "direct"))
    n_coalition_comms = len(communities.select(communities.platform.institution == "coalition"))
    n_algo_comms = len(communities.select(communities.platform.institution == "algorithmic"))

    metrics["n_direct_comms"] = n_direct_comms
    metrics["n_coalition_comms"] = n_coalition_comms
    metrics["n_algo_comms"] = n_algo_comms

    metrics["ratio_direct"] = n_direct_comms / n_comms
    metrics["ratio_coalition"] = n_coalition_comms / n_comms
    metrics["ratio_algo"] = n_algo_comms / n_comms

    util_direct = sum(
        communities.select(communities.platform.institution == "direct").current_utility
    )
    util_coalition = sum(
        communities.select(communities.platform.institution == "coalition").current_utility
    )
    util_algo = sum(
        communities.select(communities.platform.institution == "algorithmic").current_utility
    )

    metrics["util_direct"] = util_direct
    metrics["util_coalition"] = util_coalition
    metrics["util_algo"] = util_algo

    metrics["avg_utility_direct"] = util_direct / n_direct_comms if n_direct_comms else 0
    metrics["avg_utility_coalition"] = (
        util_coalition / n_coalition_comms if n_coalition_comms else 0
    )
    metrics["avg_utility_algo"] = util_algo / n_algo_comms if n_algo_comms else 0

    return metrics


def compute_extremist_metrics(communities: ap.AgentList) -> dict[str, float]:
    """Compute per-capita utility for extremist and mainstream communities."""
    extremists = communities.select(communities.type == "extremist")
    mainstream = communities.select(communities.type == "mainstream")
    return {
        "average_extremist_utility": sum(extremists.current_utility) / len(extremists),
        "average_mainstream_utility": sum(mainstream.current_utility) / len(mainstream),
    }
