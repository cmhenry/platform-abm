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
    """Per-capita utility for mainstream, extremist, and (when present)
    ideologue and griefer subgroups.

    Ideologue/griefer keys are omitted when their subgroup is empty so
    endpoints of the f_g sweep don't trigger ZeroDivisionError.
    """
    extremists = communities.select(communities.type == "extremist")
    mainstream = communities.select(communities.type == "mainstream")
    metrics: dict[str, float] = {
        "average_extremist_utility": sum(extremists.current_utility) / len(extremists),
        "average_mainstream_utility": sum(mainstream.current_utility) / len(mainstream),
    }
    ideologues = communities.select(communities.subtype == "ideologue")
    griefers = communities.select(communities.subtype == "griefer")
    if len(ideologues) > 0:
        metrics["average_ideologue_utility"] = (
            sum(ideologues.current_utility) / len(ideologues)
        )
    if len(griefers) > 0:
        metrics["average_griefer_utility"] = (
            sum(griefers.current_utility) / len(griefers)
        )
    return metrics
