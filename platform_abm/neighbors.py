"""Governance-aware neighbor resolution for communities on platforms."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

from platform_abm.config import CommunityType, InstitutionType

if TYPE_CHECKING:
    from platform_abm.agents.community import Community
    from platform_abm.agents.platform import Platform


def get_neighbors(community: Community, platform: Platform) -> list[Community]:
    """Return neighboring communities on a platform, excluding self.

    Dispatches by institution type:
    - direct: all other communities on the platform
    - coalition: winning coalition mates (if community voted for winner), else all others
    - algorithmic: group mates only
    """
    institution = platform.institution
    if institution == InstitutionType.COALITION.value:
        return _neighbors_coalition(community, platform)
    elif institution == InstitutionType.ALGORITHMIC.value:
        return _neighbors_algorithmic(community, platform)
    else:
        return _neighbors_direct(community, platform)


def get_neighbor_counts(community: Community, platform: Platform) -> dict[str, int]:
    """Return counts of mainstream and extremist neighbors."""
    neighbors = get_neighbors(community, platform)
    n_mainstream = sum(
        1 for c in neighbors if c.type == CommunityType.MAINSTREAM.value
    )
    n_extremist = sum(
        1 for c in neighbors if c.type == CommunityType.EXTREMIST.value
    )
    return {"n_mainstream": n_mainstream, "n_extremist": n_extremist}


def _neighbors_direct(community: Community, platform: Platform) -> list[Community]:
    """All other communities on the platform."""
    return [c for c in platform.communities if c is not community]


def _neighbors_coalition(community: Community, platform: Platform) -> list[Community]:
    """Winning coalition mates if community voted for winner, else all others."""
    votes = getattr(platform, "coalition_votes", [])
    winner = getattr(platform, "winning_coalition_index", None)

    if not votes or winner is None:
        return _neighbors_direct(community, platform)

    # Find this community's index in the platform's community list
    comm_index = None
    for i, c in enumerate(platform.communities):
        if c is community:
            comm_index = i
            break

    if comm_index is None or comm_index >= len(votes):
        return _neighbors_direct(community, platform)

    if votes[comm_index] == winner:
        # Return only other communities that also voted for the winner
        return [
            c
            for i, c in enumerate(platform.communities)
            if c is not community and i < len(votes) and votes[i] == winner
        ]
    else:
        # Non-winner sees all other communities
        return _neighbors_direct(community, platform)


def _neighbors_algorithmic(community: Community, platform: Platform) -> list[Community]:
    """Group mates only for algorithmic platforms."""
    grouped = getattr(platform, "grouped_communities", [])
    group = getattr(community, "group", "")

    if not grouped or group == "" or not isinstance(group, Integral):
        return _neighbors_direct(community, platform)

    if group < 0 or group >= len(grouped):
        return _neighbors_direct(community, platform)

    return [c for c in grouped[group] if c is not community]
