"""Utility computation with proportional vampirism."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from platform_abm.config import CommunityType, InstitutionType
from platform_abm.neighbors import get_neighbor_counts

if TYPE_CHECKING:
    from platform_abm.agents.community import Community
    from platform_abm.agents.platform import Platform


def _get_relevant_policy(community: Community, platform: Platform) -> NDArray[np.int_]:
    """Determine which policy vector applies to this community on this platform."""
    if platform.institution == InstitutionType.ALGORITHMIC.value:
        if platform.group_policies and community.group in platform.group_policies:
            return platform.group_policies[community.group][1]
        # Fallback: use first cold-start policy if available
        if hasattr(platform, "policies") and len(platform.policies) > 0:
            if platform.policies.ndim > 1:
                return platform.policies[0]
            return platform.policies
    return platform.policies


def compute_base_utility(community: Community, platform: Platform) -> int:
    """Hamming similarity between community preferences and relevant platform policy."""
    policy = _get_relevant_policy(community, platform)
    return community.utility(policy)


def compute_utility(community: Community, platform: Platform) -> float:
    """Full utility with proportional vampirism.

    Mainstream: u_base - alpha * (n_ext / (n_ext + n_main))
    Extremist:  u_base + alpha * (n_main / (n_main + n_ext))

    Division-by-zero guard: vampirism term is 0 when denominator is 0 (solo community).
    """
    u_base = compute_base_utility(community, platform)
    counts = get_neighbor_counts(community, platform)
    n_main = counts["n_mainstream"]
    n_ext = counts["n_extremist"]
    total = n_main + n_ext
    alpha = getattr(community, "alpha", 1.0)

    if total == 0:
        return float(u_base)

    if community.type == CommunityType.EXTREMIST.value:
        return u_base + alpha * (n_main / total)
    elif community.type == CommunityType.MAINSTREAM.value:
        return u_base - alpha * (n_ext / total)
    else:
        return float(u_base)
