"""Asymmetric search: full utility on current platform vs base-only on destinations."""

from __future__ import annotations

import random as _random
from typing import TYPE_CHECKING

from platform_abm.config import InstitutionType
from platform_abm.utility import compute_base_utility, compute_utility

if TYPE_CHECKING:
    from platform_abm.agents.community import Community
    from platform_abm.agents.platform import Platform


def _evaluate_destination(community: Community, platform: Platform) -> int:
    """Evaluate base utility on a destination platform.

    For algorithmic platforms, evaluates the best across all group policies
    (community can't know which group it'd be assigned to).
    """
    if platform.institution == InstitutionType.ALGORITHMIC.value:
        if platform.group_policies:
            return max(
                community.utility(gp[1])
                for gp in platform.group_policies.values()
            )
        # Cold start: evaluate best across cold-start policy bundles
        if hasattr(platform, "policies") and len(platform.policies) > 0:
            if platform.policies.ndim > 1:
                return max(community.utility(p) for p in platform.policies)
            return community.utility(platform.policies)
    return compute_base_utility(community, platform)


def search_and_select(
    community: Community,
    current_platform: Platform,
    all_platforms: list[Platform],
    rng: _random.Random,
    moving_cost: float = 0.0,
) -> tuple[str, Platform]:
    """Asymmetric search: full utility on current vs base-only on destinations.

    1. Compute full utility (with vampirism) on current platform.
    2. For each other platform, compute base utility only.
    3. Best destination = argmax base utility, ties broken randomly.
    4. MOVE if best_base > current_full + moving_cost, else STAY.

    Returns: ('move', destination) or ('stay', current_platform).
    """
    current_full = compute_utility(community, current_platform)

    best_base = float("-inf")
    best_destinations: list[Platform] = []

    for platform in all_platforms:
        if platform is current_platform:
            continue
        base = _evaluate_destination(community, platform)
        if base > best_base:
            best_base = base
            best_destinations = [platform]
        elif base == best_base:
            best_destinations.append(platform)

    if not best_destinations or best_base <= current_full + moving_cost:
        return ("stay", current_platform)

    destination = rng.choice(best_destinations)
    return ("move", destination)
