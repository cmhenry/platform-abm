"""Direct voting institution strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from platform_abm.config import MAJORITY_THRESHOLD
from platform_abm.institutions.base import Institution

if TYPE_CHECKING:
    from platform_abm.agents.community import Community
    from platform_abm.agents.platform import Platform


class DirectInstitution(Institution):
    """Majority voting on individual policies."""

    def direct_poll(self, platform: Platform) -> list[int]:
        """Poll individual policies and return vote counts for policy=0."""
        n_policies = len(platform.policies)
        platform.aggregate_preferences()
        votes = [0] * n_policies
        for col_idx in range(n_policies):
            votes[col_idx] = int(np.sum(platform.community_preferences[:, col_idx] == 0))
        return votes

    def run_election(self, platform: Platform) -> None:
        """Gather votes and flip policies below majority threshold."""
        votes = self.direct_poll(platform)
        threshold = np.floor(MAJORITY_THRESHOLD * len(platform.communities))
        num_policies = len(platform.policies)
        for i in range(num_policies):
            if votes[i] < threshold:
                platform.policies[i] = platform.policies[i] ^ 1

    def get_policy_for_community(self, platform: Platform, community: Community) -> Any:
        """Return the platform's single policy slate."""
        return platform.policies
