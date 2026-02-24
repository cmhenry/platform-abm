"""Algorithmic institution strategy (K-means + personalized recommendations)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from platform_abm.config import COLD_START_BUNDLE_COUNT
from platform_abm.institutions.base import Institution

if TYPE_CHECKING:
    from platform_abm.agents.community import Community
    from platform_abm.agents.platform import Platform


class AlgorithmicInstitution(Institution):
    """K-means clustering of communities + personalized policy recommendations."""

    def group_communities(self, platform: Platform) -> None:
        """Sort communities into groups using K-means."""
        platform.aggregate_preferences()
        n_communities = len(platform.communities)
        svd_groups = min(n_communities, platform.p.svd_groups)

        kmeans = KMeans(n_clusters=svd_groups, random_state=0, n_init=2)
        groups = kmeans.fit_predict(platform.community_preferences)

        platform.grouped_communities = [[] for _ in range(svd_groups)]
        for i, group_id in enumerate(groups):
            platform.grouped_communities[group_id].append(platform.communities[i])
            platform.communities[i].group = group_id

    def cold_start_policies(self, platform: Platform) -> NDArray[np.int_]:
        """Construct random policy bundles for cold start."""
        rng = platform.model.random
        return np.array(
            [
                [rng.randint(0, 1) for _ in range(platform.p.p_space)]
                for _ in range(COLD_START_BUNDLE_COUNT)
            ],
            dtype=int,
        )

    def rate_policies(self, platform: Platform) -> None:
        """Serve policy bundles to community groups and collect ratings."""
        platform.ui_array = []
        for group_idx, group in enumerate(platform.grouped_communities):
            for community in group:
                for bundle in platform.policies:
                    fitness = community.utility(bundle)
                    platform.ui_array.append([community.id, group_idx, bundle, fitness])

    def set_group_policies(self, platform: Platform) -> None:
        """Set the best-rated policy for each group."""
        highest_ratings: dict[int, tuple[int, NDArray[np.int_]]] = {}
        for _community, group, bundle, rating in platform.ui_array:
            if group not in highest_ratings:
                highest_ratings[group] = (rating, bundle)
            else:
                current_rating = highest_ratings[group][0]
                if rating > current_rating:
                    highest_ratings[group] = (rating, bundle)
        platform.group_policies = highest_ratings

    def run_election(self, platform: Platform) -> None:
        """Produce new content slate, cluster, rate, and set group policies."""
        platform.policies = self.cold_start_policies(platform)
        self.group_communities(platform)
        self.rate_policies(platform)
        self.set_group_policies(platform)

    def get_policy_for_community(self, platform: Platform, community: Community) -> Any:
        """Return the group-specific policy for a community."""
        if platform.group_policies and community.group in platform.group_policies:
            return platform.group_policies[community.group][1]  # type: ignore[index]
        return platform.policies[0] if len(platform.policies) > 0 else np.array([])
