"""Platform agent class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import agentpy as ap
import numpy as np
from numpy.typing import NDArray

from platform_abm.institutions import INSTITUTION_REGISTRY
from platform_abm.institutions.base import Institution

if TYPE_CHECKING:
    from platform_abm.agents.community import Community


class Platform(ap.Agent):
    """Platform agent that hosts communities and runs elections."""

    _type = "platform"

    institution: str
    policies: Any  # NDArray or list
    communities: list[Community]
    community_preferences: NDArray[np.int_]
    institution_strategy: Institution
    group_policies: dict[int, tuple[int, NDArray[np.int_]]]
    grouped_communities: list[list[Community]]

    def setup(self) -> None:
        """Initialize platform variables."""
        self.institution = ""
        self.policies = []
        self.communities = []
        self.community_preferences = np.array([], dtype=int)
        self.group_policies = {}
        self.grouped_communities = []
        self.institution_strategy = None  # type: ignore[assignment]

    def set_institution(self, institution_name: str) -> None:
        """Set the institution type and strategy."""
        self.institution = institution_name
        if institution_name in INSTITUTION_REGISTRY:
            self.institution_strategy = INSTITUTION_REGISTRY[institution_name]()

    def add_community(self, community: Community) -> None:
        """Add community to platform."""
        self.communities.append(community)

    def rm_community(self, community: Community) -> None:
        """Remove community from platform."""
        self.communities.remove(community)

    def aggregate_preferences(self) -> None:
        """Build an array of community preferences."""
        self.community_preferences = np.zeros(
            shape=(len(self.communities), self.p.p_space), dtype=int
        )
        for idx, community in enumerate(self.communities):
            self.community_preferences[idx] = community.preferences

    def election(self) -> None:
        """Run the election mechanism via the institution strategy."""
        if self.institution_strategy is not None:
            self.institution_strategy.run_election(self)
