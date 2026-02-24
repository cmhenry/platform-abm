"""Abstract base class for institution strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from platform_abm.agents.community import Community
    from platform_abm.agents.platform import Platform


class Institution(ABC):
    """Base class for platform institution strategies."""

    @abstractmethod
    def run_election(self, platform: Platform) -> None:
        """Run the election mechanism and update platform policies."""

    @abstractmethod
    def get_policy_for_community(self, platform: Platform, community: Community) -> Any:
        """Return the policy array relevant to a specific community."""
