"""Community agent class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import agentpy as ap
import numpy as np
from numpy.typing import NDArray

from platform_abm.config import CommunityType, Strategy
from platform_abm.utils import generate_binary_preferences

if TYPE_CHECKING:
    from platform_abm.agents.platform import Platform


class Community(ap.Agent):
    """Community agent with binary preferences over a policy space."""

    type: str
    preferences: NDArray[np.int_]
    current_utility: int | float
    platform: Any  # Platform, but typed as Any to avoid circular import issues with agentpy
    strategy: str
    candidates: list[Any]
    group: int | str
    moves: int
    last_move_step: int
    alpha: float

    def setup(self) -> None:
        """Initialize a new community agent."""
        rng = self.model.random
        self.type = CommunityType.MAINSTREAM.value
        self.preferences = generate_binary_preferences(rng, self.p.p_space)
        self.current_utility = 0
        self.platform = ""
        self.strategy = Strategy.UNSET.value
        self.candidates = []
        self.group = ""
        self.moves = 0
        self.last_move_step = 0
        self.alpha = self.p.alpha if hasattr(self.p, 'alpha') else 1.0

    def utility(self, policies: NDArray[np.int_] | list[int]) -> int:
        """Calculate utility as preference-policy alignment."""
        return int(sum(pref == pol for pref, pol in zip(self.preferences, policies)))

    def update_utility(self) -> None:
        """Update utility from current platform, including proportional vampirism."""
        from platform_abm.utility import compute_utility

        self.current_utility = compute_utility(self, self.platform)

    def join_platform(self, platform: Platform) -> None:
        """Join a platform."""
        self.moves += 1
        self.platform = platform

    def find_new_platform(self) -> None:
        """Use stored search result to populate candidates."""
        if hasattr(self, "_search_destination") and self._search_destination is not None:
            self.candidates = [self._search_destination]
        else:
            self.candidates = [self.platform]

    def set_strategy(self) -> None:
        """Delegate to asymmetric search and store the result."""
        from platform_abm.search import search_and_select

        rng = self.model.random
        decision, destination = search_and_select(
            self, self.platform, list(self.model.platforms), rng
        )
        if decision == "move":
            self.strategy = Strategy.MOVE.value
            self._search_destination = destination
        else:
            self.strategy = Strategy.STAY.value
            self._search_destination = None
