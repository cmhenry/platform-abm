"""Community agent class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import agentpy as ap
import numpy as np
from numpy.typing import NDArray

from platform_abm.config import (
    VAMPIRISM_GAIN,
    VAMPIRISM_LOSS,
    CommunityType,
    InstitutionType,
    Strategy,
)
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

    def utility(self, policies: NDArray[np.int_] | list[int]) -> int:
        """Calculate utility as preference-policy alignment."""
        return int(sum(pref == pol for pref, pol in zip(self.preferences, policies)))

    def update_utility(self) -> None:
        """Update utility from current platform, including vampirism effects."""
        if self.platform.institution == InstitutionType.ALGORITHMIC.value:
            c_util = self.utility(self.platform.group_policies[self.group][1])
        else:
            c_util = self.utility(self.platform.policies)

        e_util = 0
        if self.type == CommunityType.EXTREMIST.value:
            for neighbor in self.platform.communities:
                if neighbor.type == CommunityType.MAINSTREAM.value:
                    e_util += VAMPIRISM_GAIN
        elif self.type == CommunityType.MAINSTREAM.value:
            for neighbor in self.platform.communities:
                if neighbor.type == CommunityType.EXTREMIST.value:
                    e_util -= VAMPIRISM_LOSS

        self.current_utility = c_util + e_util

    def join_platform(self, platform: Platform) -> None:
        """Join a platform."""
        self.moves += 1
        self.platform = platform

    def find_new_platform(self) -> None:
        """Find candidate platforms with higher utility."""
        rng = self.model.random
        self.candidates = []
        for platform in self.model.platforms:
            if platform.institution == InstitutionType.ALGORITHMIC.value:
                if not platform.group_policies:
                    platform.policies = platform.institution_strategy.cold_start_policies(platform)
                    new_policy = platform.policies[rng.randrange(len(platform.policies))]
                    if self.utility(new_policy) > self.current_utility:
                        self.candidates.append(platform)
                else:
                    for group_policy in platform.group_policies.values():
                        new_policy = group_policy[1]
                        if self.utility(new_policy) > self.current_utility:
                            self.candidates.append(platform)
            else:
                new_policy = platform.policies
                if self.utility(new_policy) > self.current_utility:
                    self.candidates.append(platform)
        if not self.candidates:
            self.candidates.append(self.platform)

    def set_strategy(self) -> None:
        """Compare utilities and decide whether to move."""
        rng = self.model.random
        for platform in self.model.platforms:
            if platform.institution == InstitutionType.ALGORITHMIC.value:
                if not platform.group_policies:
                    platform.policies = platform.institution_strategy.cold_start_policies(platform)
                    new_policy = platform.policies[rng.randrange(len(platform.policies))]
                else:
                    keys = list(platform.group_policies.keys())
                    new_policy = platform.group_policies[rng.choice(keys)][1]
            else:
                new_policy = platform.policies

            if self.utility(new_policy) > self.current_utility:
                self.strategy = Strategy.MOVE.value
                return
        self.strategy = Strategy.STAY.value
