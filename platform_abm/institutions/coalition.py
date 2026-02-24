"""Coalition/movement institution strategy."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from platform_abm.institutions.base import Institution
from platform_abm.utils import generate_binary_preferences

if TYPE_CHECKING:
    from platform_abm.agents.community import Community
    from platform_abm.agents.platform import Platform


class CoalitionInstitution(Institution):
    """Genetic algorithm-style policy evolution with community voting."""

    def create_coalitions(self, platform: Platform) -> None:
        """Generate random coalitions."""
        rng = platform.model.random
        p_space = platform.p.p_space
        n_coalitions = platform.p.coalitions
        platform.coalitions = np.zeros(shape=(n_coalitions, p_space), dtype=int)
        for idx in range(n_coalitions):
            platform.coalitions[idx] = generate_binary_preferences(rng, p_space)

    def fitness(self, platform: Platform, coalition: NDArray[np.int_] | list[int]) -> int:
        """Sum of utility gained from each community for a coalition."""
        fitness = 0
        for community in platform.communities:
            fitness += community.utility(coalition)
        return fitness

    def coalition_mutate(
        self, platform: Platform, coalition: NDArray[np.int_] | list[int]
    ) -> list[int]:
        """Mutate a coalition through hill-climbing search."""
        rng = platform.model.random
        platform.aggregate_preferences()
        current_fitness = self.fitness(platform, coalition)
        best_coalition = list(coalition)

        for _ in range(platform.p.search_steps):
            new_coalition = best_coalition.copy()
            indices = rng.sample(range(len(new_coalition)), platform.p.mutations)
            for idx in indices:
                new_coalition[idx] = new_coalition[idx] ^ 1
            new_fitness = self.fitness(platform, new_coalition)
            if new_fitness > current_fitness:
                best_coalition = new_coalition
                current_fitness = new_fitness

        return best_coalition

    def coalition_poll(self, platform: Platform) -> list[int | None]:
        """Gather coalition votes from communities."""
        votes: list[int | None] = []
        for community in platform.communities:
            min_utility = float("inf")
            vote_index: int | None = None
            for idx, coalition in enumerate(platform.coalitions):
                utility = sum(community.preferences == coalition)
                if utility < min_utility:
                    min_utility = utility
                    vote_index = idx
            votes.append(vote_index)
        return votes

    def run_election(self, platform: Platform) -> None:
        """Create coalitions, mutate, vote, and set winning policy."""
        rng = platform.model.random
        self.create_coalitions(platform)

        # Mutate coalitions - fix: store mutated coalitions back
        for idx in range(len(platform.coalitions)):
            platform.coalitions[idx] = self.coalition_mutate(platform, platform.coalitions[idx])

        votes = self.coalition_poll(platform)
        count = Counter(votes)
        winners = [item for item, freq in count.items() if freq == max(count.values())]
        if not winners:
            new_policies = generate_binary_preferences(rng, platform.p.p_space)
        else:
            winner = rng.choice(winners)
            new_policies = platform.coalitions[winner]
        platform.policies = np.array(new_policies)

    def get_policy_for_community(self, platform: Platform, community: Community) -> Any:
        """Return the platform's single policy slate."""
        return platform.policies
