"""ExperimentConfig dataclass for parameterizing simulation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentConfig:
    """Configuration for a single experimental condition across iterations."""

    name: str
    experiment: str  # "exp1", "exp2", "oat", "interactions"
    n_communities: int
    n_platforms: int
    p_space: int
    t_max: int  # maps to "steps"
    institution: str
    rho_extremist: float  # maps to percent_extremists = int(rho * 100)
    alpha: float
    mu: float = 0.05
    coalitions: int = 5
    mutations: int = 3
    svd_groups: int = 10
    search_steps: int = 10
    initial_distribution: str = "equal"
    tracking_enabled: bool = False
    n_iterations: int = 200
    seed_base: int = 42

    def to_params(self, iteration: int) -> dict[str, Any]:
        """Convert to AgentPy params dict for a specific iteration."""
        percent = int(self.rho_extremist * 100)
        return {
            "n_comms": self.n_communities,
            "n_plats": self.n_platforms,
            "p_space": self.p_space,
            "p_type": "binary",
            "steps": self.t_max,
            "institution": self.institution,
            "extremists": "yes" if percent > 0 else "no",
            "percent_extremists": percent,
            "coalitions": self.coalitions,
            "mutations": self.mutations,
            "search_steps": self.search_steps,
            "svd_groups": self.svd_groups,
            "stop_condition": "steps",
            "alpha": self.alpha,
            "mu": self.mu,
            "initial_distribution": self.initial_distribution,
            "seed": self.seed_base + iteration,
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe serialization of the config."""
        return {
            "name": self.name,
            "experiment": self.experiment,
            "n_communities": self.n_communities,
            "n_platforms": self.n_platforms,
            "p_space": self.p_space,
            "t_max": self.t_max,
            "institution": self.institution,
            "rho_extremist": self.rho_extremist,
            "alpha": self.alpha,
            "mu": self.mu,
            "coalitions": self.coalitions,
            "mutations": self.mutations,
            "svd_groups": self.svd_groups,
            "search_steps": self.search_steps,
            "initial_distribution": self.initial_distribution,
            "tracking_enabled": self.tracking_enabled,
            "n_iterations": self.n_iterations,
            "seed_base": self.seed_base,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ExperimentConfig:
        """Reconstruct from a JSON-safe dict."""
        return ExperimentConfig(**d)
