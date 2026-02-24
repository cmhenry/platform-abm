"""Enums, constants, and Pydantic configuration for the simulation."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, field_validator, model_validator


class InstitutionType(str, Enum):
    DIRECT = "direct"
    COALITION = "coalition"
    ALGORITHMIC = "algorithmic"
    MIXED = "mixed"


class CommunityType(str, Enum):
    MAINSTREAM = "mainstream"
    EXTREMIST = "extremist"


class Strategy(str, Enum):
    STAY = "stay"
    MOVE = "move"
    UNSET = ""


class StopCondition(str, Enum):
    STEPS = "steps"
    SATISFICED = "satisficed"


# Named constants
MAJORITY_THRESHOLD = 0.5
COLD_START_BUNDLE_COUNT = 5
VAMPIRISM_GAIN = 1
VAMPIRISM_LOSS = 1
INSTITUTION_TYPE_COUNT = 3


class SimulationConfig(BaseModel):
    """Pydantic model for simulation parameters with validation."""

    n_comms: int
    n_plats: int
    p_space: int
    p_type: str = "binary"
    steps: int = 50
    institution: InstitutionType
    extremists: bool = False
    percent_extremists: int = 5
    coalitions: int = 3
    mutations: int = 2
    search_steps: int = 10
    svd_groups: int = 3
    stop_condition: StopCondition = StopCondition.STEPS
    alpha: float = 1.0
    initial_distribution: str = "random"
    seed: int | None = None

    @field_validator("institution", mode="before")
    @classmethod
    def normalize_institution(cls, v: str) -> str:
        if v == "algorithm":
            return "algorithmic"
        return v

    @field_validator("extremists", mode="before")
    @classmethod
    def normalize_extremists(cls, v: Any) -> bool:
        if isinstance(v, str):
            return v.lower() == "yes"
        return bool(v)

    @field_validator("stop_condition", mode="before")
    @classmethod
    def normalize_stop_condition(cls, v: str) -> str:
        return v.lower()

    @model_validator(mode="after")
    def validate_counts(self) -> SimulationConfig:
        if self.n_comms < 1:
            raise ValueError("n_comms must be >= 1")
        if self.n_plats < 1:
            raise ValueError("n_plats must be >= 1")
        if self.p_space < 1:
            raise ValueError("p_space must be >= 1")
        return self

    def to_agentpy_params(self) -> dict[str, Any]:
        """Convert to dict compatible with AgentPy parameters."""
        return {
            "n_comms": self.n_comms,
            "n_plats": self.n_plats,
            "p_space": self.p_space,
            "p_type": self.p_type,
            "steps": self.steps,
            "institution": self.institution.value,
            "extremists": "yes" if self.extremists else "no",
            "percent_extremists": self.percent_extremists,
            "coalitions": self.coalitions,
            "mutations": self.mutations,
            "search_steps": self.search_steps,
            "svd_groups": self.svd_groups,
            "stop_condition": self.stop_condition.value,
            "alpha": self.alpha,
            "initial_distribution": self.initial_distribution,
            **({"seed": self.seed} if self.seed is not None else {}),
        }
