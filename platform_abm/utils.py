"""Utility functions for the simulation."""

from __future__ import annotations

import random as _random

import numpy as np
from numpy.typing import NDArray


def generate_binary_preferences(rng: _random.Random, size: int) -> NDArray[np.int_]:
    """Generate a binary preference vector of given size using the provided RNG."""
    return np.array([rng.randint(0, 1) for _ in range(size)], dtype=int)


def generate_zero_preferences(size: int) -> NDArray[np.int_]:
    """Generate an all-zero preference vector (used for extremists)."""
    return np.zeros(size, dtype=int)


def generate_ones_preferences(size: int) -> NDArray[np.int_]:
    """Generate an all-ones preference vector (used for extremists)."""
    return np.ones(size, dtype=int)
