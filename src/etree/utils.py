"""General helper utilities used by examples and search."""

from __future__ import annotations

import numpy as np


def make_grid(start: float = -1.0, stop: float = 1.0, n: int = 200) -> np.ndarray:
    """Create a dense 1D grid for sampling target functions."""
    return np.linspace(start, stop, n)
