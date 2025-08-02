"""Model evaluation helpers."""
from __future__ import annotations

from typing import Any


def evaluate(model: Any, X, y) -> float:
    """Evaluate *model* on the provided data using ``model.score``."""
    return float(model.score(X, y))
