"""Model evaluation helpers."""
from __future__ import annotations

from typing import Any

from glassbox.schemas import Evaluator


class SklearnEvaluator(Evaluator):
    """Default evaluator using ``model.score`` from scikit-learn models."""

    def __init__(self, name: str = "sklearn") -> None:
        super().__init__(name)

    def evaluate(self, model: Any, X, y) -> float:
        return float(model.score(X, y))
