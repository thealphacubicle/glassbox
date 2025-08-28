"""Abstract evaluator base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Evaluator(ABC):
    """Base interface for model evaluation."""

    def __init__(self, name: str = "evaluator") -> None:
        self.name = name

    @abstractmethod
    def evaluate(self, model, X, y) -> float:
        """Return a numeric score for *model* on the provided data."""
        raise NotImplementedError
