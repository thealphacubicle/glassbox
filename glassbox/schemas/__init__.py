"""Pydantic schemas and base classes used across Glassbox."""

from .trial import TrialResult
from .evaluator import Evaluator

__all__ = ["TrialResult", "Evaluator"]
