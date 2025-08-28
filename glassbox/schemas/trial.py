"""Schema models for search trials."""

from __future__ import annotations

from typing import Any, Dict
from pydantic import BaseModel


class TrialResult(BaseModel):
    """Pydantic model capturing the outcome of a single search trial."""

    trial_id: int
    params: Dict[str, Any]
    metrics: Dict[str, float]
    duration: float
