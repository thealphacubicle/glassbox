"""Weights & Biases tracking integration."""
from __future__ import annotations

from typing import Any, Dict

from ..utils.lazy_imports import optional_import


class WandbTracker:
    """Thin wrapper around :mod:`wandb` allowing lazy import."""

    def __init__(self) -> None:
        self._wandb = optional_import("wandb")
        self._run = None

    def start(self, config: Dict[str, Any]) -> None:
        self._run = self._wandb.init(project="glassbox", config=config)

    def log(self, trial_id: int, metrics: Dict[str, float]) -> None:
        if self._run is None:
            return
        self._wandb.log({"trial_id": trial_id, **metrics})

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
