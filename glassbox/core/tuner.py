"""High-level ModelTuner API."""
from __future__ import annotations

import json
from typing import Any

from ..config.defaults import SEARCH_SPACES
from ..tracking.wandb_tracker import WandbTracker
from ..utils.gpu import is_gpu_available, supports_gpu
from ..logger import logger
from .search import (
    TrialResult,
    grid_search,
    optuna_search,
    random_search,
)


class ModelTuner:
    """Orchestrates hyperparameter search with optional tracking and UI."""

    def __init__(
        self,
        model: Any,
        strategy: str = "random",
        tracking: str | None = None,
        dashboard: bool = False,
        enable_gpu: bool = False,
    ) -> None:
        self.model = model
        self.strategy = strategy
        self.search_space = SEARCH_SPACES.get(model.__class__.__name__, {})
        self.tracker = WandbTracker() if tracking == "wandb" else None
        self.dashboard = dashboard
        self.enable_gpu = enable_gpu

        if enable_gpu:
            if not is_gpu_available():
                raise RuntimeError("GPU requested but none detected")
            if not supports_gpu(model):
                raise RuntimeError("Model does not appear to support GPU")

    def _run_search(self, X, y, show_progress: bool) -> list[TrialResult]:
        kwargs = {"show_progress": show_progress}
        if self.strategy == "grid":
            return grid_search(self.model, X, y, self.search_space, **kwargs)
        if self.strategy == "optuna":
            return optuna_search(self.model, X, y, self.search_space, **kwargs)
        return random_search(self.model, X, y, self.search_space, **kwargs)

    def search(self, X, y, time_limit: str = "10m"):
        if self.tracker:
            self.tracker.start({"strategy": self.strategy})
        show_progress = not self.dashboard
        results = self._run_search(X, y, show_progress)
        for res in results:
            if self.tracker:
                self.tracker.log(res.trial_id, res.metrics)
        if self.tracker:
            self.tracker.finish()
        if self.dashboard:
            with open(logger.state_path, "w") as f:
                json.dump([r.__dict__ for r in results], f)
        best = max(results, key=lambda r: r.metrics.get("score", 0.0))
        if show_progress:
            logger.log(f"Best trial {best.trial_id} with params {best.params}")
        best_model = self.model.__class__(**{**self.model.get_params(), **best.params})
        best_model.fit(X, y)
        return best_model
