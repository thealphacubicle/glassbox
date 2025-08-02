"""High-level ModelTuner API."""
from __future__ import annotations

import json
import logging
from typing import Any

from ..config.defaults import SEARCH_SPACES
from ..tracking.wandb_tracker import WandbTracker
from ..ui.dashboard import DashboardServer
from ..utils.gpu import is_gpu_available, supports_gpu
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
        self.dashboard = DashboardServer() if dashboard else None
        self.enable_gpu = enable_gpu
        self.logger = logging.getLogger(self.__class__.__name__)
        if not dashboard:
            if not self.logger.hasHandlers():
                handler = logging.StreamHandler()
                handler.setLevel(logging.INFO)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        if enable_gpu:
            if not is_gpu_available():
                raise RuntimeError("GPU requested but none detected")
            if not supports_gpu(model):
                raise RuntimeError("Model does not appear to support GPU")

    def _run_search(self, X, y, show_progress: bool) -> list[TrialResult]:
        kwargs = {"show_progress": show_progress, "logger": self.logger if show_progress else None}
        if self.strategy == "grid":
            return grid_search(self.model, X, y, self.search_space, **kwargs)
        if self.strategy == "optuna":
            return optuna_search(self.model, X, y, self.search_space, **kwargs)
        return random_search(self.model, X, y, self.search_space, **kwargs)

    def search(self, X, y, time_limit: str = "10m"):
        if self.tracker:
            self.tracker.start({"strategy": self.strategy})
        if self.dashboard:
            self.dashboard.run()
        show_progress = self.dashboard is None
        results = self._run_search(X, y, show_progress)
        for res in results:
            if self.tracker:
                self.tracker.log(res.trial_id, res.metrics)
            elif show_progress:
                self.logger.info("Trial %s metrics: %s", res.trial_id, res.metrics)
        if self.tracker:
            self.tracker.finish()
        if self.dashboard:
            with open(self.dashboard.state_path, "w") as f:
                json.dump([r.__dict__ for r in results], f)
        best = max(results, key=lambda r: r.metrics.get("score", 0.0))
        if show_progress:
            self.logger.info("Best trial %s with params %s", best.trial_id, best.params)
        best_model = self.model.__class__(**{**self.model.get_params(), **best.params})
        best_model.fit(X, y)
        return best_model
