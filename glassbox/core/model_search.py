"""High-level ModelSearch API."""
from __future__ import annotations

from typing import Any, List

from glassbox.plugins import Plugin, PluginManager
from glassbox.tracking.wandb_tracker import WandbTracker
from glassbox.utils.gpu import is_gpu_available, supports_gpu
from glassbox.logger import logger
from glassbox.core.search import Search
from glassbox.schemas import Evaluator


class ModelSearch:
    """Orchestrates hyperparameter search with optional tracking and plugins."""

    def __init__(
        self,
        model: Any,
        search: Search,
        evaluator: Evaluator,
        plugins: List[Plugin] | None = None,
        *,
        tracking: str | None = None,
        enable_gpu: bool = False,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> None:
        self.model = model
        self.searcher = search
        self.evaluator = evaluator
        self.tracker = WandbTracker() if tracking == "wandb" else None
        self.enable_gpu = enable_gpu
        self.verbose = verbose
        self.show_progress = show_progress
        self.plugin_manager = PluginManager()
        for plugin in (plugins or [Plugin()]):
            self.plugin_manager.register(plugin)

        logger.set_verbose(verbose)

        if enable_gpu:
            if not is_gpu_available():
                logger.log("GPU requested but none detected", level="error")
                raise RuntimeError("GPU requested but none detected")
            if not supports_gpu(model):
                logger.log("Model does not appear to support GPU", level="error")
                raise RuntimeError("Model does not appear to support GPU")

    def search(self, X, y):
        if self.tracker:
            self.tracker.start({"strategy": self.searcher.name})
        self.plugin_manager.trigger("on_training_start")
        results = self.searcher.run(
            self.model,
            X,
            y,
            evaluator=self.evaluator,
            show_progress=self.show_progress,
            plugin_manager=self.plugin_manager,
        )
        for res in results:
            if self.tracker:
                self.tracker.log(res.trial_id, res.metrics)
        if self.tracker:
            self.tracker.finish()
        self.plugin_manager.trigger("on_training_end")
        best = max(results, key=lambda r: r.metrics.get("score", 0.0))
        logger.log(f"Best trial {best.trial_id} with params {best.params}")
        best_model = self.model.__class__(**{**self.model.get_params(), **best.params})
        best_model.fit(X, y)
        return best_model
