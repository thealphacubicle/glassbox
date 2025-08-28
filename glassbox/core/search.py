"""Search strategies for hyperparameter tuning."""
from __future__ import annotations

import itertools
import random
import math
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from glassbox.schemas import Evaluator, TrialResult
from glassbox.utils.lazy_imports import optional_import
from glassbox.logger import logger
from glassbox.plugins.manager import PluginManager


class Search:
    """Encapsulates different hyperparameter search strategies."""

    def __init__(
        self,
        strategy: str,
        search_space: Dict[str, Iterable[Any]],
        *,
        n_trials: int = 10,
        name: str | None = None,
    ) -> None:
        if not search_space:
            logger.log("search_space must be provided", level="error")
            raise ValueError("search_space must be provided")
        self.strategy = strategy
        self.search_space = {k: list(v) for k, v in search_space.items()}
        self.n_trials = n_trials
        self.name = name or strategy
        self._strategies: Dict[
            str,
            Callable[[Any, Any, Any, Evaluator, bool, Optional[PluginManager]], List[TrialResult]],
        ] = {
            "grid": self._grid_search,
            "random": self._random_search,
            "optuna": self._optuna_search,
        }
        if strategy not in self._strategies:
            logger.log(f"Unknown search strategy: {strategy}", level="error")
            raise ValueError(f"Unknown search strategy: {strategy}")

    def run(
        self,
        model,
        X,
        y,
        evaluator: Evaluator,
        *,
        show_progress: bool = False,
        plugin_manager: PluginManager | None = None,
    ) -> List[TrialResult]:
        return self._strategies[self.strategy](
            model,
            X,
            y,
            evaluator,
            show_progress,
            plugin_manager,
        )

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------
    def _iterate_grid(self):
        keys = list(self.search_space)
        for values in itertools.product(*(self.search_space[k] for k in keys)):
            yield dict(zip(keys, values))

    def _grid_search(
        self,
        model,
        X,
        y,
        evaluator: Evaluator,
        show_progress: bool,
        plugin_manager: PluginManager | None,
    ) -> List[TrialResult]:
        total = math.prod(len(v) for v in self.search_space.values()) if self.search_space else 0
        progress: Progress | None = None
        task_id: int | None = None
        if show_progress:
            progress = Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                console=Console(stderr=True),
            )
            progress.start()
            task_id = progress.add_task(f"{self.name} Search", total=total)
        results: List[TrialResult] = []
        for i, params in enumerate(self._iterate_grid(), 1):
            trial_model = model.__class__(**{**model.get_params(), **params})
            start = perf_counter()
            trial_model.fit(X, y)
            score = evaluator.evaluate(trial_model, X, y)
            duration = perf_counter() - start
            if progress and task_id is not None:
                progress.update(task_id, advance=1)
            logger.log(
                f"{self.name.capitalize()} trial {i}: params={params} score={score:.4f} duration={duration:.2f}s",
                to=["console"],
            )
            if plugin_manager:
                plugin_manager.trigger("on_epoch_end", metrics={"score": score})
            results.append(
                TrialResult(trial_id=i, params=params, metrics={"score": score}, duration=duration)
            )
        if progress:
            progress.stop()
        return results

    def _random_search(
        self,
        model,
        X,
        y,
        evaluator: Evaluator,
        show_progress: bool,
        plugin_manager: PluginManager | None,
    ) -> List[TrialResult]:
        keys = list(self.search_space)
        progress: Progress | None = None
        task_id: int | None = None
        if show_progress:
            progress = Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                console=Console(stderr=True),
            )
            progress.start()
            task_id = progress.add_task(f"{self.name} Search", total=self.n_trials)
        results: List[TrialResult] = []
        for i in range(1, self.n_trials + 1):
            params = {k: random.choice(self.search_space[k]) for k in keys}
            trial_model = model.__class__(**{**model.get_params(), **params})
            start = perf_counter()
            trial_model.fit(X, y)
            score = evaluator.evaluate(trial_model, X, y)
            duration = perf_counter() - start
            if progress and task_id is not None:
                progress.update(task_id, advance=1)
            logger.log(
                f"{self.name.capitalize()} trial {i}: params={params} score={score:.4f} duration={duration:.2f}s",
                to=["console"],
            )
            if plugin_manager:
                plugin_manager.trigger("on_epoch_end", metrics={"score": score})
            results.append(
                TrialResult(trial_id=i, params=params, metrics={"score": score}, duration=duration)
            )
        if progress:
            progress.stop()
        return results

    def _optuna_search(
        self,
        model,
        X,
        y,
        evaluator: Evaluator,
        show_progress: bool,
        plugin_manager: PluginManager | None,
    ) -> List[TrialResult]:
        optuna = optional_import("optuna")
        results: List[TrialResult] = []
        progress: Progress | None = None
        task_id: int | None = None
        if show_progress:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                console=Console(stderr=True),
            )
            progress.start()
            task_id = progress.add_task(f"{self.name} Search", total=self.n_trials)

        def objective(trial):
            params = {}
            for name, values in self.search_space.items():
                params[name] = trial.suggest_categorical(name, list(values))
            trial_model = model.__class__(**{**model.get_params(), **params})
            start = perf_counter()
            trial_model.fit(X, y)
            score = evaluator.evaluate(trial_model, X, y)
            duration = perf_counter() - start
            logger.log(
                f"{self.name.capitalize()} trial {trial.number}: params={params} score={score:.4f} duration={duration:.2f}s",
                to=["console"],
            )
            if progress and task_id is not None:
                progress.update(task_id, advance=1)
            if plugin_manager:
                plugin_manager.trigger("on_epoch_end", metrics={"score": score})
            results.append(
                TrialResult(
                    trial_id=trial.number,
                    params=params,
                    metrics={"score": score},
                    duration=duration,
                )
            )
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)
        if progress:
            progress.stop()
        return results
