"""Search strategies for hyperparameter tuning."""
from __future__ import annotations

import itertools
import random
import time
import math
from typing import Any, Callable, Dict, Iterable, List

from tqdm.auto import tqdm

from ..schemas import Evaluator, TrialResult
from ..utils.lazy_imports import optional_import
from ..logger import logger


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
            raise ValueError("search_space must be provided")
        self.strategy = strategy
        self.search_space = {k: list(v) for k, v in search_space.items()}
        self.n_trials = n_trials
        self.name = name or strategy
        self._strategies: Dict[str, Callable[[Any, Any, Any, Evaluator, bool], List[TrialResult]]] = {
            "grid": self._grid_search,
            "random": self._random_search,
            "optuna": self._optuna_search,
        }
        if strategy not in self._strategies:
            raise ValueError(f"Unknown search strategy: {strategy}")

    def run(self, model, X, y, evaluator: Evaluator, show_progress: bool = False) -> List[TrialResult]:
        return self._strategies[self.strategy](model, X, y, evaluator, show_progress)

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------
    def _iterate_grid(self):
        keys = list(self.search_space)
        for values in itertools.product(*(self.search_space[k] for k in keys)):
            yield dict(zip(keys, values))

    def _grid_search(self, model, X, y, evaluator: Evaluator, show_progress: bool) -> List[TrialResult]:
        total = math.prod(len(v) for v in self.search_space.values()) if self.search_space else 0
        iterator = enumerate(self._iterate_grid(), 1)
        if show_progress:
            iterator = tqdm(iterator, total=total, desc=f"{self.name} Search")
        results: List[TrialResult] = []
        for i, params in iterator:
            trial_model = model.__class__(**{**model.get_params(), **params})
            start = time.time()
            trial_model.fit(X, y)
            score = evaluator.evaluate(trial_model, X, y)
            duration = time.time() - start
            dest = ["console"] if show_progress else ["dashboard"]
            logger.log(
                f"{self.name.capitalize()} trial {i}: params={params} score={score:.4f} duration={duration:.2f}s",
                to=dest,
            )
            results.append(
                TrialResult(trial_id=i, params=params, metrics={"score": score}, duration=duration)
            )
        return results

    def _random_search(self, model, X, y, evaluator: Evaluator, show_progress: bool) -> List[TrialResult]:
        keys = list(self.search_space)
        iterator = range(1, self.n_trials + 1)
        if show_progress:
            iterator = tqdm(iterator, total=self.n_trials, desc=f"{self.name} Search")
        results: List[TrialResult] = []
        for i in iterator:
            params = {k: random.choice(self.search_space[k]) for k in keys}
            trial_model = model.__class__(**{**model.get_params(), **params})
            start = time.time()
            trial_model.fit(X, y)
            score = evaluator.evaluate(trial_model, X, y)
            duration = time.time() - start
            dest = ["console"] if show_progress else ["dashboard"]
            logger.log(
                f"{self.name.capitalize()} trial {i}: params={params} score={score:.4f} duration={duration:.2f}s",
                to=dest,
            )
            results.append(
                TrialResult(trial_id=i, params=params, metrics={"score": score}, duration=duration)
            )
        return results

    def _optuna_search(self, model, X, y, evaluator: Evaluator, show_progress: bool) -> List[TrialResult]:
        optuna = optional_import("optuna")
        results: List[TrialResult] = []
        pbar = tqdm(total=self.n_trials, desc=f"{self.name} Search") if show_progress else None

        def objective(trial):
            params = {}
            for name, values in self.search_space.items():
                params[name] = trial.suggest_categorical(name, list(values))
            trial_model = model.__class__(**{**model.get_params(), **params})
            start = time.time()
            trial_model.fit(X, y)
            score = evaluator.evaluate(trial_model, X, y)
            duration = time.time() - start
            dest = ["console"] if show_progress else ["dashboard"]
            logger.log(
                f"{self.name.capitalize()} trial {trial.number}: params={params} score={score:.4f} duration={duration:.2f}s",
                to=dest,
            )
            if pbar:
                pbar.update(1)
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
        if pbar:
            pbar.close()
        return results
