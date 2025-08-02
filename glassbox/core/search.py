"""Search strategies for hyperparameter tuning."""
from __future__ import annotations

import itertools
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from .evaluator import evaluate
from ..utils.lazy_imports import optional_import


@dataclass
class TrialResult:
    trial_id: int
    params: Dict[str, Any]
    metrics: Dict[str, float]
    duration: float


def _iterate_grid(search_space: Dict[str, Iterable[Any]]):
    keys = list(search_space)
    for values in itertools.product(*(search_space[k] for k in keys)):
        yield dict(zip(keys, values))


def grid_search(model, X, y, search_space: Dict[str, Iterable[Any]]) -> List[TrialResult]:
    results: List[TrialResult] = []
    for i, params in enumerate(_iterate_grid(search_space), 1):
        trial_model = model.__class__(**{**model.get_params(), **params})
        start = time.time()
        trial_model.fit(X, y)
        score = evaluate(trial_model, X, y)
        duration = time.time() - start
        results.append(
            TrialResult(i, params, {"score": score}, duration)
        )
    return results


def random_search(
    model,
    X,
    y,
    search_space: Dict[str, Iterable[Any]],
    n_trials: int = 10,
) -> List[TrialResult]:
    results: List[TrialResult] = []
    keys = list(search_space)
    for i in range(1, n_trials + 1):
        params = {k: random.choice(list(search_space[k])) for k in keys}
        trial_model = model.__class__(**{**model.get_params(), **params})
        start = time.time()
        trial_model.fit(X, y)
        score = evaluate(trial_model, X, y)
        duration = time.time() - start
        results.append(TrialResult(i, params, {"score": score}, duration))
    return results


def optuna_search(
    model,
    X,
    y,
    search_space: Dict[str, Iterable[Any]],
    n_trials: int = 10,
) -> List[TrialResult]:
    optuna = optional_import("optuna")

    results: List[TrialResult] = []

    def objective(trial):
        params = {}
        for name, values in search_space.items():
            params[name] = trial.suggest_categorical(name, list(values))
        trial_model = model.__class__(**{**model.get_params(), **params})
        start = time.time()
        trial_model.fit(X, y)
        score = evaluate(trial_model, X, y)
        duration = time.time() - start
        results.append(
            TrialResult(trial.number, params, {"score": score}, duration)
        )
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return results
