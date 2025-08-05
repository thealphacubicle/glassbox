"""Search strategies for hyperparameter tuning."""
from __future__ import annotations

import itertools
import random
import time
from dataclasses import dataclass
import math
import logging
from typing import Any, Dict, Iterable, List, Optional

from tqdm.auto import tqdm

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


def grid_search(
    model,
    X,
    y,
    search_space: Dict[str, Iterable[Any]],
    *,
    show_progress: bool = False,
    logger: Optional[logging.Logger] = None,
) -> List[TrialResult]:
    space_lists = {k: list(v) for k, v in search_space.items()}
    total = math.prod(len(v) for v in space_lists.values()) if space_lists else 0
    iterator = enumerate(_iterate_grid(space_lists), 1)
    if show_progress:
        iterator = tqdm(iterator, total=total, desc="Grid Search")

    results: List[TrialResult] = []
    for i, params in iterator:
        trial_model = model.__class__(**{**model.get_params(), **params})
        start = time.time()
        trial_model.fit(X, y)
        score = evaluate(trial_model, X, y)
        duration = time.time() - start
        if logger:
            logger.info(
                "Grid trial %s: params=%s score=%.4f duration=%.2fs",
                i,
                params,
                score,
                duration,
            )
        results.append(TrialResult(i, params, {"score": score}, duration))
    return results


def random_search(
    model,
    X,
    y,
    search_space: Dict[str, Iterable[Any]],
    n_trials: int = 10,
    *,
    show_progress: bool = False,
    logger: Optional[logging.Logger] = None,
) -> List[TrialResult]:
    results: List[TrialResult] = []
    keys = list(search_space)
    iterator = range(1, n_trials + 1)
    if show_progress:
        iterator = tqdm(iterator, total=n_trials, desc="Random Search")

    for i in iterator:
        params = {k: random.choice(list(search_space[k])) for k in keys}
        trial_model = model.__class__(**{**model.get_params(), **params})
        start = time.time()
        trial_model.fit(X, y)
        score = evaluate(trial_model, X, y)
        duration = time.time() - start
        if logger:
            logger.info(
                "Random trial %s: params=%s score=%.4f duration=%.2fs",
                i,
                params,
                score,
                duration,
            )
        results.append(TrialResult(i, params, {"score": score}, duration))
    return results


def optuna_search(
    model,
    X,
    y,
    search_space: Dict[str, Iterable[Any]],
    n_trials: int = 10,
    *,
    show_progress: bool = False,
    logger: Optional[logging.Logger] = None,
) -> List[TrialResult]:
    optuna = optional_import("optuna")

    results: List[TrialResult] = []
    pbar = tqdm(total=n_trials, desc="Optuna Search") if show_progress else None

    def objective(trial):
        params = {}
        for name, values in search_space.items():
            params[name] = trial.suggest_categorical(name, list(values))
        trial_model = model.__class__(**{**model.get_params(), **params})
        start = time.time()
        trial_model.fit(X, y)
        score = evaluate(trial_model, X, y)
        duration = time.time() - start
        if logger:
            logger.info(
                "Optuna trial %s: params=%s score=%.4f duration=%.2fs",
                trial.number,
                params,
                score,
                duration,
            )
        if pbar:
            pbar.update(1)
        results.append(TrialResult(trial.number, params, {"score": score}, duration))
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    if pbar:
        pbar.close()
    return results
