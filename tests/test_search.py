import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

from glassbox.core.search import Search
from glassbox.core.evaluator import SklearnEvaluator
from glassbox.core import search as search_module

X, y = load_iris(return_X_y=True)
MODEL = LogisticRegression(max_iter=50)
SEARCH_SPACE = {"C": [0.1, 1.0]}
EVALUATOR = SklearnEvaluator()


def test_grid_search_runs():
    s = Search("grid", SEARCH_SPACE)
    results = s.run(MODEL, X, y, EVALUATOR)
    assert len(results) == 2
    assert all("score" in r.metrics for r in results)


def test_random_search_runs():
    s = Search("random", SEARCH_SPACE, n_trials=2)
    results = s.run(MODEL, X, y, EVALUATOR)
    assert len(results) == 2


def test_optuna_search_requires_optuna(monkeypatch):
    monkeypatch.setattr(
        "glassbox.core.search.optional_import", lambda name: (_ for _ in ()).throw(ImportError())
    )
    s = Search("optuna", SEARCH_SPACE, n_trials=1)
    with pytest.raises(ImportError):
        s.run(MODEL, X, y, EVALUATOR)


def test_grid_search_logs_to_console(monkeypatch):
    calls = []
    monkeypatch.setattr(
        search_module.logger,
        "log",
        lambda msg, level="info", to=None: calls.append(to),
    )
    s = Search("grid", SEARCH_SPACE)
    s.run(MODEL, X, y, EVALUATOR)
    assert calls and calls[0] == ["console"]
