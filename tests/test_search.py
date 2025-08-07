import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

from glassbox.core import search

X, y = load_iris(return_X_y=True)
MODEL = LogisticRegression(max_iter=50)
SEARCH_SPACE = {"C": [0.1, 1.0]}


def test_grid_search_runs():
    results = search.grid_search(MODEL, X, y, SEARCH_SPACE)
    assert len(results) == 2
    assert all("score" in r.metrics for r in results)


def test_random_search_runs():
    results = search.random_search(MODEL, X, y, SEARCH_SPACE, n_trials=2)
    assert len(results) == 2


def test_optuna_search_requires_optuna(monkeypatch):
    monkeypatch.setattr(search, "optional_import", lambda name: (_ for _ in ()).throw(ImportError()))
    with pytest.raises(ImportError):
        search.optuna_search(MODEL, X, y, SEARCH_SPACE, n_trials=1)


def test_grid_search_logs_to_dashboard(monkeypatch):
    calls = []
    monkeypatch.setattr(search.logger, "log", lambda msg, level="info", to=None: calls.append(to))
    search.grid_search(MODEL, X, y, SEARCH_SPACE)
    assert calls and calls[0] == ["dashboard"]
