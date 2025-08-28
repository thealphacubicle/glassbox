import pytest
import pytest
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from glassbox import ModelSearch
from glassbox.core.search import Search
from glassbox.core.evaluator import SklearnEvaluator
from glassbox.core import model_search as ms_module

SEARCH_SPACE = {"C": [0.1, 1.0]}


def test_model_search_runs():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    ms = ModelSearch(
        LogisticRegression(max_iter=50),
        Search("grid", SEARCH_SPACE),
        SklearnEvaluator(),
    )
    model = ms.search(X_train, y_train)
    assert model.score(X_test, y_test) >= 0


def test_model_search_gpu_guard(monkeypatch):
    monkeypatch.setattr(ms_module, "is_gpu_available", lambda: False)
    with pytest.raises(RuntimeError):
        ModelSearch(
            LogisticRegression(),
            Search("grid", SEARCH_SPACE),
            SklearnEvaluator(),
            enable_gpu=True,
        )


def test_model_search_writes_dashboard_state(tmp_path, monkeypatch):
    X, y = load_iris(return_X_y=True)
    X_train, _, y_train, _ = train_test_split(X, y, random_state=0)
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(ms_module.logger, "state_path", str(state_file))
    monkeypatch.setattr(ms_module.logger, "use_dashboard", True)
    ms = ModelSearch(
        LogisticRegression(max_iter=50),
        Search("grid", SEARCH_SPACE),
        SklearnEvaluator(),
        dashboard=True,
    )
    ms.search(X_train, y_train)
    assert state_file.exists()
