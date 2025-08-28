import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from glassbox import ModelTuner
from glassbox.core import tuner as tuner_module


def test_model_tuner_search():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    mt = ModelTuner(LogisticRegression(max_iter=50), strategy="grid")
    model = mt.search(X_train, y_train)
    assert model.score(X_test, y_test) >= 0


def test_model_tuner_gpu_guard(monkeypatch):
    monkeypatch.setattr(tuner_module, "is_gpu_available", lambda: False)
    with pytest.raises(RuntimeError):
        ModelTuner(LogisticRegression(), enable_gpu=True)


def test_model_tuner_writes_dashboard_state(tmp_path, monkeypatch):
    X, y = load_iris(return_X_y=True)
    X_train, _, y_train, _ = train_test_split(X, y, random_state=0)
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(tuner_module.logger, "state_path", str(state_file))
    monkeypatch.setattr(tuner_module.logger, "use_dashboard", True)
    mt = ModelTuner(LogisticRegression(max_iter=50), strategy="grid", dashboard=True)
    mt.search(X_train, y_train)
    assert state_file.exists()
