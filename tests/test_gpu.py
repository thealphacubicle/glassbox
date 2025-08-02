import builtins
import pytest

from glassbox.utils.gpu import is_gpu_available, supports_gpu


def test_is_gpu_available_false(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("cupy", "xgboost", "lightgbm"):
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert is_gpu_available() is False


def test_supports_gpu_name_matching():
    class XgbModel:
        pass

    class LinearModel:
        pass

    assert supports_gpu(XgbModel()) is True
    assert supports_gpu(LinearModel()) is False
