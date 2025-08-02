"""GPU utilities."""
from __future__ import annotations

from typing import Any


def is_gpu_available() -> bool:
    """Best-effort check for GPU availability."""
    try:  # CuPy
        import cupy  # noqa: F401

        return True
    except ImportError:
        pass
    try:  # xgboost GPU build
        import xgboost

        info = getattr(xgboost, "build_info", lambda: {})()
        if isinstance(info, dict) and info.get("USE_CUDA", "") == "ON":
            return True
    except (ImportError, AttributeError):
        pass
    try:  # lightgbm GPU
        import lightgbm  # noqa: F401

        return True
    except ImportError:
        pass
    return False


def supports_gpu(model: Any) -> bool:
    """Return ``True`` if *model* appears to support GPU execution."""
    name = model.__class__.__name__.lower()
    return any(key in name for key in ["xgb", "lgb", "cuda", "gpu"])
