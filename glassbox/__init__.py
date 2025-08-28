"""Glassbox package initialization."""

from importlib.metadata import PackageNotFoundError, version

from glassbox.core.model_search import ModelSearch

try:  # pragma: no cover - fallback when package metadata is missing
    __version__ = version("glassbox")
except PackageNotFoundError:  # pragma: no cover - fallback for local usage
    __version__ = "0.0.0"

__all__ = ["ModelSearch", "__version__"]
