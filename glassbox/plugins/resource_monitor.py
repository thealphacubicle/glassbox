"""Plugin that reports CPU time and memory usage."""
from __future__ import annotations

from time import perf_counter
import resource

from glassbox.plugins.base import Plugin
from glassbox.logger import logger


class ResourceMonitor(Plugin):
    """Log resource usage at key training events."""

    def __init__(self) -> None:
        self._start: float | None = None

    def _memory_mb(self) -> float:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024  # in MB on Linux

    def on_training_start(self) -> None:  # pragma: no cover - simple timestamp
        self._start = perf_counter()
        logger.log(f"Training started | memory={self._memory_mb():.1f}MB")

    def on_epoch_end(self, metrics: dict) -> None:
        logger.log(
            f"Epoch end | memory={self._memory_mb():.1f}MB | metrics={metrics}"
        )

    def on_training_end(self) -> None:  # pragma: no cover - simple timestamp
        duration = perf_counter() - self._start if self._start else 0.0
        logger.log(
            f"Training finished in {duration:.2f}s | memory={self._memory_mb():.1f}MB"
        )
