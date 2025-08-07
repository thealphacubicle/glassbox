"""Unified logging for the Glassbox package."""
from __future__ import annotations

import json
import os
from typing import Iterable, List


class GlassboxLogger:
    """Simple logger that can write to multiple backends."""

    def __init__(
        self,
        use_dashboard: bool = False,
        use_wandb: bool = False,
        state_path: str = "dashboard_state.json",
    ) -> None:
        self.use_dashboard = use_dashboard
        self.use_wandb = use_wandb
        self.state_path = state_path
        self._dashboard_server = None

        if self.use_dashboard:
            try:  # pragma: no cover - optional dependency
                from .ui.dashboard import DashboardServer

                self._dashboard_server = DashboardServer(state_path=self.state_path)
                self._dashboard_server.run()
            except Exception:  # missing optional deps or runtime error
                self.use_dashboard = False

    def log(self, message: str, level: str = "info", to: Iterable[str] | None = None) -> None:
        """Log a message to the selected destinations.

        Parameters
        ----------
        message:
            Text to log.
        level:
            Log level as a string. Currently unused but reserved for future
            enhancements.
        to:
            Iterable of destinations. Supported values: ``"console"``,
            ``"wandb"``, ``"dashboard"``.
        """

        destinations: List[str] = list(to) if to is not None else ["console"]

        if "console" in destinations:
            print(message)

        if "wandb" in destinations and self.use_wandb:
            try:
                import wandb  # type: ignore

                wandb.log({"message": message, "level": level})
            except Exception:
                pass

        if "dashboard" in destinations and self.use_dashboard:
            self.log_to_dashboard(message, level=level)

    def log_to_dashboard(self, message: str, level: str = "info") -> None:
        """Persist log messages for the dashboard server.

        Messages are appended to a JSON file that the :class:`DashboardServer`
        reads and visualises in real time. Any errors are silently ignored so
        that logging never interrupts training.
        """

        entry = {"message": message, "level": level}
        try:
            data: List[dict] = []
            if os.path.exists(self.state_path):
                with open(self.state_path) as f:
                    existing = json.load(f)
                    if isinstance(existing, list):
                        data = existing
            data.append(entry)
            with open(self.state_path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass


logger = GlassboxLogger(use_dashboard=True, use_wandb=True)
