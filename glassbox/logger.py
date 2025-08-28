"""Unified logging for the Glassbox package."""
from __future__ import annotations

from typing import Iterable, List


class GlassboxLogger:
    """Simple logger that can write to multiple backends."""

    def __init__(
        self,
        use_wandb: bool = False,
        verbose: bool = True,
    ) -> None:
        self.use_wandb = use_wandb
        self.verbose = verbose

    def set_verbose(self, verbose: bool) -> None:
        """Globally enable or disable non-error logging."""
        self.verbose = verbose

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
            Iterable of destinations. Supported values: ``"console"`` or
            ``"wandb"``.
        """

        # Skip non-error logs when verbosity is disabled
        if not self.verbose and level.lower() != "error":
            return

        destinations: List[str] = list(to) if to is not None else ["console"]

        if "console" in destinations:
            print(message)

        if "wandb" in destinations and self.use_wandb:
            try:
                import wandb  # type: ignore

                wandb.log({"message": message, "level": level})
            except Exception:
                pass


logger = GlassboxLogger(use_wandb=True)
