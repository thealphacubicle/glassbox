"""Base class for Glassbox plugins."""


class Plugin:
    """Base plugin class.

    Subclasses can override any of the lifecycle hooks to perform actions
    during training. Hooks should be non-blocking and avoid mutating model
    state.
    """

    def on_training_start(self) -> None:  # pragma: no cover - simple pass methods
        """Called before training begins."""
        pass

    def on_training_end(self) -> None:  # pragma: no cover - simple pass methods
        """Called after training ends."""
        pass

    def on_epoch_end(self, metrics: dict) -> None:  # pragma: no cover - simple pass methods
        """Called after each epoch with training metrics."""
        pass
