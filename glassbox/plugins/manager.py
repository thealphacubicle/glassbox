"""Plugin manager for coordinating plugin hooks."""


class PluginManager:
    """Registers and dispatches events to plugins."""

    def __init__(self) -> None:
        self.plugins = []

    def register(self, plugin: "Plugin") -> None:
        """Register a plugin instance."""
        self.plugins.append(plugin)

    def trigger(self, hook_name: str, **kwargs) -> None:
        """Trigger a hook on all registered plugins."""
        for plugin in self.plugins:
            method = getattr(plugin, hook_name, None)
            if callable(method):
                method(**kwargs)
