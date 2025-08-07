from glassbox.plugins.base import Plugin
from glassbox.plugins.manager import PluginManager
from glassbox.plugins import knocknotifier


def test_plugin_manager_triggers_hooks():
    class Dummy(Plugin):
        def __init__(self):
            self.started = False
        def on_training_start(self):
            self.started = True
    pm = PluginManager()
    dummy = Dummy()
    pm.register(dummy)
    pm.trigger("on_training_start")
    assert dummy.started


def test_knocknotifier_handles_missing_dependency(monkeypatch, capsys):
    monkeypatch.setattr(knocknotifier, "telegram_sender", None)
    notifier = knocknotifier.KnockNotifier("token", 123)
    notifier.on_training_end()
    out = capsys.readouterr().out
    assert "Training complete" in out
    assert "not installed" in out
