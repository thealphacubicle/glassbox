from glassbox.plugins.base import Plugin
from glassbox.plugins.manager import PluginManager
from glassbox.plugins import knocknotifier
from glassbox.plugins.resource_monitor import ResourceMonitor
from glassbox.core.search import Search
from glassbox.core.evaluator import SklearnEvaluator
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


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


def test_resource_monitor_logs_memory(capsys):
    X, y = load_iris(return_X_y=True)
    monitor = ResourceMonitor()
    pm = PluginManager()
    pm.register(monitor)
    s = Search("random", {"C": [0.1]}, n_trials=1)
    model = LogisticRegression(max_iter=10)
    evaluator = SklearnEvaluator()
    s.run(model, X, y, evaluator, show_progress=False, plugin_manager=pm)
    out = capsys.readouterr().out
    assert "memory" in out.lower()
