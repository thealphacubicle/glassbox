from glassbox.tracking.wandb_tracker import WandbTracker


class DummyRun:
    def __init__(self):
        self.logged = []
        self.finished = False

    def log(self, data):
        self.logged.append(data)

    def finish(self):
        self.finished = True


class DummyWandb:
    def __init__(self):
        self.run = DummyRun()
        self.init_called = False

    def init(self, project, config):
        self.init_called = True
        self.project = project
        self.config = config
        return self.run

    def log(self, data):
        self.run.log(data)


def test_wandb_tracker_integration(monkeypatch):
    dummy = DummyWandb()
    monkeypatch.setattr(
        "glassbox.tracking.wandb_tracker.optional_import", lambda name: dummy
    )
    tracker = WandbTracker()
    tracker.start({"a": 1})
    tracker.log(1, {"metric": 0.5})
    tracker.finish()
    assert dummy.init_called is True
    assert dummy.run.logged[0]["trial_id"] == 1
    assert dummy.run.finished is True
