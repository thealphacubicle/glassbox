import json
from glassbox.logger import GlassboxLogger


def test_console_logging(capsys):
    logger = GlassboxLogger(use_dashboard=False, use_wandb=False)
    logger.log("hello", to=["console"])
    captured = capsys.readouterr()
    assert "hello" in captured.out


def test_dashboard_logging(tmp_path):
    logger = GlassboxLogger(use_dashboard=False, use_wandb=False)
    state_file = tmp_path / "state.json"
    logger.state_path = str(state_file)
    logger.use_dashboard = True
    logger.log("dash", to=["dashboard"])
    assert state_file.exists()
    with open(state_file) as f:
        data = json.load(f)
    assert data[0]["message"] == "dash"
