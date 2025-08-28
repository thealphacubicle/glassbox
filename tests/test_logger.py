from glassbox.logger import GlassboxLogger


def test_console_logging(capsys):
    logger = GlassboxLogger(use_wandb=False)
    logger.log("hello", to=["console"])
    captured = capsys.readouterr()
    assert "hello" in captured.out
