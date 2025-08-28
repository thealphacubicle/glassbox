from glassbox.logger import GlassboxLogger


def test_console_logging(capsys):
    logger = GlassboxLogger(use_wandb=False)
    logger.log("hello", to=["console"])
    captured = capsys.readouterr()
    assert "hello" in captured.out


def test_logger_respects_verbose_flag(capsys):
    logger = GlassboxLogger(use_wandb=False, verbose=False)
    logger.log("hidden", to=["console"])
    logger.log("oops", level="error", to=["console"])
    captured = capsys.readouterr()
    assert "hidden" not in captured.out
    assert "oops" in captured.out
