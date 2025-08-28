"""KnockNotifier plugin using the knockknock telegram sender."""

from __future__ import annotations

from threading import Thread

from glassbox.plugins.base import Plugin
from glassbox.logger import logger

try:  # pragma: no cover - optional dependency
    from knockknock import telegram_sender
except Exception:  # pragma: no cover - optional dependency
    telegram_sender = None


class KnockNotifier(Plugin):
    """Send a notification when training completes."""

    def __init__(self, telegram_token: str, chat_id: int) -> None:
        self.telegram_token = telegram_token
        self.chat_id = chat_id

    def on_training_end(self) -> None:
        logger.log("KnockNotifier: Training complete", to=["console"])
        self._notify("✅ Glassbox training finished!")

    def _notify(self, msg: str) -> None:
        """Dispatch the telegram notification in a background thread."""

        if telegram_sender is None:
            logger.log("KnockNotifier: knockknock not installed", to=["console"])
            return

        def send() -> None:
            try:
                send_fn = telegram_sender(token=self.telegram_token, chat_id=self.chat_id)
                decorated = send_fn(lambda: None)
                decorated()
            except Exception as exc:  # pragma: no cover - network/telegram errors
                logger.log(f"KnockNotifier error: {exc}", to=["console"])

        Thread(target=send, daemon=True).start()
