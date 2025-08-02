"""Simple real-time dashboard using Reflex or Streamlit."""
from __future__ import annotations

import json
import os
import threading
from typing import Any, List

from ..utils.lazy_imports import optional_import


class DashboardServer:
    """Launch a minimal dashboard visualising trial results.

    The MVP implementation stores shared state in a JSON file on disk.  The
    dashboard, powered by Reflex or Streamlit, periodically refreshes and
    displays the leaderboard of trials.
    """

    def __init__(self, state_path: str = "dashboard_state.json") -> None:
        self.state_path = state_path

    def _run_streamlit(self) -> None:
        import streamlit as st
        import pandas as pd
        import time

        st.title("Glassbox Dashboard")
        placeholder = st.empty()
        while True:
            if os.path.exists(self.state_path):
                try:
                    with open(self.state_path) as f:
                        data = json.load(f)
                    df = pd.DataFrame(data)
                    placeholder.dataframe(df)
                except Exception:
                    pass
            time.sleep(1)

    def run(self) -> None:
        """Start the dashboard in a background thread."""
        try:
            optional_import("reflex")  # not used directly in MVP
            # A real Reflex app would be defined here.
            # For the MVP we fall back to Streamlit for a lightweight UI.
        except ImportError:
            pass
        try:
            optional_import("streamlit")
        except ImportError as exc:
            raise ImportError(
                "Install optional dependency via `pip install glassbox[ui]`"
            ) from exc
        thread = threading.Thread(target=self._run_streamlit, daemon=True)
        thread.start()
