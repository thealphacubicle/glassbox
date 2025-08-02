import pytest

from glassbox.ui.dashboard import DashboardServer


def test_dashboard_requires_streamlit(monkeypatch):
    monkeypatch.setattr(
        "glassbox.ui.dashboard.optional_import",
        lambda name: (_ for _ in ()).throw(ImportError()),
    )
    server = DashboardServer()
    with pytest.raises(ImportError):
        server.run()
