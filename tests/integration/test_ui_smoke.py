"""Smoke tests: static UI assets are served correctly."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app import create_app


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    db_path = tmp_path / "ui_smoke.db"
    monkeypatch.setenv("ADAPTIVE_DUELIST_DB", str(db_path))
    app = create_app()
    with TestClient(app) as client:
        yield client


class TestUISmoke:

    def test_index_html_served(self, api_client):
        res = api_client.get("/ui/index.html")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]

    def test_app_js_served(self, api_client):
        res = api_client.get("/ui/app.js")
        assert res.status_code == 200
        assert "javascript" in res.headers["content-type"]

    def test_styles_css_served(self, api_client):
        res = api_client.get("/ui/styles.css")
        assert res.status_code == 200
        assert "css" in res.headers["content-type"]

    def test_ui_root_redirects_or_serves(self, api_client):
        res = api_client.get("/ui/", follow_redirects=True)
        assert res.status_code in (200, 301, 307)
