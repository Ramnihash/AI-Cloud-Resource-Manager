"""Regression test: every documented route is registered on the app and answers."""
from __future__ import annotations

import pytest

from backend.app import create_app

API_ROUTES = {
    ("GET", "/health"),
    ("POST", "/api/v1/cpu/action"),
    ("POST", "/api/v1/memory/page-replacement"),
    ("POST", "/api/v1/memory/monitor"),
    ("POST", "/api/v1/memory/analysis"),
    ("POST", "/api/v1/device/predictive-action"),
    ("POST", "/api/v1/device/anomaly-score"),
    ("GET", "/api/v1/monitor/overview"),
    ("POST", "/api/v1/system/step"),
    ("POST", "/api/v1/system/reset"),
    ("GET", "/api/v1/system/history"),
}

DASHBOARD_PAGES = ["/dashboard/", "/dashboard/cpu", "/dashboard/memory", "/dashboard/device", "/dashboard/activity"]


@pytest.fixture(scope="module")
def app():
    return create_app(test_config={"TESTING": True})


def registered(app) -> set[tuple[str, str]]:
    return {
        (method, rule.rule)
        for rule in app.url_map.iter_rules()
        for method in rule.methods
        if method not in ("HEAD", "OPTIONS")
    }


def test_all_api_routes_are_registered(app):
    missing = API_ROUTES - registered(app)
    assert not missing, f"routes missing from the app: {sorted(missing)}"


@pytest.mark.parametrize("path", DASHBOARD_PAGES)
def test_dashboard_pages_render(app, path):
    response = app.test_client().get(path)
    assert response.status_code == 200
    assert b"reset-system-btn" in response.data


def test_health_ok(app):
    assert app.test_client().get("/health").get_json() == {"status": "ok"}


def test_unknown_route_returns_404(app):
    assert app.test_client().get("/api/v1/nope").status_code == 404
