"""CORS is opt-in and narrow: it exists so the GitHub Pages loading page can poll /health."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.safety import configure_cors, cors_origins

PAGES = "https://someone.github.io"


def make_app() -> TestClient:
    app = FastAPI()
    configure_cors(app)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/search")
    def search():
        return {}

    return TestClient(app)


def test_no_cors_headers_by_default():
    r = make_app().get("/health", headers={"Origin": PAGES})
    assert "access-control-allow-origin" not in r.headers


def test_listed_origin_may_read_health(monkeypatch):
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", PAGES)
    r = make_app().get("/health", headers={"Origin": PAGES})
    assert r.headers["access-control-allow-origin"] == PAGES
    assert "access-control-allow-credentials" not in r.headers


def test_unlisted_origin_gets_no_access(monkeypatch):
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", PAGES)
    r = make_app().get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_cross_origin_post_is_not_permitted(monkeypatch):
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", PAGES)
    preflight = make_app().options("/search", headers={
        "Origin": PAGES, "Access-Control-Request-Method": "POST"})
    assert preflight.status_code == 400  # POST isn't in the allowed methods


@pytest.mark.parametrize("raw, expected", [
    ("", []),
    (f"{PAGES}/", [PAGES]),                                  # trailing slash tolerated
    (f"{PAGES}, http://localhost:8080 ", [PAGES, "http://localhost:8080"]),
    (f"{PAGES}   # my pages site", [PAGES]),                 # inline comment from a .env
])
def test_origin_list_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", raw)
    assert cors_origins() == expected
