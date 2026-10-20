"""Shared fixtures. Tests must never touch the network or real API keys, so every
test starts with provider keys removed and generous rate limits; tests that
exercise limiting set their own."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.ingest import build_chunks
from app.safety import limiter
from app.search import SearchIndex, build_search_index

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
                "TRUSTED_PROXY_HOPS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("RATE_LIMIT", "1000/minute")
    monkeypatch.setenv("GLOBAL_RATE_LIMIT", "10000/hour")
    limiter.reset()


@pytest.fixture(scope="session")
def chunks():
    return build_chunks(DOCS_DIR)


@pytest.fixture
def index(chunks) -> SearchIndex:
    """Real sample-corpus index using the keyless TF-IDF backend."""
    return build_search_index(chunks)


@pytest.fixture
def client():
    from app.main import app
    with TestClient(app) as c:  # `with` runs the lifespan (index build)
        yield c
