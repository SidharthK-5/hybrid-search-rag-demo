"""Env parsing must behave the same whether a value came from python-dotenv,
`docker run --env-file` (which keeps quotes and inline comments), or a dashboard."""
import pytest

from app.config import env_int, env_str


@pytest.mark.parametrize("raw, expected", [
    ("plain", "plain"),
    ("  padded  ", "padded"),
    ('"quoted"', "quoted"),
    ("'single'", "single"),
    ('" spaced inside "', "spaced inside"),
    ("10/minute          # per client IP", "10/minute"),
    ('"key"  # trailing note', "key"),
    ("has#hash-no-space", "has#hash-no-space"),   # '#' only starts a comment after whitespace
    ('"unbalanced', '"unbalanced'),
])
def test_env_str_normalizes_values(monkeypatch, raw, expected):
    monkeypatch.setenv("X_VAR", raw)
    assert env_str("X_VAR") == expected


@pytest.mark.parametrize("raw", ["", "   ", '""', "# only a comment"])
def test_blank_values_use_the_default(monkeypatch, raw):
    monkeypatch.setenv("X_VAR", raw)
    assert env_str("X_VAR", "fallback") == "fallback"


def test_unset_uses_the_default(monkeypatch):
    monkeypatch.delenv("X_VAR", raising=False)
    assert env_str("X_VAR", "d") == "d"


def test_env_int(monkeypatch):
    monkeypatch.setenv("N", "2   # hops")
    assert env_int("N", 0) == 2
    monkeypatch.setenv("N", "two")
    assert env_int("N", 7) == 7


def test_decorated_values_still_work_for_rate_limits_and_keys(monkeypatch):
    """The exact shapes found in a user's .env when passed through `docker run --env-file`."""
    from app.generate import ExtractiveFallback, GeminiProvider, get_provider
    from app.safety import ip_limit
    monkeypatch.setenv("RATE_LIMIT", "10/minute          # per client IP")
    assert ip_limit() == "10/minute"
    monkeypatch.setenv("GOOGLE_API_KEY", '"abc123"')
    monkeypatch.setattr("google.genai.Client", lambda **kw: kw)
    assert get_provider()._client["api_key"] == "abc123"
