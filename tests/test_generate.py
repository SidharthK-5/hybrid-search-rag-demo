"""Provider selection (mocked env vars + mocked SDKs) and answer generation."""
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app import generate
from app.generate import (
    AnthropicProvider,
    ExtractiveFallback,
    GeminiProvider,
    OpenAIProvider,
    get_provider,
)

CONTEXT = [{"source": "Leave Of Absence Policy", "text": "Up to 12 weeks of medical leave."}]


@pytest.fixture(autouse=True)
def fake_sdks(monkeypatch):
    """Provider constructors import their SDKs lazily; stub them so no SDK is
    needed (anthropic/openai are optional installs) and nothing hits the network."""
    for name, attr in (("anthropic", "Anthropic"), ("openai", "OpenAI")):
        module = ModuleType(name)
        setattr(module, attr, MagicMock(name=attr))
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr("google.genai.Client", MagicMock(name="genai.Client"))


def test_no_keys_selects_extractive():
    assert isinstance(get_provider(), ExtractiveFallback)


@pytest.mark.parametrize("var, expected", [
    ("ANTHROPIC_API_KEY", AnthropicProvider),
    ("OPENAI_API_KEY", OpenAIProvider),
    ("GOOGLE_API_KEY", GeminiProvider),
])
def test_each_key_selects_its_provider(monkeypatch, var, expected):
    monkeypatch.setenv(var, "test-key")
    assert isinstance(get_provider(), expected)


def test_precedence_when_several_keys_are_set(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "g")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    assert isinstance(get_provider(), OpenAIProvider)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert isinstance(get_provider(), AnthropicProvider)


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_keys_are_treated_as_unset(monkeypatch, blank):
    # A hosting dashboard may leave an unfilled secret as an empty string.
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(var, blank)
    assert isinstance(get_provider(), ExtractiveFallback)


def test_provider_names_distinguish_generated_from_extractive(monkeypatch):
    assert get_provider().name == "extractive"
    monkeypatch.setenv("GOOGLE_API_KEY", "g")
    assert get_provider().name == "generated"


# --- extractive fallback output ---

def test_extractive_with_no_context():
    assert "No matching passages" in ExtractiveFallback().generate("q", [])


def test_extractive_cites_sources_and_truncates_long_passages():
    long_hit = {"source": "Doc", "text": "word " * 200}
    answer = ExtractiveFallback().generate("q", [long_hit] * 5)
    assert answer.count("[Doc]") == 3          # top three only
    assert "..." in answer
    assert max(len(line) for line in answer.split("\n\n")[1:]) < 300


# --- Gemini provider ---

def test_gemini_uses_flash_latest_and_returns_text(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "g")
    provider = GeminiProvider()
    provider._client.models.generate_content.return_value = SimpleNamespace(text="12 weeks [Leave Of Absence Policy].")
    assert provider.generate("How long is medical leave?", CONTEXT) == "12 weeks [Leave Of Absence Policy]."
    kwargs = provider._client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == generate.DEFAULT_GEMINI_MODEL == "gemini-flash-lite-latest"
    assert "Leave Of Absence Policy" in kwargs["contents"]
    assert kwargs["config"].system_instruction == generate.SYSTEM_PROMPT


def test_gemini_empty_response_raises_so_the_api_can_fall_back(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "g")
    provider = GeminiProvider()
    provider._client.models.generate_content.return_value = SimpleNamespace(text=None)
    with pytest.raises(RuntimeError):
        provider.generate("q", CONTEXT)


def test_every_provider_is_built_with_a_hard_timeout(monkeypatch):
    # Regression: a Gemini 503 outage once hung requests for over a minute because
    # the SDK default had no practical timeout.
    monkeypatch.setenv("GOOGLE_API_KEY", "g")
    GeminiProvider()
    http_options = __import__("google.genai", fromlist=["Client"]).Client.call_args.kwargs["http_options"]
    assert http_options.timeout == generate.DEFAULT_LLM_TIMEOUT_SECONDS * 1000
    assert http_options.retry_options.attempts == 1

    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    AnthropicProvider()
    assert sys.modules["anthropic"].Anthropic.call_args.kwargs["timeout"] == generate.DEFAULT_LLM_TIMEOUT_SECONDS

    monkeypatch.setenv("OPENAI_API_KEY", "o")
    OpenAIProvider()
    assert sys.modules["openai"].OpenAI.call_args.kwargs["timeout"] == generate.DEFAULT_LLM_TIMEOUT_SECONDS


@pytest.mark.parametrize("override, expected", [
    ("gemini-3.5-flash-lite", "gemini-3.5-flash-lite"),
    ("   ", "gemini-flash-lite-latest"),   # blank -> default
])
def test_gemini_model_can_be_overridden_by_env(monkeypatch, override, expected):
    monkeypatch.setenv("GOOGLE_API_KEY", "g")
    monkeypatch.setenv("GEMINI_GENERATION_MODEL", override)
    provider = GeminiProvider()
    provider._client.models.generate_content.return_value = SimpleNamespace(text="ok")
    provider.generate("q", CONTEXT)
    assert provider._client.models.generate_content.call_args.kwargs["model"] == expected


def test_llm_timeout_is_configurable(monkeypatch):
    assert generate.llm_timeout() == generate.DEFAULT_LLM_TIMEOUT_SECONDS
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "40   # slow provider")
    assert generate.llm_timeout() == 40
    monkeypatch.setenv("GOOGLE_API_KEY", "g")
    GeminiProvider()
    import google.genai
    assert google.genai.Client.call_args.kwargs["http_options"].timeout == 40_000
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "0")
    assert generate.llm_timeout() == 1   # never zero/negative


def test_extractive_notice_can_be_customised():
    answer = ExtractiveFallback(ExtractiveFallback.UNAVAILABLE).generate("q", CONTEXT)
    assert answer.startswith("The language model is temporarily unavailable")
