"""Optional answer generation on top of the retrieved chunks.

The app runs with zero API keys out of the box: if none of ANTHROPIC_API_KEY,
OPENAI_API_KEY or GOOGLE_API_KEY is set, it falls back to an extractive "answer"
built directly from the top retrieved passages. Set any one of them to see full
generative RAG instead. (GOOGLE_API_KEY also switches vector search to dense
Gemini embeddings — see app/vector.py.)
"""
from abc import ABC, abstractmethod

from app.config import env_int, env_str

# Hard cap on any one LLM call, overridable with LLM_TIMEOUT_SECONDS. The vendor SDKs'
# defaults (up to 10 minutes) would leave a visitor staring at a spinner during a
# provider outage; on timeout the API layer falls back to the extractive answer
# instead (see app/main.py). Raise it if your provider is slow but succeeding.
DEFAULT_LLM_TIMEOUT_SECONDS = 15


def llm_timeout() -> int:
    return max(1, env_int("LLM_TIMEOUT_SECONDS", DEFAULT_LLM_TIMEOUT_SECONDS))


# Flash-Lite rather than Flash: the free tier gives the lighter models noticeably more
# throughput, and this task (answer from a few short excerpts) doesn't need more.
# "-latest" is a Google-maintained alias; override with GEMINI_GENERATION_MODEL to pin
# a specific model (e.g. gemini-3.5-flash-lite) or move up to a larger one.
DEFAULT_GEMINI_MODEL = "gemini-flash-lite-latest"

SYSTEM_PROMPT = (
    "You answer questions using only the provided policy excerpts. "
    "Cite the source document name for any claim. If the excerpts don't "
    "contain the answer, say so plainly rather than guessing."
)


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def generate(self, query: str, context: list[dict]) -> str:
        ...


class ExtractiveFallback(LLMProvider):
    """No API key configured (or the LLM just failed): returns the top passages
    directly instead of a generated answer, so the app is still useful."""

    name = "extractive"
    NOT_CONFIGURED = "No LLM configured — showing the top matching passages instead:"
    UNAVAILABLE = "The language model is temporarily unavailable — showing the top matching passages instead:"

    def __init__(self, notice: str = NOT_CONFIGURED):
        self._notice = notice

    def generate(self, query: str, context: list[dict]) -> str:
        if not context:
            return "No matching passages were found for that query."
        lines = [f"{self._notice}\n"]
        for hit in context[:3]:
            snippet = hit["text"].strip().replace("\n", " ")
            if len(snippet) > 280:
                snippet = snippet[:280].rsplit(" ", 1)[0] + "..."
            lines.append(f"[{hit['source']}] {snippet}")
        return "\n\n".join(lines)


class AnthropicProvider(LLMProvider):
    name = "generated"

    def __init__(self):
        import anthropic
        self._client = anthropic.Anthropic(
            api_key=env_str("ANTHROPIC_API_KEY"), timeout=llm_timeout(), max_retries=1
        )

    def generate(self, query: str, context: list[dict]) -> str:
        context_block = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in context)
        message = self._client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Policy excerpts:\n\n{context_block}\n\nQuestion: {query}",
                }
            ],
        )
        return message.content[0].text


class OpenAIProvider(LLMProvider):
    name = "generated"

    def __init__(self):
        from openai import OpenAI
        self._client = OpenAI(
            api_key=env_str("OPENAI_API_KEY"), timeout=llm_timeout(), max_retries=1
        )

    def generate(self, query: str, context: list[dict]) -> str:
        context_block = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in context)
        response = self._client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=400,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Policy excerpts:\n\n{context_block}\n\nQuestion: {query}",
                },
            ],
        )
        return response.choices[0].message.content


class GeminiProvider(LLMProvider):
    name = "generated"

    def __init__(self):
        from google import genai
        from google.genai import types
        self._client = genai.Client(
            api_key=env_str("GOOGLE_API_KEY"),
            http_options=types.HttpOptions(
                timeout=llm_timeout() * 1000,  # milliseconds
                retry_options=types.HttpRetryOptions(attempts=1),  # fail fast; main.py falls back
            ),
        )

    def generate(self, query: str, context: list[dict]) -> str:
        from google.genai import types
        context_block = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in context)
        response = self._client.models.generate_content(
            model=env_str("GEMINI_GENERATION_MODEL", DEFAULT_GEMINI_MODEL),
            contents=f"Policy excerpts:\n\n{context_block}\n\nQuestion: {query}",
            # Generous output budget: "flash" models may spend part of it on
            # internal reasoning before emitting visible text.
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT, max_output_tokens=1024
            ),
        )
        if not response.text:
            raise RuntimeError("Gemini returned no text")
        return response.text


def _key_present(var: str) -> bool:
    """True if the env var is set to something non-blank (a blank value, as a
    hosting dashboard may leave for an unfilled secret, counts as unset)."""
    return bool(env_str(var))


def get_provider() -> LLMProvider:
    """Picks a provider based on whichever API key is present in the environment.
    Precedence if several are set: Anthropic, then OpenAI, then Gemini."""
    if _key_present("ANTHROPIC_API_KEY"):
        return AnthropicProvider()
    if _key_present("OPENAI_API_KEY"):
        return OpenAIProvider()
    if _key_present("GOOGLE_API_KEY"):
        return GeminiProvider()
    return ExtractiveFallback()
