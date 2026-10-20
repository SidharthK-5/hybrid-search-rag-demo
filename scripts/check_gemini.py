"""Quick diagnostic: is a Gemini failure in this app, or at Google?

Makes one embedding call and a few generation calls with your configured keys and
models (GOOGLE_API_KEY, GEMINI_EMBEDDING_MODEL, GEMINI_GENERATION_MODEL, read from
.env), and prints latency and the exact error for each. Consistently slow calls or
503/504 errors here — outside the app — mean the provider is degraded, not this code.

    python -m scripts.check_gemini [--runs 3] [--model gemini-3.5-flash-lite ...]
"""
import argparse
import sys
import time

from app.config import env_str
from app.embeddings import GeminiEmbedder
from app.generate import DEFAULT_GEMINI_MODEL, llm_timeout

PROMPT = ("Policy excerpts:\n\n[Procurement Guidelines]\nPurchases above $25,000 require a formal "
          "request for proposal (RFP) process.\n\nQuestion: When is a request for proposal required?")


def timed(fn) -> str:
    start = time.monotonic()
    try:
        detail = fn()
        return f"OK    {time.monotonic() - start:5.1f}s  {detail}"
    except Exception as exc:  # report whatever the SDK raised
        return f"FAIL  {time.monotonic() - start:5.1f}s  {type(exc).__name__}: {str(exc)[:140]}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=int, default=3, help="generation calls per model")
    parser.add_argument("--model", action="append", help="generation model(s) to try (default: configured)")
    args = parser.parse_args(argv)

    api_key = env_str("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY is not set (checked the environment and .env).", file=sys.stderr)
        return 1

    from google import genai
    from google.genai import types

    embedder = GeminiEmbedder(api_key)
    print(f"Embedding   {embedder.model} (dim {embedder.dim})")
    print("  ", timed(lambda: f"{embedder.embed_query('medical leave notice').shape[0]}-dim vector"))

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
        timeout=llm_timeout() * 1000, retry_options=types.HttpRetryOptions(attempts=1)))
    for model in args.model or [env_str("GEMINI_GENERATION_MODEL", DEFAULT_GEMINI_MODEL)]:
        print(f"Generation  {model} (timeout {llm_timeout()}s)")
        for _ in range(args.runs):
            print(
                "  ",
                timed(
                    lambda: repr(
                        client.models.generate_content(
                            model=model, contents=PROMPT
                        ).text.strip()[:60]
                    )
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
