"""Hybrid search RAG demo.

Loads the repo-root .env (if present) on first import, so API keys and tuning
variables work for the server, the eval script and the helper scripts alike —
uvicorn does not read .env files itself. Variables already set in the real
environment take precedence (that is how a hosting dashboard like Render's
supplies them), and a missing .env is simply ignored.
"""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
