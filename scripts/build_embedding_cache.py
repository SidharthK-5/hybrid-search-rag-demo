"""Precomputes the sample corpus's Gemini embeddings and writes them to
data/embeddings_cache/, so a freshly started container (e.g. a Render free-tier
cold start) needn't call the embedding API for the corpus at all.

Run once after changing any sample document, then commit the new .npy file:

    GOOGLE_API_KEY=... python -m scripts.build_embedding_cache

The cache filename embeds a hash of the chunk text, so a stale file is never
used: edit a document, forget to rebuild, and the app simply re-embeds at startup.
"""
import sys

from app.config import env_str
from app.embeddings import (
    CACHE_DIR,
    GeminiEmbedder,
    corpus_cache_path,
    load_or_embed_corpus,
)
from app.ingest import build_chunks
from app.main import DOCS_DIR


def main() -> int:
    api_key = env_str("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY to build the embedding cache.", file=sys.stderr)
        return 1
    chunks = build_chunks(DOCS_DIR)
    embedder = GeminiEmbedder(api_key)
    matrix = load_or_embed_corpus(chunks, embedder)
    print(f"{matrix.shape[0]} chunk embeddings ({embedder.model}, dim {matrix.shape[1]}) -> "
          f"{corpus_cache_path(chunks, embedder.model, embedder.dim, CACHE_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
