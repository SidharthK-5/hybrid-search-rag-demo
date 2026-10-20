"""Vector-search backends behind one small interface.

Two interchangeable backends score chunks against a query:

  * `DenseIndex`  — Gemini embeddings + cosine similarity (used when GOOGLE_API_KEY is set;
                    model chosen by GEMINI_EMBEDDING_MODEL)
  * `TfidfIndex`  — TF-IDF + cosine similarity (zero keys, zero cost, zero downloads)

The TF-IDF index is always built. It is the whole backend when no key is set, and
it is also the safety net for the dense backend: if the embedding API is down,
rate-limited, or times out, a query degrades to TF-IDF instead of returning a 500.
Every result reports which backend actually produced it, so the UI never
claims "dense" for results that quietly came from the fallback.
"""
import logging
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import env_str
from app.embeddings import (
    EmbeddingError,
    GeminiEmbedder,
    load_or_embed_corpus,
)
from app.ingest import Chunk

logger = logging.getLogger(__name__)

TFIDF_BACKEND = "tfidf"
TFIDF_FALLBACK_BACKEND = "tfidf (fallback)"
FALLBACK_COOLDOWN_SECONDS = 60.0


@dataclass
class VectorResult:
    """Raw cosine scores ({chunk_id: score}) plus the backend that produced them."""
    scores: dict[int, float]
    backend: str


class VectorIndex(Protocol):
    name: str

    def search(self, query: str, top_n: int) -> VectorResult:
        ...


class TfidfIndex:
    """TF-IDF vector space over the chunks, queried by cosine similarity."""

    name = TFIDF_BACKEND

    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query: str, top_n: int, backend: str = TFIDF_BACKEND) -> VectorResult:
        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._matrix).flatten()
        if sims.size == 0 or sims.max() <= 0:
            return VectorResult({}, backend)
        top_idx = np.argsort(sims)[::-1][:top_n]
        return VectorResult(
            {self._chunks[i].chunk_id: float(sims[i]) for i in top_idx if sims[i] > 0}, backend
        )


class DenseIndex:
    """Dense Gemini embeddings (precomputed for the corpus) + cosine similarity."""

    def __init__(self, chunks: list[Chunk], matrix: np.ndarray, embedder: GeminiEmbedder,
                 fallback: TfidfIndex):
        self.name = embedder.model
        self._chunks = chunks
        self._matrix = matrix          # rows are unit vectors, so dot product == cosine
        self._embedder = embedder
        self._fallback = fallback
        self._paused_until = 0.0

    def search(self, query: str, top_n: int) -> VectorResult:
        # Circuit breaker: after a failure, skip the API for a minute instead of
        # making every visitor wait out a timeout while the API is unhealthy.
        if time.monotonic() >= self._paused_until:
            try:
                q = self._embedder.embed_query(query)
                sims = self._matrix @ q
                top_idx = np.argsort(sims)[::-1][:top_n]
                return VectorResult(
                    {self._chunks[i].chunk_id: float(sims[i]) for i in top_idx}, self.name
                )
            except EmbeddingError as exc:
                logger.warning("Query embedding failed, using TF-IDF fallback: %s", exc)
                self._paused_until = time.monotonic() + FALLBACK_COOLDOWN_SECONDS
        return self._fallback.search(query, top_n, backend=TFIDF_FALLBACK_BACKEND)


def build_vector_index(chunks: list[Chunk]) -> VectorIndex:
    """Dense index if GOOGLE_API_KEY is set and the corpus embeds; TF-IDF otherwise."""
    tfidf = TfidfIndex(chunks)
    api_key = env_str("GOOGLE_API_KEY")
    if not api_key:
        return tfidf
    try:
        embedder = GeminiEmbedder(api_key)
        matrix = load_or_embed_corpus(chunks, embedder)
    except Exception as exc:  # any startup failure must leave the app usable
        logger.warning("Dense embeddings unavailable (%s); using TF-IDF.", exc)
        return tfidf
    return DenseIndex(chunks, matrix, embedder, tfidf)
