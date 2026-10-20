"""Dense embeddings via the Gemini API, with an on-disk cache for the corpus.

Why a cache: the sample corpus is static, so its embeddings only need computing
once. They are keyed by a hash of (model, dimensionality, chunk contents) and
stored as a small .npy file; edit a document and the key changes, so a stale
cache can never be served. A precomputed cache committed to the repo means a
fresh container (e.g. a Render free-tier cold start) makes zero embedding calls
for the corpus — only one small call per new, uncached user query.

Model choice lives in the environment, not the code (GEMINI_EMBEDDING_MODEL,
GEMINI_EMBEDDING_DIM), so a model change is a config edit. The default,
`gemini-embedding-2`, is what Google's docs currently recommend. It takes no
`task_type`; the task is expressed in the input text itself (see `document_prompt` /
`query_prompt`). Older models (`gemini-embedding-001`, `text-embedding-*`) are the
reverse — they take `task_type` and plain text — and are handled too. Different
models are different vector spaces, so the cache key includes model and
dimensionality: switching models re-embeds the corpus once, never mixes spaces.
"""
import hashlib
import json
import logging
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from app.config import env_int, env_str
from app.ingest import Chunk

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_EMBEDDING_DIM = 768   # docs list 768/1536/3072 as recommended; 768 is plenty for 24 chunks
# Models that predate gemini-embedding-2 take a task_type parameter, not prompt prefixes.
_TASK_TYPE_MODEL_PREFIXES = ("gemini-embedding-001", "text-embedding-", "embedding-")
BATCH_SIZE = 50
REQUEST_TIMEOUT_MS = 10_000
QUERY_CACHE_SIZE = 256
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "embeddings_cache"


class EmbeddingError(RuntimeError):
    """Raised for any failure to obtain embeddings (network, quota, bad response)."""


def embedding_model() -> str:
    return env_str("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def embedding_dim() -> int:
    return env_int("GEMINI_EMBEDDING_DIM", DEFAULT_EMBEDDING_DIM)


def uses_task_type(model: str) -> bool:
    """True for the older model family that takes `task_type` instead of prompt prefixes."""
    return model.removeprefix("models/").startswith(_TASK_TYPE_MODEL_PREFIXES)


def document_prompt(chunk: Chunk) -> str:
    """Document-side input format recommended by the gemini-embedding-2 docs."""
    return f"title: {chunk.source} | text: {chunk.text}"


def query_prompt(query: str) -> str:
    """Query-side input format recommended by the gemini-embedding-2 docs."""
    return f"task: search result | query: {query}"


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class GeminiEmbedder:
    """Thin wrapper over google-genai's embed_content, returning unit vectors."""

    def __init__(self, api_key: str, client=None, model: str | None = None, dim: int | None = None):
        self.model = model or embedding_model()
        self.dim = dim or embedding_dim()
        if client is None:
            from google import genai
            from google.genai import types
            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=REQUEST_TIMEOUT_MS,
                    retry_options=types.HttpRetryOptions(attempts=1),  # fail fast; DenseIndex falls back
                ),
            )
        self._client = client
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        # FastAPI runs sync endpoints on a thread pool; guard the shared LRU.
        self._cache_lock = threading.Lock()

    def _embed(self, texts: list[str], task_type: str | None = None) -> np.ndarray:
        from google.genai import types
        # One Content per text: gemini-embedding-2 otherwise aggregates all the
        # parts of a single request into ONE embedding.
        contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in texts]
        config = types.EmbedContentConfig(output_dimensionality=self.dim, task_type=task_type)
        try:
            result = self._client.models.embed_content(
                model=self.model, contents=contents, config=config
            )
            vectors = [e.values for e in result.embeddings]
        except Exception as exc:  # SDK raises many types; callers only need one
            raise EmbeddingError(f"Gemini embedding request failed: {exc}") from exc
        if len(vectors) != len(texts):
            raise EmbeddingError(f"Expected {len(texts)} embeddings, got {len(vectors)}")
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))

    def embed_documents(self, chunks: list[Chunk]) -> np.ndarray:
        if uses_task_type(self.model):
            prompts, task_type = [c.text for c in chunks], "RETRIEVAL_DOCUMENT"
        else:
            prompts, task_type = [document_prompt(c) for c in chunks], None
        batches = [self._embed(prompts[i:i + BATCH_SIZE], task_type)
                   for i in range(0, len(prompts), BATCH_SIZE)]
        return np.vstack(batches) if batches else np.zeros((0, self.dim), dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds one query; repeats (example buttons, compare view) are served from an LRU."""
        with self._cache_lock:
            if query in self._query_cache:
                self._query_cache.move_to_end(query)
                return self._query_cache[query]
        if uses_task_type(self.model):
            vector = self._embed([query], "RETRIEVAL_QUERY")[0]
        else:
            vector = self._embed([query_prompt(query)])[0]  # network call: outside the lock
        with self._cache_lock:
            self._query_cache[query] = vector
            while len(self._query_cache) > QUERY_CACHE_SIZE:
                self._query_cache.popitem(last=False)
        return vector


def corpus_cache_path(chunks: list[Chunk], model: str, dim: int, cache_dir: Path = CACHE_DIR) -> Path:
    payload = json.dumps([model, dim, [(c.source, c.text) for c in chunks]], ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model)  # env-supplied, so keep it filename-safe
    return cache_dir / f"{safe_model}-{dim}-{digest}.npy"


def load_or_embed_corpus(
    chunks: list[Chunk], embedder: GeminiEmbedder, cache_dir: Path = CACHE_DIR
) -> np.ndarray:
    """Returns the (n_chunks, dim) matrix, from disk if cached, else embeds and caches it."""
    path = corpus_cache_path(chunks, embedder.model, embedder.dim, cache_dir)
    if path.exists():
        try:
            matrix = np.load(path, allow_pickle=False)
            if matrix.shape == (len(chunks), embedder.dim):
                logger.info("Loaded %d cached chunk embeddings from %s", len(chunks), path.name)
                return matrix
            logger.warning("Ignoring cache %s: unexpected shape %s", path.name, matrix.shape)
        except (OSError, ValueError):
            logger.warning("Ignoring unreadable embedding cache %s", path.name)

    matrix = embedder.embed_documents(chunks)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, matrix, allow_pickle=False)
        os.replace(tmp, path)  # atomic: a crash mid-write can't leave a corrupt cache
    except OSError as exc:  # e.g. read-only filesystem — caching is an optimisation only
        logger.warning("Could not write embedding cache: %s", exc)
    return matrix
