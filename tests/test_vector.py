"""Vector backends, the Gemini embedder (mocked — no network), the corpus cache,
and the automatic fallback to TF-IDF."""
from types import SimpleNamespace

import numpy as np
import pytest

from app import vector
from app.embeddings import (
    DEFAULT_EMBEDDING_DIM as EMBEDDING_DIM,
)
from app.embeddings import (
    DEFAULT_EMBEDDING_MODEL as EMBEDDING_MODEL,
)
from app.embeddings import (
    EmbeddingError,
    GeminiEmbedder,
    corpus_cache_path,
    embedding_dim,
    embedding_model,
    load_or_embed_corpus,
    query_prompt,
    uses_task_type,
)
from app.ingest import Chunk
from app.vector import DenseIndex, TfidfIndex, build_vector_index


class FakeGenAIClient:
    """Mimics client.models.embed_content with deterministic pseudo-embeddings."""

    def __init__(self, fail: bool = False, drop_one: bool = False):
        self.calls: list[dict] = []
        self.fail, self.drop_one = fail, drop_one
        self.models = SimpleNamespace(embed_content=self._embed_content)

    def _embed_content(self, model, contents, config):
        self.calls.append({"model": model, "n": len(contents), "config": config})
        if self.fail:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        vectors = [
            SimpleNamespace(values=list(np.random.default_rng(len(self.calls) * 100 + i).normal(size=EMBEDDING_DIM)))
            for i in range(len(contents))
        ]
        return SimpleNamespace(embeddings=vectors[:-1] if self.drop_one else vectors)


@pytest.fixture
def tiny_chunks():
    return [Chunk(0, "Doc A", "alpha text"), Chunk(1, "Doc B", "beta text")]


# --- GeminiEmbedder ---

def test_embedder_requests_the_current_model_with_output_dimensionality(tiny_chunks):
    client = FakeGenAIClient()
    GeminiEmbedder("k", client=client).embed_documents(tiny_chunks)
    call = client.calls[0]
    assert call["model"] == EMBEDDING_MODEL == "gemini-embedding-2"
    assert call["config"].output_dimensionality == EMBEDDING_DIM
    assert call["config"].task_type is None
    assert call["n"] == 2  # one Content per chunk -> one embedding per chunk


def test_embeddings_are_unit_length(tiny_chunks):
    matrix = GeminiEmbedder("k", client=FakeGenAIClient()).embed_documents(tiny_chunks)
    assert matrix.shape == (2, EMBEDDING_DIM)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)


def test_documents_are_embedded_in_batches(monkeypatch):
    monkeypatch.setattr("app.embeddings.BATCH_SIZE", 2)
    client = FakeGenAIClient()
    chunks = [Chunk(i, "D", f"t{i}") for i in range(5)]
    assert GeminiEmbedder("k", client=client).embed_documents(chunks).shape[0] == 5
    assert [c["n"] for c in client.calls] == [2, 2, 1]


def test_api_failure_surfaces_as_embedding_error(tiny_chunks):
    with pytest.raises(EmbeddingError):
        GeminiEmbedder("k", client=FakeGenAIClient(fail=True)).embed_documents(tiny_chunks)


def test_wrong_embedding_count_is_an_error(tiny_chunks):
    with pytest.raises(EmbeddingError):
        GeminiEmbedder("k", client=FakeGenAIClient(drop_one=True)).embed_documents(tiny_chunks)


def test_repeated_queries_hit_the_lru_not_the_api():
    client = FakeGenAIClient()
    embedder = GeminiEmbedder("k", client=client)
    first = embedder.embed_query("notice period")
    second = embedder.embed_query("notice period")
    assert len(client.calls) == 1
    assert np.array_equal(first, second)
    assert query_prompt("x").startswith("task: search result | query:")


# --- corpus cache ---

def test_corpus_embeddings_are_cached_on_disk(tmp_path, tiny_chunks):
    client = FakeGenAIClient()
    embedder = GeminiEmbedder("k", client=client)
    first = load_or_embed_corpus(tiny_chunks, embedder, tmp_path)
    second = load_or_embed_corpus(tiny_chunks, embedder, tmp_path)
    assert len(client.calls) == 1            # second load never touched the API
    assert np.array_equal(first, second)
    assert len(list(tmp_path.glob("*.npy"))) == 1


def test_cache_key_changes_when_chunk_text_changes(tmp_path, tiny_chunks):
    edited = [tiny_chunks[0], Chunk(1, "Doc B", "beta text, edited")]
    assert (corpus_cache_path(tiny_chunks, EMBEDDING_MODEL, EMBEDDING_DIM, tmp_path)
            != corpus_cache_path(edited, EMBEDDING_MODEL, EMBEDDING_DIM, tmp_path))


def test_corrupt_cache_is_ignored_and_rebuilt(tmp_path, tiny_chunks):
    corpus_cache_path(tiny_chunks, EMBEDDING_MODEL, EMBEDDING_DIM, tmp_path).write_bytes(b"not a numpy file")
    matrix = load_or_embed_corpus(tiny_chunks, GeminiEmbedder("k", client=FakeGenAIClient()), tmp_path)
    assert matrix.shape == (2, EMBEDDING_DIM)


def test_unwritable_cache_dir_does_not_break_embedding(tmp_path, tiny_chunks):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    matrix = load_or_embed_corpus(tiny_chunks, GeminiEmbedder("k", client=FakeGenAIClient()), blocker / "sub")
    assert matrix.shape[0] == 2


# --- TF-IDF backend ---

def test_tfidf_ranks_lexically_similar_chunk_first(chunks):
    result = TfidfIndex(chunks).search("request for proposal threshold", top_n=3)
    best = max(result.scores, key=result.scores.get)
    assert chunks[best].source == "Procurement Guidelines"
    assert result.backend == "tfidf"


def test_tfidf_returns_nothing_for_unknown_vocabulary(chunks):
    assert TfidfIndex(chunks).search("zzzzqqqq", top_n=3).scores == {}


# --- DenseIndex + fallback ---

class StubEmbedder:
    model = "stub-model"
    dim = EMBEDDING_DIM

    def __init__(self, vec=None, error=None):
        self.vec, self.error, self.calls = vec, error, 0

    def embed_query(self, query):
        self.calls += 1
        if self.error:
            raise self.error
        return self.vec


def test_dense_index_scores_by_cosine(chunks):
    matrix = np.eye(len(chunks), EMBEDDING_DIM, dtype=np.float32)
    query_vec = matrix[3]  # identical to chunk 3's vector
    index = DenseIndex(chunks, matrix, StubEmbedder(query_vec), TfidfIndex(chunks))
    result = index.search("anything", top_n=2)
    assert result.backend == index.name == "stub-model"
    assert max(result.scores, key=result.scores.get) == chunks[3].chunk_id
    assert len(result.scores) == 2


def test_dense_failure_falls_back_to_tfidf_and_pauses_the_api(chunks):
    embedder = StubEmbedder(error=EmbeddingError("quota"))
    index = DenseIndex(chunks, np.zeros((len(chunks), EMBEDDING_DIM), dtype=np.float32), embedder, TfidfIndex(chunks))
    first = index.search("medical leave", top_n=3)
    second = index.search("medical leave", top_n=3)
    assert first.backend == second.backend == "tfidf (fallback)"
    assert first.scores                      # the fallback still returns real results
    assert embedder.calls == 1               # circuit breaker: no second attempt during cooldown


# --- build_vector_index selection ---

def test_no_api_key_builds_tfidf_without_importing_the_sdk(chunks, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not construct an embedder without a key")
    monkeypatch.setattr(vector, "GeminiEmbedder", boom)
    assert isinstance(build_vector_index(chunks), TfidfIndex)


def test_blank_api_key_counts_as_unset(chunks, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "   ")
    assert isinstance(build_vector_index(chunks), TfidfIndex)


def test_api_key_builds_dense_index(chunks, monkeypatch, tmp_path):
    monkeypatch.setattr(vector, "GeminiEmbedder", lambda key: GeminiEmbedder(key, client=FakeGenAIClient()))
    monkeypatch.setattr("app.embeddings.CACHE_DIR", tmp_path)
    monkeypatch.setattr(vector, "load_or_embed_corpus",
                        lambda c, e: load_or_embed_corpus(c, e, tmp_path))
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    assert isinstance(build_vector_index(chunks), DenseIndex)


def test_startup_embedding_failure_degrades_to_tfidf(chunks, monkeypatch, tmp_path):
    monkeypatch.setattr(vector, "GeminiEmbedder",
                        lambda key: GeminiEmbedder(key, client=FakeGenAIClient(fail=True)))
    monkeypatch.setattr(vector, "load_or_embed_corpus",
                        lambda c, e: load_or_embed_corpus(c, e, tmp_path))
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    assert isinstance(build_vector_index(chunks), TfidfIndex)


def test_real_sdk_serializes_the_documented_request_shape():
    """Runs the real google-genai client against a mock HTTP transport (no network),
    so a mistake in how we call the SDK shows up here, not in production."""
    import json

    import httpx
    from google import genai
    from google.genai import types

    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append({"url": str(request.url), "body": body})
        return httpx.Response(200, json={"embeddings": [{"values": [0.1] * EMBEDDING_DIM}
                                                          for _ in body["requests"]]})

    client = genai.Client(
        api_key="fake",
        http_options=types.HttpOptions(httpx_client=httpx.Client(transport=httpx.MockTransport(handler))),
    )
    matrix = GeminiEmbedder("fake", client=client).embed_documents(
        [Chunk(0, "Doc A", "alpha"), Chunk(1, "Doc B", "beta")])

    assert matrix.shape == (2, EMBEDDING_DIM)
    (call,) = seen
    assert call["url"].endswith("/models/gemini-embedding-2:batchEmbedContents")
    reqs = call["body"]["requests"]
    assert len(reqs) == 2                                         # one embedding per chunk
    assert all(r["outputDimensionality"] == EMBEDDING_DIM for r in reqs)
    assert all("taskType" not in r for r in reqs)                 # gemini-embedding-2 takes no task_type
    assert reqs[0]["content"]["parts"][0]["text"] == "title: Doc A | text: alpha"


# --- model choice comes from the environment ---

def test_defaults_when_env_is_unset():
    assert embedding_model() == "gemini-embedding-2"
    assert embedding_dim() == 768


def test_env_overrides_model_and_dimensionality(monkeypatch, tiny_chunks):
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-3")
    monkeypatch.setenv("GEMINI_EMBEDDING_DIM", "1536")
    client = FakeGenAIClient()
    embedder = GeminiEmbedder("k", client=client)
    assert (embedder.model, embedder.dim) == ("gemini-embedding-3", 1536)
    embedder._embed(["x"])
    assert client.calls[0]["model"] == "gemini-embedding-3"
    assert client.calls[0]["config"].output_dimensionality == 1536


def test_invalid_dimensionality_in_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("GEMINI_EMBEDDING_DIM", "lots")
    assert embedding_dim() == EMBEDDING_DIM


def test_cache_is_keyed_by_model_and_dimensionality(tmp_path, tiny_chunks):
    base = corpus_cache_path(tiny_chunks, "gemini-embedding-2", 768, tmp_path)
    assert base != corpus_cache_path(tiny_chunks, "gemini-embedding-3", 768, tmp_path)
    assert base != corpus_cache_path(tiny_chunks, "gemini-embedding-2", 1536, tmp_path)


def test_model_names_with_slashes_make_safe_cache_filenames(tmp_path, tiny_chunks):
    path = corpus_cache_path(tiny_chunks, "models/gemini-embedding-2", 768, tmp_path)
    assert path.parent == tmp_path


@pytest.mark.parametrize("model, legacy", [
    ("gemini-embedding-2", False),
    ("models/gemini-embedding-2", False),
    ("gemini-embedding-3", False),
    ("gemini-embedding-001", True),
    ("text-embedding-004", True),
])
def test_task_type_family_detection(model, legacy):
    assert uses_task_type(model) is legacy


def test_legacy_models_use_task_type_and_plain_text(tiny_chunks):
    client = FakeGenAIClient()
    embedder = GeminiEmbedder("k", client=client, model="gemini-embedding-001")
    embedder.embed_documents(tiny_chunks)
    embedder.embed_query("notice period")
    doc_call, query_call = client.calls
    assert doc_call["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert query_call["config"].task_type == "RETRIEVAL_QUERY"
