"""FastAPI integration tests via TestClient: endpoints, input limits, rate limiting."""
import pytest
from fastapi import HTTPException

from app.safety import MAX_BODY_BYTES, checked_query
from app.schemas import MAX_QUERY_CHARS, SearchRequest


def post(client, path, **body):
    return client.post(path, json=body)


# --- /health and static ---

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "chunks_indexed": 24, "vector_backend": "tfidf",
                        "llm_mode": "extractive"}


def test_index_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]


# --- /search ---

def test_search_returns_ranked_hits_and_extractive_answer(client):
    r = post(client, "/search", query="When is a request for proposal required?", top_k=3)
    assert r.status_code == 200
    body = r.json()
    assert body["results"][0]["source"] == "Procurement Guidelines"
    assert len(body["results"]) == 3
    assert body["answer_mode"] == "extractive" and "[Procurement Guidelines]" in body["answer"]
    assert body["vector_backend"] == "tfidf"


def test_keyless_answer_says_no_llm_is_configured(client):
    assert "No LLM configured" in post(client, "/search", query="data breach").json()["answer"]


def test_search_without_answer(client):
    body = post(client, "/search", query="data breach", generate_answer=False).json()
    assert body["answer"] is None and body["results"]


def test_search_with_no_matches_returns_empty_results(client):
    r = post(client, "/search", query="xyzzyplugh")
    assert r.status_code == 200 and r.json()["results"] == []


@pytest.mark.parametrize("payload", [
    {},                                   # missing query
    {"query": ""},                        # empty
    {"query": "ok", "top_k": 0},
    {"query": "ok", "top_k": 21},
    {"query": "ok", "generate_answer": "maybe"},
])
def test_search_rejects_invalid_payloads(client, payload):
    assert client.post("/search", json=payload).status_code == 422


def test_whitespace_only_query_is_rejected_by_the_handler(client):
    # Passes the schema (min_length=1) — the API layer must catch it.
    r = post(client, "/search", query="   \n\t ")
    assert r.status_code == 422 and "text" in r.json()["detail"]


def test_malformed_json_is_a_422_not_a_500(client):
    r = client.post("/search", content="{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_llm_failure_falls_back_to_extractive_and_pauses_the_provider(client, monkeypatch):
    from app.main import state

    class Broken:
        name = "generated"
        calls = 0

        def generate(self, query, context):
            Broken.calls += 1
            raise RuntimeError("provider down")

    monkeypatch.setitem(state, "llm", Broken())
    for _ in range(3):
        r = post(client, "/search", query="medical leave")
        assert r.status_code == 200
        assert r.json()["answer_mode"] == "extractive"
        # The LLM *is* configured here, so the notice must not claim otherwise.
        assert "temporarily unavailable" in r.json()["answer"]
        assert "No LLM configured" not in r.json()["answer"]
    assert Broken.calls == 1  # circuit breaker: later requests don't wait on the dead provider


# --- /compare ---

def test_compare_returns_three_rankings(client):
    r = post(client, "/compare", query="medical leave notice", top_k=3)
    assert r.status_code == 200
    body = r.json()
    for mode in ("keyword", "vector", "hybrid"):
        assert 0 < len(body[mode]) <= 3
    assert body["vector_backend"] == "tfidf"


def test_compare_shares_validation_with_search(client):
    assert post(client, "/compare", query="  ").status_code == 422
    assert post(client, "/compare", query="x" * (MAX_QUERY_CHARS + 1)).status_code == 422


# --- query length cap, enforced at the API layer ---

def test_query_at_the_cap_is_accepted_and_one_over_is_rejected(client):
    assert post(client, "/search", query="a" * MAX_QUERY_CHARS).status_code == 200
    assert post(client, "/search", query="a" * (MAX_QUERY_CHARS + 1)).status_code == 422


def test_handler_enforces_cap_even_if_schema_validation_is_bypassed():
    # model_construct skips validation — the situation a refactor or internal caller could create.
    req = SearchRequest.model_construct(query="a" * (MAX_QUERY_CHARS + 1), top_k=5, generate_answer=False)
    with pytest.raises(HTTPException) as exc:
        checked_query(req.query)
    assert exc.value.status_code == 422


def test_checked_query_normalizes_whitespace_and_control_characters():
    assert checked_query("  how   many\n days\x00 ? ") == "how many days ?"


def test_oversized_body_is_rejected_with_413_before_parsing(client):
    big = '{"query": "' + "a" * (MAX_BODY_BYTES + 10) + '"}'
    r = client.post("/search", content=big, headers={"content-type": "application/json"})
    assert r.status_code == 413


def test_oversized_chunked_body_without_content_length_is_rejected(client):
    def stream():
        yield b'{"query": "'
        for _ in range(MAX_BODY_BYTES // 1024 + 2):
            yield b"a" * 1024
        yield b'"}'

    r = client.post("/search", content=stream(), headers={"content-type": "application/json"})
    assert r.status_code == 413


# --- rate limiting ---

def test_per_ip_rate_limit_returns_429_with_retry_after(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT", "3/minute")
    codes = [post(client, "/search", query="leave").status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    r = post(client, "/search", query="leave")
    assert r.headers["retry-after"] == "60" and "Rate limit" in r.json()["detail"]


def test_search_and_compare_share_one_budget(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT", "2/minute")
    assert post(client, "/search", query="leave").status_code == 200
    assert post(client, "/compare", query="leave").status_code == 200
    assert post(client, "/compare", query="leave").status_code == 429


def test_health_is_never_rate_limited(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT", "1/minute")
    assert all(client.get("/health").status_code == 200 for _ in range(5))


def test_global_limit_applies_across_different_clients(client, monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    monkeypatch.setenv("GLOBAL_RATE_LIMIT", "2/hour")
    codes = [
        client.post("/search", json={"query": "leave"}, headers={"x-forwarded-for": f"198.51.100.{i}"}).status_code
        for i in range(4)
    ]
    assert codes == [200, 200, 429, 429]


def test_forwarded_header_is_ignored_without_a_trusted_proxy(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT", "2/minute")
    codes = [
        client.post("/search", json={"query": "leave"}, headers={"x-forwarded-for": f"203.0.113.{i}"}).status_code
        for i in range(4)
    ]
    assert codes == [200, 200, 429, 429]  # rotating the header must not earn fresh budget


def test_trusted_proxy_hops_use_the_rightmost_trusted_entry(client, monkeypatch):
    """With one trusted proxy, a client-prepended fake entry must not change the key,
    but a genuinely different client (different last entry) gets its own budget."""
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    monkeypatch.setenv("RATE_LIMIT", "2/minute")

    def hit(xff):
        return client.post("/search", json={"query": "leave"}, headers={"x-forwarded-for": xff}).status_code

    assert [hit("1.1.1.1, 9.9.9.9"), hit("2.2.2.2, 9.9.9.9"), hit("3.3.3.3, 9.9.9.9")] == [200, 200, 429]
    assert hit("1.1.1.1, 8.8.8.8") == 200


def test_missing_forwarded_header_falls_back_to_the_socket_address(client, monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "2")
    monkeypatch.setenv("RATE_LIMIT", "1/minute")
    assert post(client, "/search", query="leave").status_code == 200
    assert post(client, "/search", query="leave").status_code == 429


# --- dense vector path end to end (fake embedder; runs in CI without a key) ---

class _FakeEmbedder:
    model = "fake-embedding-model"
    dim = 24   # >= number of sample chunks, so np.eye gives each chunk its own unit vector

    def __init__(self, vector=None, error=None):
        self.vector, self.error = vector, error

    def embed_query(self, query):
        if self.error:
            from app.embeddings import EmbeddingError
            raise EmbeddingError("quota exceeded")
        return self.vector


def _use_dense_index(monkeypatch, embedder):
    import numpy as np

    from app.main import state
    from app.search import SearchIndex
    from app.vector import DenseIndex, TfidfIndex

    old = state["index"]
    # Unit vector per chunk; the "query" embedding equals chunk 11's vector, so that chunk must win.
    matrix = np.eye(len(old.chunks), embedder.dim, dtype=np.float32)
    embedder.vector = embedder.vector if embedder.vector is not None else matrix[11]
    dense = DenseIndex(old.chunks, matrix, embedder, TfidfIndex(old.chunks))
    monkeypatch.setitem(state, "index", SearchIndex(old.chunks, old.keyword_conn, dense))
    return old


def test_search_through_a_dense_index_reports_its_backend(client, monkeypatch):
    _use_dense_index(monkeypatch, _FakeEmbedder())
    body = post(client, "/search", query="anything at all", top_k=5, generate_answer=False).json()
    assert body["vector_backend"] == "fake-embedding-model"
    assert client.get("/health").json()["vector_backend"] == "fake-embedding-model"


def test_compare_vector_ranking_follows_the_dense_scores(client, monkeypatch):
    _use_dense_index(monkeypatch, _FakeEmbedder())
    body = post(client, "/compare", query="anything at all", top_k=3).json()
    assert body["vector"][0]["chunk_id"] == 11


def test_embedding_outage_degrades_to_tfidf_via_the_api(client, monkeypatch):
    _use_dense_index(monkeypatch, _FakeEmbedder(error=True))
    r = post(client, "/search", query="medical leave notice", generate_answer=False)
    assert r.status_code == 200
    body = r.json()
    assert body["vector_backend"] == "tfidf (fallback)"
    assert body["results"][0]["source"] == "Leave Of Absence Policy"
