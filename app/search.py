"""Hybrid retrieval: keyword search (SQLite FTS5 / BM25) + vector search (dense
Gemini embeddings, or TF-IDF when no API key is set), merged and deduplicated,
then ranked.

This is the same shape as the production hybrid search pipeline it's modeled on:

    query -> [keyword search] -> \\
                                    combine + dedupe -> sort -> top-k
    query -> [vector search]  -> /

The three ranking modes (keyword-only, vector-only, hybrid) share one code path
so the evaluation harness (eval/) and the API measure exactly the same thing.

Scoring note: BM25 scores and cosine similarities live on different scales (dense
cosines cluster around 0.6-0.8 for everything in a small corpus), so each list is
min-max normalized to 0-1 *per query* before the weighted sum. Without that, the
signal with the wider spread silently dominates the blend.
"""
import re
import sqlite3
from dataclasses import dataclass, field

from app.ingest import Chunk, build_keyword_index
from app.vector import VectorIndex, VectorResult, build_vector_index

KEYWORD_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6
# Candidates pulled from each retriever before fusion. Fixed (not top_k * n) so
# the ranking for a query doesn't change with the page size requested.
CANDIDATE_POOL = 30

# A deliberately short stopword list: just enough to stop filler words from
# forcing FTS5 to require every one of them in a match.
_STOPWORDS = {
    "a", "an", "the", "do", "does", "did", "to", "for", "of", "is", "are",
    "in", "on", "at", "and", "or", "i", "how", "much", "many", "need",
    "give", "my", "me", "can", "what", "when", "will", "am", "it", "this",
    "that", "with", "as", "be",
}


@dataclass
class SearchIndex:
    """Everything the retrieval functions need, built once at startup."""
    chunks: list[Chunk]
    keyword_conn: sqlite3.Connection
    vector_index: VectorIndex
    chunk_by_id: dict[int, Chunk] = field(init=False)

    def __post_init__(self):
        self.chunk_by_id = {c.chunk_id: c for c in self.chunks}


@dataclass
class Retrieval:
    """All three rankings for one query, computed from a single pair of retriever calls."""
    keyword: list[dict]
    vector: list[dict]
    hybrid: list[dict]
    vector_backend: str


def build_search_index(chunks: list[Chunk]) -> SearchIndex:
    return SearchIndex(chunks, build_keyword_index(chunks), build_vector_index(chunks))


def _minmax(scores: dict[int, float]) -> dict[int, float]:
    """Rescales scores to 0-1 (best = 1). A single candidate, or all-equal scores,
    all map to 1.0 rather than 0 — a lone match shouldn't be scored as no match."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    span = hi - lo
    if span == 0:
        return {cid: 1.0 for cid in scores}
    return {cid: (s - lo) / span for cid, s in scores.items()}


def keyword_search(conn: sqlite3.Connection, query: str, top_n: int = CANDIDATE_POOL) -> dict[int, float]:
    """Returns {chunk_id: score} from SQLite FTS5's bm25 ranking, normalized to a
    0-1 "higher is better" score (bm25() itself is negative, more negative = better)."""
    # Tokenize on alphanumeric runs so "leave?" or "don't" still contribute terms,
    # and double-quote each term so FTS5 treats it as a literal (otherwise an
    # uppercase NOT / OR / NEAR in the query would be parsed as an operator).
    # Use OR rather than the implicit AND, so a chunk matching any meaningful
    # term is a candidate — bm25 still ranks chunks matching more terms higher.
    terms = [w for w in re.findall(r"[A-Za-z0-9]+", query) if w.lower() not in _STOPWORDS]
    if not terms:
        return {}
    safe_query = " OR ".join(f'"{t}"' for t in terms)
    try:
        rows = conn.execute(
            "SELECT chunk_id, bm25(chunks) AS rank FROM chunks WHERE chunks MATCH ? "
            "ORDER BY rank LIMIT ?",
            (safe_query, top_n),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    return _minmax({chunk_id: -rank for chunk_id, rank in rows})


def vector_search(index: SearchIndex, query: str, top_n: int = CANDIDATE_POOL) -> VectorResult:
    """Raw cosine scores from whichever vector backend is active."""
    return index.vector_index.search(query, top_n)


def _hit(index: SearchIndex, cid: int, kw: float, vec: float, score: float) -> dict:
    chunk = index.chunk_by_id[cid]
    return {
        "chunk_id": cid,
        "source": chunk.source,
        "text": chunk.text,
        "keyword_score": round(kw, 4),
        "vector_score": round(vec, 4),
        "combined_score": round(score, 4),
    }


def _rank(index: SearchIndex, kw: dict[int, float], vec: dict[int, float],
          kw_weight: float, vec_weight: float, top_k: int) -> list[dict]:
    """Merges the two score tables by chunk_id (the dedupe), weights, sorts, truncates.
    Ties break on chunk_id so rankings are deterministic."""
    ranked = sorted(
        ((cid, kw_weight * kw.get(cid, 0.0) + vec_weight * vec.get(cid, 0.0))
         for cid in set(kw) | set(vec)),
        key=lambda pair: (-pair[1], pair[0]),
    )
    return [
        _hit(index, cid, kw.get(cid, 0.0), vec.get(cid, 0.0), score)
        for cid, score in ranked[:top_k]
    ]


def retrieve(
    index: SearchIndex,
    query: str,
    top_k: int = 5,
    keyword_weight: float = KEYWORD_WEIGHT,
    vector_weight: float = VECTOR_WEIGHT,
) -> Retrieval:
    """Runs both retrievers once and derives all three rankings from the results.
    A blank query returns empty rankings without touching either retriever."""
    if not query.strip():
        return Retrieval([], [], [], index.vector_index.name)
    kw = keyword_search(index.keyword_conn, query)
    vec_result = vector_search(index, query)
    vec = _minmax(vec_result.scores)
    return Retrieval(
        keyword=_rank(index, kw, {}, 1.0, 0.0, top_k),
        vector=_rank(index, {}, vec, 0.0, 1.0, top_k),
        hybrid=_rank(index, kw, vec, keyword_weight, vector_weight, top_k),
        vector_backend=vec_result.backend,
    )


def hybrid_search(
    index: SearchIndex,
    query: str,
    top_k: int = 5,
    keyword_weight: float = KEYWORD_WEIGHT,
    vector_weight: float = VECTOR_WEIGHT,
) -> tuple[list[dict], str]:
    """Hybrid ranking only. Returns (hits, vector_backend_used)."""
    result = retrieve(index, query, top_k, keyword_weight, vector_weight)
    return result.hybrid, result.vector_backend
