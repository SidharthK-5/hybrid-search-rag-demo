"""Ranking correctness, deduplication, normalization and empty-query handling."""
import pytest

from app.ingest import Chunk, build_keyword_index
from app.search import (
    SearchIndex,
    _minmax,
    hybrid_search,
    keyword_search,
    retrieve,
)
from app.vector import VectorResult


class StubVectorIndex:
    """Returns canned raw cosine scores, so fusion math can be tested in isolation."""
    name = "stub"

    def __init__(self, scores: dict[int, float]):
        self.scores = scores
        self.calls = 0

    def search(self, query: str, top_n: int) -> VectorResult:
        self.calls += 1
        return VectorResult(dict(self.scores), self.name)


def make_index(texts: list[str], vector_scores: dict[int, float]) -> SearchIndex:
    chunks = [Chunk(i, f"Doc {i}", t) for i, t in enumerate(texts)]
    return SearchIndex(chunks, build_keyword_index(chunks), StubVectorIndex(vector_scores))


# --- normalization ---

def test_minmax_scales_best_to_one_and_worst_to_zero():
    assert _minmax({1: 10.0, 2: 20.0, 3: 30.0}) == {1: 0.0, 2: 0.5, 3: 1.0}


def test_minmax_single_or_equal_scores_map_to_one_not_zero():
    assert _minmax({7: 3.2}) == {7: 1.0}
    assert _minmax({1: 0.5, 2: 0.5}) == {1: 1.0, 2: 1.0}


def test_minmax_empty():
    assert _minmax({}) == {}


# --- keyword search ---

def test_keyword_search_finds_exact_terms(index):
    scores = keyword_search(index.keyword_conn, "RFP")
    sources = {index.chunk_by_id[cid].source for cid in scores}
    assert sources == {"Procurement Guidelines"}
    assert max(scores.values()) == 1.0


def test_keyword_search_tolerates_trailing_punctuation(index):
    # "leave?" must contribute the term "leave" rather than being dropped.
    assert keyword_search(index.keyword_conn, "parental leave?")


@pytest.mark.parametrize("query", ["", "   ", "the a an", "?!...", 'NOT', 'x" OR "y', "foo*", "(", "NEAR/2"])
def test_keyword_search_never_raises_on_odd_input(index, query):
    assert isinstance(keyword_search(index.keyword_conn, query), dict)


def test_keyword_operators_are_treated_as_literal_words(index):
    # Unquoted, an uppercase NOT would be an FTS5 operator and make this query fail.
    assert keyword_search(index.keyword_conn, "NOT reimbursable")


# --- hybrid ranking ---

@pytest.mark.parametrize("query, expected_source", [
    ("when is a request for proposal required", "Procurement Guidelines"),
    ("how many weeks of medical leave", "Leave Of Absence Policy"),
    ("corporate card reconciliation", "Expense Reimbursement Policy"),
    ("report a phishing email", "It Acceptable Use Policy"),
])
def test_hybrid_ranks_the_right_document_first(index, query, expected_source):
    hits, _ = hybrid_search(index, query, top_k=3)
    assert hits[0]["source"] == expected_source


def test_results_are_sorted_descending_and_respect_top_k(index):
    hits, _ = hybrid_search(index, "employee policy requirements", top_k=4)
    assert len(hits) <= 4
    scores = [h["combined_score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_chunk_found_by_both_retrievers_appears_once(index):
    hits, _ = hybrid_search(index, "medical leave notice", top_k=20)
    ids = [h["chunk_id"] for h in hits]
    assert len(ids) == len(set(ids))
    top = hits[0]
    assert top["keyword_score"] > 0 and top["vector_score"] > 0  # fused, not duplicated


def test_hit_fields_are_complete(index):
    (hit, *_), _ = hybrid_search(index, "data breach", top_k=1)
    assert set(hit) == {"chunk_id", "source", "text", "keyword_score", "vector_score", "combined_score"}


def test_weights_control_the_blend():
    texts = ["alpha beta", "alpha", "gamma"]
    # Keyword matches chunks 0 and 1 for "alpha"; the stub vector side prefers chunk 2.
    index = make_index(texts, {0: 0.2, 1: 0.1, 2: 0.9})
    kw_only = retrieve(index, "alpha", keyword_weight=1.0, vector_weight=0.0).hybrid
    vec_only = retrieve(index, "alpha", keyword_weight=0.0, vector_weight=1.0).hybrid
    assert {h["chunk_id"] for h in kw_only[:2]} == {0, 1}
    assert vec_only[0]["chunk_id"] == 2


def test_hybrid_can_surface_a_chunk_neither_retriever_ranks_first():
    # Chunk 1 is second on both the keyword list and the vector list, yet the best
    # overall once the two signals are fused.
    index = make_index(
        ["alpha alpha alpha", "alpha alpha beta", "alpha beta gamma delta epsilon"],
        {0: 0.1, 1: 0.95, 2: 1.0},
    )
    r = retrieve(index, "alpha")
    assert [h["chunk_id"] for h in r.keyword] == [0, 1, 2]
    assert r.vector[0]["chunk_id"] == 2 and r.vector[1]["chunk_id"] == 1
    assert r.hybrid[0]["chunk_id"] == 1


def test_ties_break_deterministically_by_chunk_id():
    index = make_index(["same", "same", "same"], {0: 0.5, 1: 0.5, 2: 0.5})
    ids = [h["chunk_id"] for h in retrieve(index, "same").hybrid]
    assert ids == [0, 1, 2]


def test_retrieve_returns_all_three_modes_and_backend(index):
    r = retrieve(index, "remote work stipend", top_k=3)
    assert r.keyword and r.vector and r.hybrid
    assert r.vector_backend == "tfidf"
    assert all(h["vector_score"] == 0 for h in r.keyword)   # keyword mode carries no vector signal
    assert all(h["keyword_score"] == 0 for h in r.vector)


# --- empty / degenerate queries ---

@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_query_returns_nothing_and_skips_the_vector_backend(query):
    index = make_index(["alpha"], {0: 1.0})
    hits, backend = hybrid_search(index, query)
    assert hits == []
    assert index.vector_index.calls == 0
    assert backend == "stub"


def test_stopword_only_query_does_not_crash(index):
    hits, _ = hybrid_search(index, "the of and")
    assert isinstance(hits, list)


def test_query_with_no_corpus_overlap_returns_empty_keyword_side(index):
    r = retrieve(index, "xyzzyplugh quuxfrobnicate")
    assert r.keyword == []
