"""The evaluation harness: metric math, query-set integrity, and the guard against
scoring TF-IDF fallback results as if they were dense."""
import numpy as np
import pytest

from app.embeddings import DEFAULT_EMBEDDING_DIM as EMBEDDING_DIM
from app.embeddings import EmbeddingError
from app.ingest import build_keyword_index
from app.search import SearchIndex
from app.vector import DenseIndex, TfidfIndex
from eval.evaluate_retrieval import (
    BackendDegraded,
    Query,
    build_report,
    evaluate,
    hit_at_k,
    load_queries,
    precision_at_k,
    reciprocal_rank,
    run_backend,
    win_tie_loss,
)


def test_precision_at_k():
    rel = [False, True, True, False]
    assert precision_at_k(rel, 1) == 0.0
    assert precision_at_k(rel, 2) == 0.5
    assert precision_at_k(rel, 3) == pytest.approx(2 / 3)


def test_precision_counts_missing_ranks_as_irrelevant():
    assert precision_at_k([True], 3) == pytest.approx(1 / 3)


def test_hit_at_k():
    assert hit_at_k([False, False, True], 2) == 0.0
    assert hit_at_k([False, False, True], 3) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank([True]) == 1.0
    assert reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)
    assert reciprocal_rank([False, False]) == 0.0
    assert reciprocal_rank([]) == 0.0


def test_win_tie_loss():
    hybrid = [{"RR": 1.0}, {"RR": 0.5}, {"RR": 0.5}]
    other = [{"RR": 0.5}, {"RR": 0.5}, {"RR": 1.0}]
    assert win_tie_loss(hybrid, other) == (1, 1, 1)


def test_query_set_is_well_formed_and_covers_every_document(chunks):
    queries = load_queries()
    assert 15 <= len(queries) <= 20
    sources = {c.source for c in chunks}
    assert {q.source for q in queries} == sources                 # all 8 docs, no typos
    assert len({q.query for q in queries}) == len(queries)         # no duplicates
    assert {q.type for q in queries} <= {"lexical", "mixed", "paraphrase"}


def test_evaluate_without_a_key_scores_tfidf_only():
    results, notes, queries = evaluate(include_dense=False)
    assert list(results) == ["TF-IDF"]
    assert set(results["TF-IDF"]) == {"keyword", "vector", "hybrid"}
    assert all(len(rows) == len(queries) for rows in results["TF-IDF"].values())
    assert any("GOOGLE_API_KEY" in n for n in notes)


def test_report_contains_every_method_and_the_lift_table():
    results, _, queries = evaluate(include_dense=False)
    report = build_report(results, queries)
    for text in ("Keyword only", "Vector only", "Hybrid", "MRR", "P@1", "Hybrid lift", "by query type"):
        assert text in report


def test_dense_run_is_rejected_if_it_silently_fell_back(chunks):
    class Down:
        model = "down-model"
        dim = EMBEDDING_DIM

        def embed_query(self, q):
            raise EmbeddingError("quota")

    tfidf = TfidfIndex(chunks)
    dense = DenseIndex(chunks, np.zeros((len(chunks), EMBEDDING_DIM), dtype=np.float32), Down(), tfidf)
    index = SearchIndex(chunks, build_keyword_index(chunks), dense)
    with pytest.raises(BackendDegraded):
        run_backend(index, [Query("medical leave", "Leave Of Absence Policy", "lexical")], dense.name)
