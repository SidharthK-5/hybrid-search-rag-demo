"""Retrieval evaluation: keyword-only vs vector-only vs hybrid on labelled queries.

Methodology
-----------
* Ground truth is *document-level*: each query is labelled with the one sample
  document that answers it. A retrieved chunk is relevant if it comes from that
  document. (Chunk-level labels would be finer but are costly to maintain; with
  2-4 chunks per document, document-level labels are a reasonable approximation.)
* Metrics, computed per query then averaged:
    P@k      fraction of the top-k *chunks* that are relevant. Note the ceiling:
             a document with only 2 chunks caps P@3 at 0.67 even for a perfect system.
    Hit@k    1 if any relevant chunk is in the top k (a.k.a. success@k / recall@k
             with a single relevant document).
    MRR      mean of 1 / rank of the first relevant chunk (0 if not in the top 10).
* Every method is scored through the same `retrieve()` the API uses, so the
  numbers describe the shipped system, not a re-implementation.
* Queries are tagged lexical / mixed / paraphrase by how much vocabulary they
  share with the answer, so results can be broken down instead of averaged away.
* With a GOOGLE_API_KEY set, the dense Gemini backend is evaluated as well. If an
  embedding call fails mid-run the backend is discarded rather than silently
  scored on TF-IDF fallback results.

Limitations (also in the README): 20 queries over 8 short documents is a smoke-test
scale, the queries were written by the repo author, and differences of a few
points are within noise. Treat the numbers as a demonstration of the methodology.

Usage:  python -m eval.evaluate_retrieval [--markdown eval/results.md]
"""
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from app.config import env_str
from app.embeddings import embedding_model
from app.ingest import build_chunks, build_keyword_index
from app.search import SearchIndex, retrieve
from app.vector import DenseIndex, TfidfIndex, build_vector_index

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "sample_docs"
QUERIES_PATH = Path(__file__).resolve().parent / "queries.json"

MAX_RANK = 10          # rankings are cut here; MRR treats "not found" as 0
PRECISION_KS = (1, 3)
HIT_K = 3
METHODS = ("keyword", "vector", "hybrid")


@dataclass
class Query:
    query: str
    source: str
    type: str


def load_queries(path: Path = QUERIES_PATH) -> list[Query]:
    return [Query(**row) for row in json.loads(path.read_text(encoding="utf-8"))]


def relevance(hits: list[dict], source: str) -> list[bool]:
    return [h["source"] == source for h in hits]


def precision_at_k(rel: list[bool], k: int) -> float:
    return sum(rel[:k]) / k


def hit_at_k(rel: list[bool], k: int) -> float:
    return 1.0 if any(rel[:k]) else 0.0


def reciprocal_rank(rel: list[bool]) -> float:
    for rank, is_relevant in enumerate(rel, start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


def score_run(rel: list[bool]) -> dict[str, float]:
    scores = {f"P@{k}": precision_at_k(rel, k) for k in PRECISION_KS}
    scores[f"Hit@{HIT_K}"] = hit_at_k(rel, HIT_K)
    scores["RR"] = reciprocal_rank(rel)
    return scores


class BackendDegraded(RuntimeError):
    """The dense backend fell back to TF-IDF mid-evaluation; its numbers would be invalid."""


def run_backend(index: SearchIndex, queries: list[Query], expected_backend: str) -> dict[str, list[dict]]:
    """Returns {method: [per-query score dict]} for one vector backend."""
    runs: dict[str, list[dict]] = {m: [] for m in METHODS}
    for q in queries:
        result = retrieve(index, q.query, top_k=MAX_RANK)
        if result.vector_backend != expected_backend:
            raise BackendDegraded(f"{expected_backend} -> {result.vector_backend} on {q.query!r}")
        for method in METHODS:
            runs[method].append(score_run(relevance(getattr(result, method), q.source)))
    return runs


def mean(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def build_index(chunks, dense: bool) -> SearchIndex | None:
    keyword_conn = build_keyword_index(chunks)
    if not dense:
        return SearchIndex(chunks, keyword_conn, TfidfIndex(chunks))
    vector_index = build_vector_index(chunks)  # TF-IDF if the key is absent / embedding failed
    return SearchIndex(chunks, keyword_conn, vector_index) if isinstance(vector_index, DenseIndex) else None


def win_tie_loss(hybrid: list[dict], other: list[dict]) -> tuple[int, int, int]:
    """Per-query comparison of reciprocal rank: hybrid better / equal / worse."""
    wins = sum(h["RR"] > o["RR"] + 1e-9 for h, o in zip(hybrid, other))
    losses = sum(h["RR"] < o["RR"] - 1e-9 for h, o in zip(hybrid, other))
    return wins, len(hybrid) - wins - losses, losses


METRICS = [f"P@{k}" for k in PRECISION_KS] + [f"Hit@{HIT_K}", "RR"]
METRIC_LABELS = {"RR": "MRR"}


def fmt_table(headers: list[str], rows: list[list[str]]) -> str:
    """GitHub-flavoured markdown table (also readable as plain console text)."""
    widths = [max(len(r[i]) for r in [headers] + rows) for i in range(len(headers))]
    line = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(headers), sep] + [line(r) for r in rows])


def build_report(results: dict[str, dict[str, list[dict]]], queries: list[Query]) -> str:
    """results: {backend_label: {method: [per-query scores]}}"""
    method_names = {"keyword": "Keyword only (BM25)", "vector": "Vector only", "hybrid": "**Hybrid** (0.4 kw + 0.6 vec)"}
    out = [(f"Evaluated {len(queries)} queries over {len({q.source for q in queries})} documents "
           f"(document-level ground truth, chunk-level retrieval, top-{MAX_RANK}).\n")]

    headers = ["Method", "Vector backend"] + [METRIC_LABELS.get(m, m) for m in METRICS]
    rows = []
    for backend, runs in results.items():
        for method in METHODS:
            vec_label = backend if method != "keyword" else "n/a"
            rows.append([method_names[method], vec_label] + [f"{mean(runs[method], m):.3f}" for m in METRICS])
    out.append("### Overall\n\n" + fmt_table(headers, rows))

    lift_rows = []
    for backend, runs in results.items():
        best_single = max(("keyword", "vector"), key=lambda m: mean(runs[m], "RR"))
        base = mean(runs[best_single], "RR")
        delta = mean(runs["hybrid"], "RR") - base
        w_k, t_k, l_k = win_tie_loss(runs["hybrid"], runs["keyword"])
        w_v, t_v, l_v = win_tie_loss(runs["hybrid"], runs["vector"])
        lift_rows.append([backend, best_single, f"{delta:+.3f}", f"{delta / base:+.1%}" if base else "n/a",
                          f"{w_k}/{t_k}/{l_k}", f"{w_v}/{t_v}/{l_v}"])
    out.append("### Hybrid lift (by MRR)\n\n" + fmt_table(
        ["Vector backend", "Best single method", "MRR lift (abs)", "MRR lift (rel)",
         "W/T/L vs keyword", "W/T/L vs vector"], lift_rows))
    out.append("_W/T/L = queries where hybrid's reciprocal rank is higher / equal / lower._")

    for backend, runs in results.items():
        type_rows = []
        for qtype in ("lexical", "mixed", "paraphrase"):
            idx = [i for i, q in enumerate(queries) if q.type == qtype]
            if not idx:
                continue
            type_rows.append([qtype, str(len(idx))] + [
                f"{mean([runs[m][i] for i in idx], 'RR'):.3f}" for m in METHODS])
        out.append(f"### MRR by query type — {backend} vector backend\n\n" + fmt_table(
            ["Query type", "n", "Keyword", "Vector", "Hybrid"], type_rows))
    return "\n\n".join(out) + "\n"


def evaluate(queries: list[Query] | None = None, include_dense: bool | None = None):
    """Runs the evaluation; returns ({backend: {method: per-query scores}}, notes)."""
    queries = queries or load_queries()
    chunks = build_chunks(DOCS_DIR)
    notes: list[str] = []
    results = {"TF-IDF": run_backend(build_index(chunks, dense=False), queries, "tfidf")}

    want_dense = bool(env_str("GOOGLE_API_KEY")) if include_dense is None else include_dense
    if want_dense:
        dense_index = build_index(chunks, dense=True)
        if dense_index is None:
            notes.append(f"GOOGLE_API_KEY is set but {embedding_model()} could not be initialised; dense results skipped.")
        else:
            try:
                results[f"Dense ({dense_index.vector_index.name})"] = run_backend(
                    dense_index, queries, dense_index.vector_index.name)
            except BackendDegraded as exc:
                notes.append(f"Dense results discarded — embedding calls failed mid-run ({exc}).")
    else:
        notes.append(f"Dense ({embedding_model()}) not evaluated: set GOOGLE_API_KEY to include it.")
    return results, notes, queries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--markdown", type=Path, help="also write the report to this markdown file")
    args = parser.parse_args(argv)

    results, notes, queries = evaluate()
    report = build_report(results, queries)
    print(report.replace("**", ""))
    for note in notes:
        print(f"Note: {note}", file=sys.stderr)
    if args.markdown:
        footer = "".join(f"\n> {n}\n" for n in notes)
        args.markdown.write_text(report + footer, encoding="utf-8")
        print(f"Wrote {args.markdown}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
