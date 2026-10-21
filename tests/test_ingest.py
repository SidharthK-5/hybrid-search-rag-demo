"""Chunking and index-building edge cases."""
import pytest

from app.ingest import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    build_chunks,
    build_keyword_index,
    chunk_text,
    display_name,
    load_documents,
)


def test_empty_and_whitespace_text_produce_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n \t \n") == []


def test_short_text_is_a_single_chunk():
    assert chunk_text("Just one short paragraph.") == ["Just one short paragraph."]


def test_paragraphs_are_packed_without_exceeding_size():
    paras = [("word " * 20).strip() for _ in range(10)]  # ~99 chars each
    chunks = chunk_text("\n\n".join(paras), size=250, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 250 for c in chunks)
    # Nothing is lost or reordered when no paragraph needs hard-splitting.
    assert "\n".join(chunks).split() == " ".join(paras).split()


def test_oversized_paragraph_is_hard_split_with_overlap():
    text = "".join(chr(97 + i % 26) for i in range(1000))  # one 1000-char paragraph
    chunks = chunk_text(text, size=300, overlap=50)
    assert all(len(c) <= 300 for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev[-50:] == nxt[:50]
    assert chunks[0].startswith(text[:300])
    assert chunks[-1].endswith(text[-10:])


def test_hard_split_does_not_emit_a_redundant_tail_chunk():
    # 1300 chars, size 700, overlap 100 -> [0:700] and [600:1300]; a third piece
    # starting at 1200 would be entirely contained in the second.
    chunks = chunk_text("x" * 1300, size=700, overlap=100)
    assert [len(c) for c in chunks] == [700, 700]


def test_text_exactly_at_size_is_not_split():
    assert chunk_text("y" * 700, size=700, overlap=100) == ["y" * 700]


def test_overlap_not_smaller_than_size_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("anything", size=100, overlap=100)


def test_windows_line_endings_still_split_into_paragraphs():
    text = "First paragraph.\r\n\r\nSecond paragraph."
    assert chunk_text(text, size=20, overlap=5) == ["First paragraph.", "Second paragraph."]


def test_load_documents_reads_only_txt_and_titles_the_source(tmp_path):
    (tmp_path / "remote_work_policy.txt").write_text("Body.", encoding="utf-8")
    (tmp_path / "notes.md").write_text("ignored", encoding="utf-8")
    assert load_documents(tmp_path) == [("Remote Work Policy", "Body.")]


def test_build_chunks_assigns_sequential_ids_across_documents(tmp_path):
    (tmp_path / "a_doc.txt").write_text("Alpha one.\n\nAlpha two.", encoding="utf-8")
    (tmp_path / "b_doc.txt").write_text("Beta.", encoding="utf-8")
    chunks = build_chunks(tmp_path)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    assert [c.source for c in chunks][-1] == "B Doc"


def test_build_chunks_on_empty_directory_is_empty(tmp_path):
    assert build_chunks(tmp_path) == []


def test_sample_corpus_covers_all_eight_documents_within_chunk_size(chunks):
    assert len({c.source for c in chunks}) == 8
    assert all(0 < len(c.text) <= CHUNK_SIZE for c in chunks)
    assert CHUNK_OVERLAP < CHUNK_SIZE


def test_keyword_index_is_searchable(chunks):
    conn = build_keyword_index(chunks)
    (count,) = conn.execute("SELECT count(*) FROM chunks").fetchone()
    assert count == len(chunks)
    rows = conn.execute("SELECT source FROM chunks WHERE chunks MATCH 'RFP'").fetchall()
    assert {r[0] for r in rows} == {"Procurement Guidelines"}


@pytest.mark.parametrize("stem, expected", [
    ("remote_work_policy", "Remote Work Policy"),
    ("it_acceptable_use_policy", "IT Acceptable Use Policy"),   # regression: str.title() gave "It"
    ("hr_leave_policy", "HR Leave Policy"),
    ("public_records_request_procedure", "Public Records Request Procedure"),
    ("single", "Single"),
])
def test_display_name_keeps_acronyms_upper_case(stem, expected):
    assert display_name(stem) == expected
