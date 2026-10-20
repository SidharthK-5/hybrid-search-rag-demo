"""Loads sample documents, splits them into chunks, and builds the keyword
(SQLite FTS5) index that the hybrid search layer queries. The vector index lives
in app/vector.py so it can swap between dense embeddings and TF-IDF.

This mirrors the ingestion shape of a production RAG pipeline (load -> chunk ->
index) while keeping the default path local and free, so the repo runs with no
API keys and no external services.
"""
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

CHUNK_SIZE = 700      # characters per chunk
CHUNK_OVERLAP = 100   # characters of overlap between consecutive chunks


@dataclass
class Chunk:
    chunk_id: int
    source: str
    text: str


def load_documents(docs_dir: Path) -> list[tuple[str, str]]:
    """Returns a list of (filename, full_text) for every .txt file in docs_dir."""
    docs = []
    for path in sorted(docs_dir.glob("*.txt")):
        docs.append((path.stem.replace("_", " ").title(), path.read_text(encoding="utf-8")))
    return docs


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Splits text into chunks of at most `size` characters, packing whole paragraphs
    where possible. Only a single paragraph longer than `size` is hard-split, and
    only those hard-split pieces overlap (by `overlap` characters) so a sentence
    cut at the boundary still appears intact in one chunk."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= size:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)

    # Fall back to hard-splitting any paragraph that's still longer than `size`
    final_chunks: list[str] = []
    for c in chunks:
        if len(c) <= size:
            final_chunks.append(c)
        else:
            start = 0
            while True:
                final_chunks.append(c[start:start + size])
                if start + size >= len(c):
                    break  # this piece reached the end; another would be pure overlap
                start += size - overlap
    return final_chunks


def build_chunks(docs_dir: Path) -> list[Chunk]:
    """Loads every document in docs_dir and returns its chunks with sequential ids."""
    chunks: list[Chunk] = []
    chunk_id = 0
    for source, text in load_documents(docs_dir):
        for piece in chunk_text(text):
            chunks.append(Chunk(chunk_id=chunk_id, source=source, text=piece))
            chunk_id += 1
    return chunks


def build_keyword_index(chunks: list[Chunk]) -> sqlite3.Connection:
    """Builds an in-memory SQLite FTS5 full-text index over the chunks."""
    # check_same_thread=False: FastAPI runs sync endpoints in a worker thread
    # pool, while this connection is created once at startup. Safe here since
    # the index is read-only after build_keyword_index returns.
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute(
        "CREATE VIRTUAL TABLE chunks USING fts5(chunk_id UNINDEXED, source UNINDEXED, text)"
    )
    conn.executemany(
        "INSERT INTO chunks (chunk_id, source, text) VALUES (?, ?, ?)",
        [(c.chunk_id, c.source, c.text) for c in chunks],
    )
    conn.commit()
    return conn
