"""Pydantic request/response models for the search API."""
from pydantic import BaseModel, Field

# Single source of truth for the cap: the schema below declares it for OpenAPI
# docs and early rejection, and app/safety.py re-enforces it in the handlers.
MAX_QUERY_CHARS = 500


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS)
    top_k: int = Field(default=5, ge=1, le=20)
    generate_answer: bool = Field(default=True)


class CompareRequest(BaseModel):
    """Same query as /search, but returns all three rankings and never calls an LLM."""
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchHit(BaseModel):
    chunk_id: int
    source: str
    text: str
    keyword_score: float
    vector_score: float
    combined_score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]
    answer: str | None = None
    answer_mode: str  # "generated" | "extractive"
    vector_backend: str  # "gemini-embedding-2" | "tfidf" | "tfidf (fallback)"


class CompareResponse(BaseModel):
    query: str
    keyword: list[SearchHit]
    vector: list[SearchHit]
    hybrid: list[SearchHit]
    vector_backend: str
