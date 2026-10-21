"""FastAPI app: builds the hybrid search index on startup and exposes /search
(ranked passages + optional answer) and /compare (keyword vs vector vs hybrid
side by side). Serves the minimal static frontend at /.

Run with:  uvicorn app.main:app --reload
Then open: http://127.0.0.1:8000
"""
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded

from app.generate import ExtractiveFallback, get_provider
from app.ingest import build_chunks
from app.safety import (
    BodySizeLimitMiddleware,
    checked_query,
    configure_cors,
    limiter,
    rate_limit_handler,
    rate_limited,
)
from app.schemas import (
    CompareRequest,
    CompareResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from app.search import build_search_index, hybrid_search, retrieve

logger = logging.getLogger("uvicorn.error")

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "sample_docs"

state: dict = {}

# After an LLM failure, skip the provider for this long and answer extractively
# right away, so a provider outage costs one slow request, not one per visitor.
LLM_COOLDOWN_SECONDS = 60.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    chunks = build_chunks(DOCS_DIR)
    state["index"] = build_search_index(chunks)
    state["llm"] = get_provider()
    logger.info("Indexed %d chunks from %s/ — vector backend: %s — LLM mode: %s",
                len(chunks), DOCS_DIR.name, state["index"].vector_index.name, state["llm"].name)
    yield
    state.clear()


app = FastAPI(title="Hybrid Search RAG Demo", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
app.add_middleware(BodySizeLimitMiddleware)
configure_cors(app)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health():
    ready = "index" in state
    return {
        "status": "ok",
        "chunks_indexed": len(state["index"].chunks) if ready else 0,
        "vector_backend": state["index"].vector_index.name if ready else "unknown",
        "llm_mode": state["llm"].name if "llm" in state else "unknown",
    }


@app.post("/search", response_model=SearchResponse)
@rate_limited
def search(request: Request, req: SearchRequest) -> SearchResponse:
    # `request` is required by the rate limiter (it keys on the client address).
    query = checked_query(req.query)
    hits, backend = hybrid_search(state["index"], query, top_k=req.top_k)

    answer = None
    answer_mode = "extractive"
    if req.generate_answer:
        llm = state["llm"]
        if time.monotonic() >= state.get("llm_paused_until", 0.0):
            try:
                answer = llm.generate(query, hits)
                answer_mode = llm.name
            except Exception:
                # A provider outage or exhausted quota shouldn't take search down with it.
                logger.exception("LLM generation failed; extractive fallback for the next %ds",
                                 LLM_COOLDOWN_SECONDS)
                state["llm_paused_until"] = time.monotonic() + LLM_COOLDOWN_SECONDS
        if answer is None:
            # Either no LLM is configured, or it just failed / is paused after failing.
            notice = (ExtractiveFallback.NOT_CONFIGURED if isinstance(llm, ExtractiveFallback)
                      else ExtractiveFallback.UNAVAILABLE)
            answer = ExtractiveFallback(notice).generate(query, hits)

    return SearchResponse(
        query=query,
        results=[SearchHit(**h) for h in hits],
        answer=answer,
        answer_mode=answer_mode,
        vector_backend=backend,
    )


@app.post("/compare", response_model=CompareResponse)
@rate_limited
def compare(request: Request, req: CompareRequest) -> CompareResponse:
    query = checked_query(req.query)
    result = retrieve(state["index"], query, top_k=req.top_k)
    return CompareResponse(
        query=query,
        keyword=[SearchHit(**h) for h in result.keyword],
        vector=[SearchHit(**h) for h in result.vector],
        hybrid=[SearchHit(**h) for h in result.hybrid],
        vector_backend=result.vector_backend,
    )
