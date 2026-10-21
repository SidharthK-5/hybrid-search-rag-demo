"""Guards for running this app on the public internet with someone's API quota
behind it: per-IP and global rate limits, a request-body size cap, and
handler-level query validation.

Why layered: the Pydantic schema already rejects over-long queries, but it only
runs *after* the framework has read and parsed the whole request body. The
body-size middleware stops oversized payloads before that, and `checked_query`
re-validates inside the handler so the cap still holds if a code path ever
constructs a request without going through schema validation.

Rate limits are held in process memory (slowapi + `limits`). That is correct for
the single free-tier instance this deploys to; running several replicas would
need a shared store such as Redis.
"""
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import env_int, env_str
from app.schemas import MAX_QUERY_CHARS

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 16 * 1024  # a 500-char query is ~1-3 KB of JSON even if fully \u-escaped


# --- configuration (read per call so tests and deploy-time env changes take effect) ---

def ip_limit(_key: str | None = None) -> str:
    return env_str("RATE_LIMIT", "10/minute")


def global_limit(_key: str | None = None) -> str:
    """Backstop on total traffic, whoever it comes from. Per-IP limits can't stop
    a distributed or IP-spoofing client; this caps worst-case API spend regardless."""
    return env_str("GLOBAL_RATE_LIMIT", "300/hour")


def trusted_proxy_hops() -> int:
    return max(0, env_int("TRUSTED_PROXY_HOPS", 0))


def cors_origins() -> list[str]:
    """Origins allowed to read this API cross-origin (CORS_ALLOW_ORIGINS, comma-separated)."""
    return [o.strip().rstrip("/") for o in env_str("CORS_ALLOW_ORIGINS").split(",") if o.strip()]


def configure_cors(app: FastAPI) -> None:
    """Lets the static loading page (docs/, hosted on GitHub Pages) poll /health from
    another origin to learn when a sleeping free-tier instance has woken up.

    Deliberately narrow: only the listed origins, only GET, no credentials. CORS is
    a browser rule, not an access control — the rate limits are what protect the API —
    but there is no reason to let other sites read responses either. With the variable
    unset (the default) no CORS headers are sent at all."""
    origins = cors_origins()
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET"],
                           allow_headers=[], allow_credentials=False)


# --- client identification ---

def client_ip(request: Request) -> str:
    """Best-effort real client IP.

    X-Forwarded-For can be forged by the client, and proxies *append* to it, so
    only entries added by infrastructure we trust are reliable. With
    TRUSTED_PROXY_HOPS=n we take the n-th entry from the RIGHT — the address the
    outermost trusted proxy saw — and ignore anything a client prepended. With 0
    (local dev, no proxy) the header is ignored entirely.
    """
    hops = trusted_proxy_hops()
    if hops:
        forwarded = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
        if len(forwarded) >= hops:
            return forwarded[-hops]
    return request.client.host if request.client else "unknown"


def _global_key(request: Request) -> str:
    # slowapi passes the request only to key functions that name a `request` parameter.
    return "global"


limiter = Limiter(key_func=client_ip)


def rate_limited(func):
    """Applies the per-IP limit and the global limit to an endpoint. Each limit
    uses one shared scope, so /search and /compare draw on a single budget."""
    func = limiter.shared_limit(global_limit, scope="global", key_func=_global_key)(func)
    return limiter.shared_limit(ip_limit, scope="per-ip")(func)


def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    # Logging the raw header lets the operator confirm TRUSTED_PROXY_HOPS is right.
    logger.warning("rate limit hit: key=%s limit=%s x-forwarded-for=%r",
                   client_ip(request), exc.detail, request.headers.get("x-forwarded-for"))
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded — this public demo allows a small number "
                           "of searches per minute. Please wait a moment and try again."},
        headers={"Retry-After": "60"},
    )


# --- input limits ---

def checked_query(raw: str) -> str:
    """Normalizes and validates a query at the API layer (independent of Pydantic)."""
    if len(raw) > MAX_QUERY_CHARS:
        raise HTTPException(status_code=422, detail=f"Query exceeds {MAX_QUERY_CHARS} characters.")
    query = " ".join("".join(ch for ch in raw if ch.isprintable() or ch.isspace()).split())
    if not query:
        raise HTTPException(status_code=422, detail="Query must contain some text.")
    return query


class BodySizeLimitMiddleware:
    """Rejects request bodies over `max_bytes` with 413 before the app parses them.

    Content-Length is checked first (cheap, covers honest clients). Chunked
    uploads carry no such header, so the body is also read incrementally and
    abandoned the moment it passes the cap. Within the cap it is buffered (a few
    KB) and replayed to the app — raising from inside `receive` instead would be
    swallowed by FastAPI's body parsing and surface as a 400.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_BODY_BYTES):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        too_large = JSONResponse(status_code=413, content={"detail": "Request body too large."})
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await too_large(scope, receive, send)
            return

        body = b""
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body += message.get("body", b"")
            if len(body) > self.max_bytes:
                await too_large(scope, receive, send)
                return
            more_body = message.get("more_body", False)

        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()  # later calls (e.g. disconnect detection) pass through

        await self.app(scope, replay, send)
