# syntax=docker/dockerfile:1

# Single stage on purpose: every dependency ships as a prebuilt wheel, so there is
# no compiler toolchain or build cache to leave behind. A multi-stage build was
# measured and saved nothing meaningful (image size is dominated by numpy/scipy/
# scikit-learn, which the zero-key TF-IDF path needs).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

# Requirements first, so this layer is cached until the dependency lists change.
# The optional Anthropic/OpenAI SDKs are included so setting ANTHROPIC_API_KEY or
# OPENAI_API_KEY in the host's dashboard is all it takes to use them.
COPY requirements.txt requirements-optional.txt ./
RUN pip install -r requirements.txt -r requirements-optional.txt

RUN useradd --create-home --uid 1000 app
COPY --chown=app:app app ./app
COPY --chown=app:app static ./static
COPY --chown=app:app data ./data
USER app

# Render injects $PORT at runtime (default 10000); 8000 is only the local fallback.
EXPOSE 10000
# `sh -c` so ${PORT} expands at runtime; `exec` makes uvicorn PID 1 so it receives SIGTERM.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
