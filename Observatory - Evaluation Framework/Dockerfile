# Observatory — Evaluation Framework
#
# Python 3.12 is REQUIRED, not merely preferred: ragas 0.4.3 needs the langchain
# 0.3.x line, whose pydantic-v1 compatibility shim breaks on Python >= 3.14.
# See the pin rationale at the top of requirements.txt before bumping this.

# ── Stage 1: build wheels ─────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential is needed to compile the handful of dependencies that ship
# no manylinux wheel. It stays in this stage so it never reaches the runtime
# image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Model weights live in the image, not on the EFS mount: they are immutable
    # and identical for every task, so baking them turns a ~67MB cold-start
    # download into zero. Extra models a user selects at runtime land here too,
    # on the container's ephemeral layer.
    FASTEMBED_CACHE_PATH=/opt/models \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

# libgomp1 is required by onnxruntime, which fastembed uses for both the
# embedding models and the cross-encoder rerankers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels requirements.txt

# Pre-download the default embedding model and reranker. Any other model the UI
# offers is fetched on first use.
RUN mkdir -p /opt/models && python -c "\
from fastembed import TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); \
TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2'); \
print('baked default models into /opt/models')"

# Run unprivileged. /data is the EFS mount point; it is chowned here so the
# container can write uploads and the Chroma index without running as root.
RUN useradd --create-home --uid 10001 observatory \
    && mkdir -p /data/uploads /data/index \
    && chown -R observatory:observatory /data /opt/models

WORKDIR /app
COPY --chown=observatory:observatory backend/ ./backend/
COPY --chown=observatory:observatory frontend/ ./frontend/

USER observatory

ENV UPLOAD_DIR=/data/uploads \
    RAG_INDEX_CACHE_DIR=/data/index

EXPOSE 8000

# Liveness only — deliberately does not touch the database, so an RDS blip
# cannot cascade into every task being killed. /readyz reports dependencies.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

# ONE worker per task, intentionally. The retrieval index, the BM25 index and
# the reranker are cached in-process (see _INDEX_CACHE in rag_evaluator.py);
# extra workers would each hold a private copy, multiplying memory and losing
# cache hits. Scale with ECS task count, not with workers -- and read the
# concurrency note in DEPLOY.md before raising the task count above one.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
