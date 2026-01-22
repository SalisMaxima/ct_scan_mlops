# Drift API dockerfile for Cloud Run deployment
# Multi-stage build to match the repo's uv + .venv pattern
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Copy dependency files first for better caching
COPY uv.lock pyproject.toml README.md LICENSE ./

# Install dependencies (without project) with BuildKit cache
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Keep this override to avoid CUDA wheels / ensure small CPU torch,
# because your project likely depends on torch via pyproject/lock.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Copy source code
COPY src/ src/
COPY data/drift/reference.csv data/drift/reference.csv
COPY data/drift/current.csv data/drift/current.csv



# (Optional) If you keep reference.csv in repo and want it baked into the image:
# COPY data/drift/reference.csv data/drift/reference.csv

# Install the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Runtime stage
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

ENV PORT=8080
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=20
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONUNBUFFERED=1

# SSL certs (same as your api.cloudrun.dockerfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && update-ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Copy virtual environment + code
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/data /app/data

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8080

CMD ["bash", "-c", "uvicorn ct_scan_mlops.drift_api:app --host 0.0.0.0 --port ${PORT}"]
