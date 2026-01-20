# API dockerfile for Cloud Run deployment
# Multi-stage build for smaller final image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Copy dependency files first for better caching
COPY uv.lock pyproject.toml README.md LICENSE ./

# Install dependencies in a virtual environment with BuildKit cache
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Override with CPU-only PyTorch (smaller, no GPU needed for API inference)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Copy source code and configs
COPY src/ src/
COPY configs/ configs/

# Install the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Runtime stage - minimal image
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

# Environment variables for better pip behavior and Cloud Run
ENV PORT=8080
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=20
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONUNBUFFERED=1

# Ensure certs are up to date (fixes many SSL errors)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && update-ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Copy virtual environment, source, and configs from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/configs /app/configs

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Set default config and model paths (can be overridden at runtime)
ENV CONFIG_PATH="/app/configs/config.yaml"
ENV MODEL_PATH="/app/models/model.pt"

# Expose port 8080 for Cloud Run
EXPOSE 8080

# NOTE: Models are no longer baked into this image.
# They must be mounted at runtime, e.g. via Cloud Run volumes or GCS FUSE
# Override CONFIG_PATH and MODEL_PATH env vars as needed for your deployment.

# Cloud Run will provide PORT environment variable (default: 8080)
CMD ["bash", "-c", "uvicorn ct_scan_mlops.api:app --host 0.0.0.0 --port ${PORT}"]
