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

# Copy source code
COPY src/ src/

# Install the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Runtime stage - minimal image
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

# Copy virtual environment and source from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Cloud Run will set PORT=8080 by default
# Expose port 8080 for Cloud Run
EXPOSE 8080

# NOTE: Models are no longer baked into this image.
# They must be mounted at runtime, e.g. via Cloud Run volumes or GCS FUSE
# Ensure that the mounted path contains:
#   - .hydra/config.yaml
#   - model.pt (or set MODEL_FILENAME env var)

# Use shell form to allow environment variable substitution
# Cloud Run will provide PORT environment variable (default: 8080)
CMD uvicorn ct_scan_mlops.api:app --host 0.0.0.0 --port ${PORT:-8080}
