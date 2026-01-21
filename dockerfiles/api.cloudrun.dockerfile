# API Dockerfile for Cloud Run deployment
# Multi-stage build for smaller final image

############################
# Builder stage
############################
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Copy dependency files first for better caching
COPY uv.lock pyproject.toml README.md LICENSE ./

# Install dependencies into a virtual environment (without project code yet)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Force CPU-only PyTorch (Cloud Run has no GPU)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Copy source code and configs
COPY src/ src/
COPY configs/ configs/

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen


############################
# Runtime stage
############################
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

# Environment variables for Cloud Run & Python
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=20

# Ensure certificates are up to date (prevents SSL errors)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && update-ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Copy virtual environment, source code, and configs from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/configs /app/configs

# Activate virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Default paths (model & config resolved at runtime)
ENV CONFIG_PATH="/app/configs/config.yaml"
ENV MODEL_PATH="/gcs/models/model.pt"

# Cloud Run listens on port 8080
EXPOSE 8080

# IMPORTANT:
# - no shell wrapper
# - bind to 0.0.0.0
# - listen on Cloud Run port
CMD ["uvicorn", "ct_scan_mlops.api:app", "--host", "0.0.0.0", "--port", "8080"]
