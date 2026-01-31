# API dockerfile for serving predictions
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

# Copy virtual environment and source from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
# Copy configs
COPY --from=builder /app/configs/config_production.yaml /app/configs/config.yaml

# Set default config and model paths (can be overridden at runtime)
ENV CONFIG_PATH="/app/configs/config.yaml"
ENV MODEL_PATH="/app/outputs/checkpoints/model.pt"

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# NOTE: Models are no longer baked into this image.
# Mount model weights at runtime and optionally override config/model paths:
#   docker run --rm -p 8000:8000 \
#     -v /host/path/to/outputs/checkpoints:/app/outputs/checkpoints \
#     -v /host/path/to/configs:/app/configs \
#     your-image-name
# Ensure /host/path/to/outputs/checkpoints contains model.pt (or set MODEL_PATH accordingly).

EXPOSE 8000

ENTRYPOINT ["uvicorn", "ct_scan_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
CMD []
