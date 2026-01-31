# CUDA-enabled training dockerfile for GPU training
# 3-stage build: python-base -> builder -> runtime
# Optimized to install Python only ONCE (in python-base stage)

# === Stage 1: Python base (shared between builder and runtime) ===
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS python-base

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Copenhagen

# Install Python 3.12 runtime only (no dev packages, no build tools)
# Using APT cache mounts to speed up builds
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    apt-get install --no-install-recommends -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install --no-install-recommends -y python3.12 python3.12-venv

# Set python3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# === Stage 2: Builder (adds build tools on top of python-base) ===
FROM python-base AS builder

# Install build tools needed for compilation (python3.12-dev, gcc, etc.)
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    apt-get install --no-install-recommends -y \
        python3.12-dev build-essential gcc

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for better caching
COPY uv.lock pyproject.toml README.md LICENSE ./

# Install dependencies (without project)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Install CUDA-enabled PyTorch
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Copy application code
COPY src/ src/
COPY configs/ configs/

# Install the project package
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Install DVC with GCS support
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install "dvc[gcs]"

# Copy DVC metadata
COPY .dvc/ .dvc/
# Copy data and outputs directories (including any .dvc files)
RUN mkdir -p data outputs/checkpoints outputs/logs outputs/reports outputs/profiling
COPY data/ data/
COPY outputs/ outputs/

# === Stage 3: Runtime (inherits from python-base, no build tools) ===
FROM python-base AS runtime

# Copy uv for runtime commands
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Create directories for DVC files
RUN mkdir -p data outputs/checkpoints outputs/logs outputs/reports outputs/profiling

# Copy virtual environment and source from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/configs /app/configs
COPY --from=builder /app/.dvc /app/.dvc
COPY --from=builder /app/data/ /app/data/
COPY --from=builder /app/outputs/ /app/outputs/

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["bash", "-c", "if [ -d .git ]; then dvc pull -v; else echo 'Skipping dvc pull (no git repo)'; fi && exec uv run --frozen python -u -m ct_scan_mlops.train \"$@\"", "--"]
CMD []
