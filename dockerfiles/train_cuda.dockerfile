# CUDA-enabled training dockerfile for GPU training
# Base image with CUDA support (compatible with CUDA 12.4)
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Copenhagen

# Install Python 3.12 and build tools
RUN apt update && \
    apt install --no-install-recommends -y software-properties-common curl git && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install --no-install-recommends -y python3.12 python3.12-venv python3.12-dev build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install uv for fast, reliable dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set python3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

WORKDIR /app

# Copy dependency files first for better caching
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY LICENSE LICENSE

# Install dependencies (without project) using uv.lock
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Override with CUDA-enabled PyTorch (uv.lock has CPU version for CI)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Copy application code (changes most frequently)
COPY src/ src/
COPY configs/ configs/

# Install the project package
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

ENTRYPOINT ["uv", "run", "python", "-u", "-m", "ct_scan_mlops.train"]
CMD []
