# CUDA-enabled training dockerfile for GPU training
# Multistage build to remove build tools from final image

# === Builder stage ===
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Copenhagen

# Install Python 3.12 and build tools (needed for compilation)
RUN apt-get update && \
    apt-get install --no-install-recommends -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install --no-install-recommends -y \
        python3.12 python3.12-venv python3.12-dev \
        build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set python3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

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

# === Runtime stage (no build tools) ===
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Copenhagen

# Install only Python runtime (no build tools)
RUN apt-get update && \
    apt-get install --no-install-recommends -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install --no-install-recommends -y python3.12 python3.12-venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Copy uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy virtual environment and source from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/configs /app/configs

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["uv", "run", "--frozen", "python", "-u", "-m", "ct_scan_mlops.train"]
CMD []
