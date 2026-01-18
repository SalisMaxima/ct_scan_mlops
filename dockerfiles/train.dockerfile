# CPU-only training dockerfile (for CI/testing)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Copy lock + metadata first for caching
COPY uv.lock pyproject.toml README.md LICENSE ./

# Install deps except the project itself with BuildKit cache
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Force CPU PyTorch (overrides the CUDA-pinned torch from the lock)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio

# Copy code/configs (changes frequently - put last)
COPY src/ src/
COPY configs/ configs/

# Install project package (and any remaining deps)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

ENTRYPOINT ["uv", "run", "python", "-m", "ct_scan_mlops.train"]
CMD []
