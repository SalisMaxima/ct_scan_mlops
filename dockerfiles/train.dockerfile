# CPU-only training dockerfile (for CI/testing)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Copy lock + metadata first for caching
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml




# Install deps except the project itself
RUN uv sync --frozen --no-install-project --no-cache

# Force CPU PyTorch (overrides the CUDA-pinned torch from the lock)
RUN uv pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio

# Copy code/configs
COPY src/ src/
COPY configs/ configs/

# Copy README.md 
COPY README.md README.md
COPY LICENSE LICENSE

# Install project package (and any remaining deps)
RUN uv sync --frozen --no-cache

ENTRYPOINT ["uv", "run", "python", "-m", "ct_scan_mlops.train"]
CMD []
