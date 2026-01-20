# CPU-only training Dockerfile (Vertex/Cloud Build friendly)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# --- Copy dependency metadata first (better layer caching) ---
COPY uv.lock pyproject.toml README.md LICENSE ./

# --- Install deps (without installing the project yet) ---
RUN uv sync --frozen --no-install-project

# --- Force CPU PyTorch (overrides any CUDA-pinned torch from the lock) ---
RUN uv pip install --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio

# --- DVC support (so the container can pull data from your GCS DVC remote) ---
RUN uv pip install "dvc[gcs]"

# Copy DVC metadata (required for dvc pull)
COPY .dvc/ .dvc/
# Copy data and artifacts directories (including any .dvc files)
RUN mkdir -p data artifacts
COPY data/ data/
COPY artifacts/ artifacts/

# --- Copy code/configs last (changes frequently) ---
COPY src/ src/
COPY configs/ configs/

# --- Install the project package (and any remaining deps) ---
RUN uv sync --frozen

# --- Pull data then train ---
# This guarantees training has the data even if the python module doesn't pull.
ENTRYPOINT ["bash", "-c", "dvc pull -v && exec uv run --frozen python -u -m ct_scan_mlops.train \"$@\"", "--"]
CMD []
