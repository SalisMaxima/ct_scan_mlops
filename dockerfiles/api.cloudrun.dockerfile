FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PORT=8080
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=20
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Ensure certs are up to date (fixes many SSL errors)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
  && update-ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Copy app code/config
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY configs/ configs/

# Install deps; pull torch/torchvision from the CPU index
RUN python -m pip install -U pip setuptools wheel \
  && python -m pip install \
      fastapi \
      "uvicorn[standard]" \
      pillow \
      omegaconf \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      torch torchvision

# Install your package (editable)
RUN python -m pip install -e .

ENV CONFIG_PATH="/app/configs/config.yaml"
ENV MODEL_PATH="/app/models/model.pt"

ENV PORT=8080
EXPOSE 8080
CMD ["bash","-lc","uvicorn ct_scan_mlops.api:app --host 0.0.0.0 --port ${PORT}"]
