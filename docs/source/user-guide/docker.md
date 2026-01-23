# Docker

Guide to using Docker for training and deployment.

## Available Dockerfiles

| Dockerfile | Purpose |
|------------|---------|
| `dockerfiles/train.dockerfile` | CPU training environment |
| `dockerfiles/train_cuda.dockerfile` | GPU training environment |
| `dockerfiles/api.dockerfile` | FastAPI inference server |

## Building Images

### CPU Training Image

```bash
invoke docker-build
```

### GPU Training Image

```bash
invoke docker-build-cuda
```

### API Image

```bash
docker build -f dockerfiles/api.dockerfile -t ct-scan-api .
```

## Running Training in Docker

### CPU Training

```bash
invoke docker-train
```

### GPU Training

Requires NVIDIA Container Toolkit:

```bash
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/outputs:/app/outputs \
    ct-scan-train-cuda \
    invoke train
```

## Running the API

### Build and Run

```bash
# Build
docker build -f dockerfiles/api.dockerfile -t ct-scan-api .

# Run
docker run -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/configs:/app/configs \
    ct-scan-api
```

### API Endpoints

Once running, the API is available at `http://localhost:8000`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Classify a CT scan image |
| `/feedback` | POST | Submit prediction feedback |
| `/metrics` | GET | Prometheus metrics |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
    -F "file=@path/to/ct_scan.png"
```

Response:

```json
{
    "pred_index": 0,
    "pred_class": "adenocarcinoma"
}
```

## Environment Variables

Configure the API container with environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `CONFIG_PATH` | Path to config file | `configs/config.yaml` |
| `MODEL_PATH` | Path to model weights | `models/model.pt` |
| `LOAD_MODEL` | Load model on startup | `1` |
| `FEEDBACK_DIR` | Directory for feedback images | `feedback` |

Example:

```bash
docker run -p 8000:8000 \
    -e MODEL_PATH=/app/models/custom_model.pt \
    -v $(pwd)/models:/app/models \
    ct-scan-api
```

## Docker Compose

For development, you can use Docker Compose:

```yaml
version: '3.8'
services:
  api:
    build:
      context: .
      dockerfile: dockerfiles/api.dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./configs:/app/configs
    environment:
      - MODEL_PATH=/app/models/model.pt
```

Run with:

```bash
docker-compose up
```

## Monitoring

The API exposes Prometheus metrics at `/metrics`:

- `prediction_error`: Counter of prediction errors
- `system_cpu_percent`: System CPU utilization
- `system_memory_percent`: System memory utilization
- `process_rss_bytes`: Process memory usage
- HTTP request metrics (latency, count, etc.)
