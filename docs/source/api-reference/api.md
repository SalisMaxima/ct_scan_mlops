# FastAPI Module

REST API for CT scan classification inference.

## Overview

The API module provides:

- **FastAPI application** for serving predictions
- **Health check** endpoint for monitoring
- **Prediction endpoint** for classification
- **Feedback endpoint** for collecting corrections
- **Prometheus metrics** for observability

## Application

The FastAPI application is initialized with a lifespan manager that loads the model on startup.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and status |
| `/predict` | POST | Classify a CT scan image |
| `/feedback` | POST | Submit prediction feedback |
| `/metrics` | GET | Prometheus metrics |

## API Reference

### Health Check

```python
GET /health

Response:
{
    "ok": true,
    "device": "cuda",
    "model_loaded": true,
    "config_path": "configs/config.yaml",
    "model_path": "models/model.pt"
}
```

### Prediction

```python
POST /predict
Content-Type: multipart/form-data
file: <image file>

Response:
{
    "pred_index": 0,
    "pred_class": "adenocarcinoma"
}
```

### Feedback

```python
POST /feedback
Content-Type: multipart/form-data
file: <image file>
predicted_class: "adenocarcinoma"
is_correct: false
correct_class: "normal"

Response:
{
    "saved_to": "feedback/normal/image-abc123.png",
    "class": "normal"
}
```

## Configuration

Configure the API with environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `CONFIG_PATH` | Path to Hydra config | `configs/config.yaml` |
| `MODEL_PATH` | Path to model weights | `models/model.pt` |
| `LOAD_MODEL` | Load model on startup (`1`/`0`) | `1` |
| `FEEDBACK_DIR` | Directory for feedback images | `feedback` |
| `DRIFT_CURRENT_PATH` | Path for drift detection CSV | `data/drift/current.csv` |

## Monitoring

### Prometheus Metrics

The API exposes Prometheus metrics at `/metrics`:

- `prediction_error` - Counter of prediction errors
- `system_cpu_percent` - System CPU utilization
- `system_memory_percent` - System memory utilization
- `process_rss_bytes` - Process memory usage
- HTTP request metrics (via prometheus-fastapi-instrumentator)

### Data Drift Tracking

Each prediction appends image statistics to a CSV file for drift detection:

- Mean, std, min, max of pixel values
- Percentiles (1st, 50th, 99th)
- Image dimensions
- Prediction class and confidence

## Usage Examples

### Run the API

```bash
# With invoke
invoke api

# Direct with uvicorn
uvicorn ct_scan_mlops.api:app --host 0.0.0.0 --port 8000

# With custom config
CONFIG_PATH=custom/config.yaml MODEL_PATH=models/best.pt uvicorn ct_scan_mlops.api:app
```

### Make Predictions

```bash
# Using curl
curl -X POST "http://localhost:8000/predict" \
    -F "file=@path/to/ct_scan.png"

# Using Python requests
import requests

with open("ct_scan.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )

print(response.json())
# {"pred_index": 0, "pred_class": "adenocarcinoma"}
```

### Submit Feedback

```bash
curl -X POST "http://localhost:8000/feedback" \
    -F "file=@ct_scan.png" \
    -F "predicted_class=adenocarcinoma" \
    -F "is_correct=false" \
    -F "correct_class=normal"
```

## Docker Deployment

```bash
# Build
docker build -f dockerfiles/api.dockerfile -t ct-scan-api .

# Run
docker run -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/configs:/app/configs \
    ct-scan-api
```
