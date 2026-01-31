# Docker Guide for Dual Pathway Models

**Version**: 2.0
**Last Updated**: 2026-01-31

This guide covers Docker usage for the CT Scan MLOps project, with special attention to dual pathway model requirements.

---

## 🐳 Available Dockerfiles

| Dockerfile | Purpose | Use Case |
|------------|---------|----------|
| `train.dockerfile` | CPU training | CI/testing, small experiments |
| `train_cuda.dockerfile` | GPU training | Production training with CUDA |
| `api.dockerfile` | API server (standalone) | Local/custom deployments |
| `api.cloudrun.dockerfile` | API server (Cloud Run) | GCP Cloud Run deployment |
| `drift_api.dockerfile` | Drift monitoring API | Data drift detection service |

---

## 🔧 What Changed in v2.0

### All Dockerfiles Fixed

1. **Removed problematic `COPY outputs/` commands**
   - `outputs/` is generated during training, not copied into images
   - Now only creates empty output directories

2. **Added dual pathway support**
   - API dockerfiles now support feature mounting
   - Proper directory structure for features

3. **Updated module paths**
   - Monitoring modules moved to `ct_scan_mlops.monitoring.*`

4. **Created necessary directories**
   - `outputs/{checkpoints,logs,reports,profiling,sweeps}`
   - `data/processed` for features

---

## 🚀 Training with Docker

### CPU Training (for testing)

```bash
# Build
docker build -f dockerfiles/train.dockerfile -t ct-scan-train:cpu .

# Run with default config
docker run --rm \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/data:/app/data \
  ct-scan-train:cpu

# Run with custom config
docker run --rm \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/data:/app/data \
  ct-scan-train:cpu \
  model=dual_pathway_top_features train.max_epochs=10
```

### GPU Training (for production)

```bash
# Build
docker build -f dockerfiles/train_cuda.dockerfile -t ct-scan-train:cuda .

# Run with GPU
docker run --rm --gpus all \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/data:/app/data \
  ct-scan-train:cuda

# Run dual pathway training (ensure features are extracted first)
docker run --rm --gpus all \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/data:/app/data \
  ct-scan-train:cuda \
  model=dual_pathway_top_features
```

**Prerequisites for dual pathway training**:
1. Extract features locally: `invoke extract-features --features top_features`
2. Mount `data/processed` directory containing `features_top_features.pkl`

---

## 🌐 API Deployment with Docker

### Local API Deployment

#### Single Pathway Models (CNN, ResNet18)

```bash
# Build
docker build -f dockerfiles/api.dockerfile -t ct-scan-api:latest .

# Run (mount model checkpoint)
docker run --rm -p 8000:8000 \
  -v $(pwd)/outputs/checkpoints:/app/outputs/checkpoints \
  -e MODEL_PATH=/app/outputs/checkpoints/best_model.ckpt \
  ct-scan-api:latest
```

#### Dual Pathway Models

```bash
# Build
docker build -f dockerfiles/api.dockerfile -t ct-scan-api:latest .

# Run (mount model + features)
docker run --rm -p 8000:8000 \
  -v $(pwd)/outputs/checkpoints:/app/outputs/checkpoints \
  -v $(pwd)/data/processed:/app/data/processed \
  -e MODEL_PATH=/app/outputs/checkpoints/dual_pathway_model.ckpt \
  -e FEATURES_PATH=/app/data/processed/features_top_features.pkl \
  ct-scan-api:latest
```

**Test the API**:
```bash
curl http://localhost:8000/health

# Or use the provided test script
bash test_api.sh
```

### Cloud Run Deployment

#### Build and Push to GCR

```bash
# Set variables
PROJECT_ID="your-gcp-project"
IMAGE_NAME="gcr.io/${PROJECT_ID}/ct-scan-api:v2.0"

# Build for Cloud Run
docker build -f dockerfiles/api.cloudrun.dockerfile -t ${IMAGE_NAME} .

# Push to Google Container Registry
docker push ${IMAGE_NAME}
```

#### Deploy to Cloud Run (Single Pathway)

```bash
gcloud run deploy ct-scan-api \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars MODEL_PATH=gs://your-bucket/models/best_model.ckpt
```

#### Deploy to Cloud Run (Dual Pathway)

```bash
# First, upload features to GCS
gsutil cp data/processed/features_top_features.pkl gs://your-bucket/features/

# Deploy with both model and features
gcloud run deploy ct-scan-api \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars MODEL_PATH=gs://your-bucket/models/dual_pathway_model.ckpt,FEATURES_PATH=gs://your-bucket/features/features_top_features.pkl
```

### Cloud Build Deployment

Use the provided `cloudbuild-api.yaml`:

```bash
# Submit build
gcloud builds submit \
  --config cloudbuild-api.yaml \
  --substitutions=_IMAGE_NAME=gcr.io/${PROJECT_ID}/ct-scan-api:v2.0
```

---

## 📊 Drift Monitoring API

```bash
# Build
docker build -f dockerfiles/drift_api.dockerfile -t ct-scan-drift:latest .

# Run
docker run --rm -p 8080:8080 \
  -v $(pwd)/data/drift:/app/data/drift \
  ct-scan-drift:latest
```

**Note**: The drift API uses the updated module path `ct_scan_mlops.monitoring.drift_api`.

---

## 🔍 Volume Mounting Best Practices

### Directory Structure for Mounting

```
# Host structure
your-project/
├── data/
│   ├── processed/
│   │   ├── features_top_features.pkl  # For dual pathway
│   │   └── features_default.pkl       # For all features
│   └── raw/                            # Original data
├── outputs/
│   ├── checkpoints/                    # Model checkpoints
│   ├── logs/                           # Training logs
│   ├── reports/                        # Analysis reports
│   └── sweeps/                         # Sweep results
└── configs/                            # Hydra configs

# Mount points in container
/app/data/processed     → Host: data/processed
/app/outputs            → Host: outputs
/app/configs            → Host: configs (if overriding)
```

### Required Mounts by Use Case

| Use Case | Required Mounts |
|----------|----------------|
| Train single pathway | `data/raw`, `outputs` |
| Train dual pathway | `data/raw`, `data/processed`, `outputs` |
| API single pathway | `outputs/checkpoints` |
| API dual pathway | `outputs/checkpoints`, `data/processed` |
| Drift monitoring | `data/drift` |

---

## 🚨 Common Issues & Solutions

### Issue 1: "outputs/ not found" during Docker build

**Error**:
```
ERROR: failed to compute cache key: "/outputs": not found
```

**Cause**: Dockerfile tried to COPY outputs/ which doesn't exist and is in .dockerignore.

**Solution**: Already fixed in v2.0. Update to latest Dockerfiles.

### Issue 2: Dual pathway model fails with "features not found"

**Error**:
```
FileNotFoundError: features_top_features.pkl not found
```

**Cause**: Features not mounted or wrong path.

**Solution**:
```bash
# Extract features locally first
invoke extract-features --features top_features

# Mount the data/processed directory
docker run --rm -p 8000:8000 \
  -v $(pwd)/data/processed:/app/data/processed \
  -v $(pwd)/outputs/checkpoints:/app/outputs/checkpoints \
  ct-scan-api:latest
```

### Issue 3: GPU not available in container

**Error**:
```
RuntimeError: No CUDA GPUs are available
```

**Cause**: Not using `--gpus` flag or nvidia-docker not configured.

**Solution**:
```bash
# Ensure nvidia-docker is installed
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Use --gpus all flag
docker run --rm --gpus all \
  -v $(pwd)/outputs:/app/outputs \
  ct-scan-train:cuda
```

### Issue 4: Drift API fails with import error

**Error**:
```
ModuleNotFoundError: No module named 'ct_scan_mlops.drift_api'
```

**Cause**: Using old module path (drift_api moved to monitoring/).

**Solution**: Already fixed in v2.0. Use updated `drift_api.dockerfile`:
```bash
docker build -f dockerfiles/drift_api.dockerfile -t ct-scan-drift:latest .
```

### Issue 5: Permission denied writing to outputs

**Error**:
```
PermissionError: [Errno 13] Permission denied: '/app/outputs/checkpoints/model.ckpt'
```

**Cause**: Container user doesn't have write permissions to mounted volume.

**Solution**:
```bash
# Option 1: Fix host permissions
chmod -R 777 outputs/

# Option 2: Run container as current user
docker run --rm --user $(id -u):$(id -g) \
  -v $(pwd)/outputs:/app/outputs \
  ct-scan-train:cpu
```

---

## 🔐 Environment Variables Reference

### Training Containers

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHONUNBUFFERED` | `1` | Disable Python output buffering |
| `CUDA_VISIBLE_DEVICES` | All | GPU selection (e.g., `0,1`) |

### API Containers

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/app/outputs/checkpoints/model.pt` | Path to model checkpoint |
| `CONFIG_PATH` | `/app/configs/config.yaml` | Path to Hydra config |
| `FEATURES_PATH` | `/app/data/processed/features_top_features.pkl` | Path to radiomics features |
| `PORT` | `8000` (api), `8080` (cloudrun) | API port |

### Cloud Run Specific

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | Port (set by Cloud Run) |
| `AIP_HTTP_PORT` | - | Vertex AI port override |

---

## 📈 Performance Considerations

### Build Time Optimization

1. **Use BuildKit cache mounts**:
   ```bash
   DOCKER_BUILDKIT=1 docker build -f dockerfiles/train_cuda.dockerfile .
   ```

2. **Layer caching**: Dependency layers are cached separately from code layers.

3. **Multi-stage builds**: API dockerfiles use multi-stage builds to minimize image size.

### Runtime Optimization

1. **GPU memory**: Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` if OOM occurs.

2. **CPU inference**: API dockerfiles use CPU-only PyTorch (smaller, faster for inference).

3. **Volume mounts**: Use bind mounts for development, named volumes for production.

---

## 🔄 Migration from v1.x Dockerfiles

### What Changed

| v1.x | v2.0 | Reason |
|------|------|--------|
| `COPY outputs/` | `RUN mkdir -p outputs/` | outputs/ doesn't exist at build time |
| API expects models baked in | API expects models mounted | More flexible deployment |
| No feature support | Feature mounting for dual pathway | Support new architecture |
| Old monitoring paths | `ct_scan_mlops.monitoring.*` | Module reorganization |

### Migration Steps

1. **Rebuild all images**:
   ```bash
   docker build -f dockerfiles/train_cuda.dockerfile -t ct-scan-train:cuda .
   docker build -f dockerfiles/api.cloudrun.dockerfile -t ct-scan-api:cloudrun .
   ```

2. **Update deployment scripts** to mount features for dual pathway:
   ```bash
   # Old (v1.x)
   docker run -v $(pwd)/outputs:/app/outputs ct-scan-api

   # New (v2.0) - dual pathway
   docker run \
     -v $(pwd)/outputs:/app/outputs \
     -v $(pwd)/data/processed:/app/data/processed \
     ct-scan-api
   ```

3. **Update CI/CD pipelines** to use new Dockerfiles.

---

## 🧪 Testing Docker Images

### Test Training Image

```bash
# Build
docker build -f dockerfiles/train.dockerfile -t ct-scan-train:test .

# Test with small dataset
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  ct-scan-train:test \
  train.max_epochs=1 data.batch_size=2
```

### Test API Image

```bash
# Build
docker build -f dockerfiles/api.dockerfile -t ct-scan-api:test .

# Start API
docker run --rm -d \
  --name ct-scan-api-test \
  -p 8000:8000 \
  -v $(pwd)/outputs/checkpoints:/app/outputs/checkpoints \
  ct-scan-api:test

# Test health endpoint
curl http://localhost:8000/health

# Stop container
docker stop ct-scan-api-test
```

---

## 📚 Additional Resources

- **Docker Documentation**: https://docs.docker.com/
- **Cloud Run Documentation**: https://cloud.google.com/run/docs
- **NVIDIA Docker**: https://github.com/NVIDIA/nvidia-docker
- **Project Migration Guide**: [MIGRATION_DUAL_PATHWAY.md](MIGRATION_DUAL_PATHWAY.md)
- **API Testing**: `test_api.sh` script in project root

---

## 🆘 Getting Help

**Docker Issues**:
- Check Docker logs: `docker logs <container_id>`
- Inspect image: `docker inspect <image_name>`
- Debug interactively: `docker run -it --entrypoint /bin/bash <image_name>`

**Build Issues**:
- Enable BuildKit: `export DOCKER_BUILDKIT=1`
- Clear cache: `docker builder prune`
- Check .dockerignore: Ensure necessary files aren't excluded

**Runtime Issues**:
- Check mounted volumes: `docker inspect <container_id> | grep Mounts`
- Verify environment variables: `docker exec <container_id> env`
- Test inside container: `docker exec -it <container_id> /bin/bash`

---

**Last Updated**: 2026-01-31
**Maintainer**: CT Scan MLOps Team
**Status**: ✅ All Dockerfiles Updated for v2.0
