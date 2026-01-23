# Installation

Complete setup instructions for the CT Scan MLOps project.

## Prerequisites

### Python 3.12

Download and install Python 3.12 from [python.org](https://www.python.org/downloads/).

Verify installation:

```bash
python --version  # Should show Python 3.12.x
```

### uv Package Manager

Install uv, the fast Python package manager:

=== "Linux/Mac"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"

    ```powershell
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

### Git

Install Git from [git-scm.com](https://git-scm.com/downloads).

## Step 1: Clone the Repository

```bash
git clone https://github.com/SalisMaxima/ct_scan_mlops.git
cd ct_scan_mlops
```

## Step 2: Set Up Environment

Create a virtual environment and install dependencies:

```bash
# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install all dependencies
uv sync --all-groups
```

## Step 3: Authenticate with GCP

The dataset is stored in Google Cloud Storage. Authenticate to access it:

```bash
# Login to Google Cloud (one-time setup)
gcloud auth login
gcloud auth application-default login
```

!!! note "GCP Access"
    You need access to the GCP project to pull data. Contact the project admin if you don't have access.

## Step 4: Pull the Dataset

Download the chest CT scan dataset from GCS:

```bash
dvc pull
```

This downloads ~120MB of CT scan images to `data/raw/`.

## Step 5: Verify Setup

Check that everything works:

```bash
# Check Python version
invoke python

# Run tests
invoke test
```

## Troubleshooting

### "Module not found" errors

Reinstall all dependencies:

```bash
uv sync --all-groups
```

### DVC authentication issues

Re-authenticate with GCP:

```bash
gcloud auth application-default login
```

### CUDA/GPU not detected

Ensure NVIDIA drivers are installed. For GPU training in Docker:

```bash
invoke docker-build-cuda
```

### Windows-specific issues

- Use PowerShell or Git Bash (not CMD)
- Replace `source .venv/bin/activate` with `.venv\Scripts\activate`

## Next Steps

- [Quick Start Guide](quickstart.md) - Essential commands
- [Training Guide](../user-guide/training.md) - Start training models
