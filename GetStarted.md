# Getting Started (For Group Members)

## Prerequisites

Make sure you have these installed:

| Tool | Installation |
|------|--------------|
| **Python 3.12** | [python.org](https://www.python.org/downloads/) |
| **uv** | `curl -LsSf https://astral.sh/uv/install.sh \| sh` (Linux/Mac) or `powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"` (Windows) |
| **Git** | [git-scm.com](https://git-scm.com/downloads) |
| **DVC** | Installed automatically with project dependencies |
| **GCP Access** | Ask Mathias for access to the GCP project |

## Step 1: Clone the Repository

```bash
git clone https://github.com/SalisMaxima/ct_scan_mlops.git
cd ct_scan_mlops
```

## Step 2: Set Up Environment

```bash
# Create virtual environment and install all dependencies
uv venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

uv sync --all-groups
```

## Step 3: Authenticate with GCP (for DVC)

```bash
# Login to Google Cloud (one-time setup)
gcloud auth login
gcloud auth application-default login
```

## Step 4: Pull the Dataset

```bash
# Download the chest CT scan dataset from GCS
dvc pull
```

This downloads ~120MB of CT scan images to `data/raw/`.

## Step 5: Verify Setup

```bash
# Check everything works
invoke python      # Should show Python 3.12
invoke test        # Run tests (may have placeholder tests initially)
```

---

## Troubleshooting

### "Module not found" errors
```bash
uv sync --all-groups  # Reinstall all dependencies
```

### DVC authentication issues
```bash
gcloud auth application-default login
```

### CUDA/GPU not detected
Make sure you have NVIDIA drivers installed and use `invoke docker-build-cuda` for GPU training.

### Windows-specific issues
- Use PowerShell or Git Bash (not CMD)
- Replace `source .venv/bin/activate` with `.venv\Scripts\activate`
