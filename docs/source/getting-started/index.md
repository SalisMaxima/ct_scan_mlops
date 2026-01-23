# Getting Started

This section covers everything you need to set up and start using the CT Scan MLOps project.

## Overview

The CT Scan MLOps project is designed to be easy to set up and use. Follow these guides to get started:

1. **[Installation](installation.md)** - Set up your development environment
2. **[Quick Start](quickstart.md)** - Run your first training job

## Prerequisites

Before you begin, ensure you have:

| Tool | Purpose | Installation |
|------|---------|--------------|
| **Python 3.12** | Runtime | [python.org](https://www.python.org/downloads/) |
| **uv** | Package manager | See [installation guide](installation.md) |
| **Git** | Version control | [git-scm.com](https://git-scm.com/downloads) |
| **GCP Access** | Data storage | Contact project admin |

## TL;DR

For experienced users, here's the minimal setup:

```bash
# Clone and setup
git clone https://github.com/SalisMaxima/ct_scan_mlops.git
cd ct_scan_mlops
uv venv && source .venv/bin/activate
uv sync --all-groups

# Authenticate and get data
gcloud auth application-default login
dvc pull

# Train
invoke train
```

## Next Steps

- [Installation Guide](installation.md) - Detailed setup instructions
- [Quick Start Guide](quickstart.md) - Essential commands
- [Training Guide](../user-guide/training.md) - Full training documentation
