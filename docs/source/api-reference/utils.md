# Utilities Module

Shared utility functions for the CT Scan MLOps project.

## Overview

The utilities module provides common functions used across the project.

## Functions

::: ct_scan_mlops.utils.get_device
    options:
      heading_level: 3

## Usage Examples

### Get Best Available Device

```python
from ct_scan_mlops.utils import get_device

device = get_device()
print(f"Using device: {device}")
# Using device: cuda  (if GPU available)
# Using device: mps   (if Apple Silicon)
# Using device: cpu   (fallback)
```

### Use with Models

```python
from ct_scan_mlops.utils import get_device
from ct_scan_mlops.model import build_model

device = get_device()
model = build_model(cfg).to(device)
```

## Device Priority

The `get_device()` function checks for available devices in this order:

1. **CUDA** - NVIDIA GPU with CUDA support
2. **MPS** - Apple Silicon GPU (Metal Performance Shaders)
3. **CPU** - Fallback to CPU computation

This ensures the best available hardware is used automatically without manual configuration.
