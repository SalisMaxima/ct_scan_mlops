"""Shared utility functions for the CT scan MLOps project."""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Get the best available device.

    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
