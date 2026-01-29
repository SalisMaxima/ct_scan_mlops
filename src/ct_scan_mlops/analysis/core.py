"""Core logic for analysis: Model loading, Inference, and Data structures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from omegaconf.base import Container, ContainerMetadata
from torch.utils.data import DataLoader
from tqdm import tqdm

from ct_scan_mlops.data import CLASSES
from ct_scan_mlops.model import build_model

# Safe globals for torch.load
torch.serialization.add_safe_globals([DictConfig, Container, ContainerMetadata, Any])

# Model names that use radiomics features
DUAL_PATHWAY_MODEL_NAMES = frozenset({"dual_pathway", "dualpathway", "hybrid"})


@dataclass
class LoadedModel:
    """Container for loaded model with metadata."""

    model: nn.Module
    config: DictConfig
    uses_features: bool
    model_name: str
    checkpoint_path: Path


@dataclass
class PredictionResults:
    """Standardized results from an inference run."""

    # All arrays aligned by index
    predictions: np.ndarray  # (N,) class indices
    targets: np.ndarray  # (N,) class indices
    probabilities: np.ndarray  # (N, num_classes)
    logits: np.ndarray  # (N, num_classes)

    # Optional data (if collected)
    features: np.ndarray | None = None  # (N, feature_dim)
    images: np.ndarray | None = None  # (N, C, H, W) or flattened

    # Metadata
    sample_indices: list[int] | None = None

    @property
    def accuracy(self) -> float:
        return float((self.predictions == self.targets).mean())

    @property
    def num_samples(self) -> int:
        return len(self.targets)


def load_model_from_checkpoint(
    checkpoint_path: Path,
    cfg: DictConfig,
    device: torch.device,
) -> torch.nn.Module:
    """Load model from checkpoint file. Moved from evaluate.py to centralize logic."""
    logger.info(f"Loading model from {checkpoint_path}")

    # Build model from config
    model = build_model(cfg).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            # Lightning .ckpt format
            state_dict = checkpoint["state_dict"]
            model_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("model."):
                    model_state_dict[key[6:]] = value
                else:
                    model_state_dict[key] = value
            model.load_state_dict(model_state_dict)
            logger.info(f"Loaded Lightning checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        elif "model_state_dict" in checkpoint:
            # Full checkpoint format
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # Raw state dict
            model.load_state_dict(checkpoint)
    else:
        # Simple state_dict
        model.load_state_dict(checkpoint)

    return model


def unpack_batch(
    batch: tuple,
    device: torch.device,
    use_features: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Unpack a batch from dataloader handling both 2-tuple and 3-tuple formats."""
    if len(batch) == 3:
        images, features, targets = batch
        images = images.to(device)
        features = features.to(device) if use_features else None
        targets = targets.to(device)
    else:
        images, targets = batch
        images = images.to(device)
        targets = targets.to(device)
        features = None
    return images, features, targets


def model_forward(
    model: nn.Module,
    images: torch.Tensor,
    features: torch.Tensor | None,
    use_features: bool,
) -> torch.Tensor:
    """Run model forward pass handling feature/no-feature cases."""
    if use_features and features is not None:
        return model(images, features)
    return model(images)


class ModelLoader:
    """Centralized model loading with automatic config detection."""

    @staticmethod
    def detect_uses_features(cfg: DictConfig) -> bool:
        model_name = cfg.model.name.lower()
        return model_name in DUAL_PATHWAY_MODEL_NAMES

    @staticmethod
    def find_config(
        checkpoint_path: Path,
        config_override: Path | str | None = None,
    ) -> DictConfig:
        if config_override is not None:
            config_path = Path(config_override)
            if not config_path.exists():
                raise FileNotFoundError(f"Config override not found: {config_path}")
            return OmegaConf.load(config_path)

        # Try .hydra/config.yaml
        hydra_config = checkpoint_path.parent / ".hydra" / "config.yaml"
        if hydra_config.exists():
            return OmegaConf.load(hydra_config)

        raise FileNotFoundError(
            f"Config not found for checkpoint {checkpoint_path}. Searched: {hydra_config} "
            "Provide config via --config/-c option."
        )

    @classmethod
    def load(
        cls,
        checkpoint_path: Path | str,
        device: torch.device,
        config_override: Path | str | None = None,
    ) -> LoadedModel:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        cfg = cls.find_config(checkpoint_path, config_override)
        uses_features = cls.detect_uses_features(cfg)
        model_name = cfg.model.name.lower()

        model = load_model_from_checkpoint(checkpoint_path, cfg, device)

        logger.info(f"Loaded {model_name} model (uses_features={uses_features}) from {checkpoint_path}")

        return LoadedModel(
            model=model,
            config=cfg,
            uses_features=uses_features,
            model_name=model_name,
            checkpoint_path=checkpoint_path,
        )


class InferenceEngine:
    """Unified engine for running model inference and collecting results."""

    def __init__(self, loaded_model: LoadedModel, device: torch.device):
        self.model = loaded_model.model
        self.uses_features = loaded_model.uses_features
        self.device = device
        self.model.eval()
        self.model.to(device)

    def run_inference(
        self,
        dataloader: DataLoader,
        collect_images: bool = False,
        collect_features: bool = True,
        desc: str = "Running Inference",
    ) -> PredictionResults:
        """Run full inference loop on dataloader."""
        # Accumulators
        all_preds = []
        all_targets = []
        all_probs = []
        all_logits = []
        all_features = [] if collect_features else None
        all_images = [] if collect_images else None

        logger.info(f"{desc}...")

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                images, features, targets = unpack_batch(batch, self.device, self.uses_features)

                # Forward pass
                outputs = model_forward(self.model, images, features, self.uses_features)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                # Move to CPU and store
                all_logits.append(outputs.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

                if collect_features and features is not None:
                    all_features.append(features.cpu().numpy())

                if collect_images:
                    all_images.append(images.cpu().numpy())

        # Concatenate
        results = PredictionResults(
            predictions=np.concatenate(all_preds),
            targets=np.concatenate(all_targets),
            probabilities=np.concatenate(all_probs),
            logits=np.concatenate(all_logits),
            features=np.concatenate(all_features) if all_features else None,
            images=np.concatenate(all_images) if all_images else None,
            sample_indices=list(range(len(np.concatenate(all_targets)))),
        )

        logger.info(f"Inference complete. Accuracy: {results.accuracy:.4f} ({results.num_samples} samples)")

        return results

    def get_misclassified(self, results: PredictionResults) -> list[dict]:
        """Filter results for misclassified samples."""
        errors = []
        mask = results.predictions != results.targets
        indices = np.where(mask)[0]

        for idx in indices:
            true_idx = results.targets[idx]
            pred_idx = results.predictions[idx]
            error = {
                "sample_idx": idx,
                "true_label": CLASSES[true_idx],
                "pred_label": CLASSES[pred_idx],
                "confidence": float(results.probabilities[idx, pred_idx]),
                "logits": results.logits[idx].tolist(),
                "features": results.features[idx].tolist() if results.features is not None else None,
            }
            errors.append(error)

        return errors
