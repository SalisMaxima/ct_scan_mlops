"""Custom loss functions for CT scan classification.

This module provides loss functions designed to address class imbalance
and specific confusion patterns (e.g., adenocarcinoma-squamous).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance and hard examples.

    Focal loss reduces the contribution of easy examples and focuses training
    on hard-to-classify samples. This is particularly useful when the model
    is confidently wrong on certain classes.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (default: 2.0). Higher values increase focus on hard examples.
        alpha: Class weights tensor or None for uniform weights.
        reduction: Reduction method ('mean', 'sum', 'none').

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | list[float] | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, list):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Logits of shape (N, C) where C is the number of classes.
            targets: Ground truth labels of shape (N,).

        Returns:
            Scalar loss value.
        """
        # Compute softmax probabilities
        probs = F.softmax(inputs, dim=1)

        # Get the probability of the true class
        # targets: (N,) -> (N, 1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()

        # p_t: probability of true class
        p_t = (probs * targets_one_hot).sum(dim=1)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Cross entropy: -log(p_t)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Cross entropy loss with label smoothing.

    Label smoothing prevents the model from becoming overconfident by
    replacing hard targets (0, 1) with soft targets (epsilon/K, 1-epsilon).

    Args:
        smoothing: Smoothing parameter (default: 0.1).
        num_classes: Number of classes.
        reduction: Reduction method ('mean', 'sum', 'none').
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        num_classes: int = 4,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label smoothing loss.

        Args:
            inputs: Logits of shape (N, C).
            targets: Ground truth labels of shape (N,).

        Returns:
            Scalar loss value.
        """
        confidence = 1.0 - self.smoothing
        log_probs = F.log_softmax(inputs, dim=1)

        # Create smooth labels
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), confidence)

        # KL divergence with smooth targets
        loss = -smooth_targets * log_probs
        loss = loss.sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class PairwiseConfusionLoss(nn.Module):
    """Extra penalty for specific class pair confusions.

    This loss adds an additional penalty when the model predicts one class
    when the true class is another (e.g., adenocarcinoma-squamous confusion).

    Args:
        confusion_pairs: List of (true_class_idx, pred_class_idx) tuples to penalize.
        penalty_weight: Weight for the confusion penalty (default: 0.5).
        base_loss: Base loss function (default: CrossEntropyLoss).
    """

    def __init__(
        self,
        confusion_pairs: list[tuple[int, int]],
        penalty_weight: float = 0.5,
        base_loss: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.confusion_pairs = confusion_pairs
        self.penalty_weight = penalty_weight
        self.base_loss = base_loss if base_loss is not None else nn.CrossEntropyLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute pairwise confusion loss.

        Args:
            inputs: Logits of shape (N, C).
            targets: Ground truth labels of shape (N,).

        Returns:
            Scalar loss value.
        """
        # Base loss
        base = self.base_loss(inputs, targets)

        # Compute probabilities
        probs = F.softmax(inputs, dim=1)

        # Add penalty for specific confusion pairs
        penalty = torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype)

        for true_idx, pred_idx in self.confusion_pairs:
            # Find samples where true class is true_idx
            mask = targets == true_idx

            if mask.sum() > 0:
                # Penalize probability mass on the confused class
                confused_probs = probs[mask, pred_idx]
                penalty = penalty + confused_probs.mean()

        return base + self.penalty_weight * penalty


class WeightedCrossEntropyLoss(nn.Module):
    """Cross entropy loss with class weights.

    Args:
        class_weights: Tensor of weights for each class.
        reduction: Reduction method ('mean', 'sum', 'none').
    """

    def __init__(
        self,
        class_weights: torch.Tensor | list[float],
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if isinstance(class_weights, list):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", class_weights)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted cross entropy loss.

        Args:
            inputs: Logits of shape (N, C).
            targets: Ground truth labels of shape (N,).

        Returns:
            Scalar loss value.
        """
        return F.cross_entropy(inputs, targets, weight=self.class_weights, reduction=self.reduction)


def build_loss(cfg: DictConfig) -> nn.Module:
    """Build loss function from configuration.

    Args:
        cfg: Configuration containing loss settings under cfg.train.loss

    Returns:
        Configured loss function

    Config format:
        loss:
          type: focal  # or cross_entropy, weighted_ce, label_smoothing
          gamma: 2.0  # for focal loss
          class_weights: [1.0, 1.0, 1.3, 0.7]  # optional
          smoothing: 0.1  # for label smoothing

    Raises:
        ValueError: If loss type is unknown
    """
    # Get loss config, default to cross entropy if not specified
    loss_cfg = cfg.get("train", {}).get("loss", {})
    loss_type = loss_cfg.get("type", "cross_entropy")

    # Parse class weights if provided
    class_weights = loss_cfg.get("class_weights")
    if class_weights is not None:
        class_weights = torch.tensor(list(class_weights), dtype=torch.float32)
        logger.info(f"Using class weights: {class_weights.tolist()}")

    if loss_type == "cross_entropy":
        loss = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
        logger.info("Using CrossEntropyLoss")

    elif loss_type == "focal":
        gamma = loss_cfg.get("gamma", 2.0)
        loss = FocalLoss(gamma=gamma, alpha=class_weights)
        logger.info(f"Using FocalLoss with gamma={gamma}")

    elif loss_type == "weighted_ce":
        if class_weights is None:
            raise ValueError("weighted_ce loss requires class_weights to be specified")
        loss = WeightedCrossEntropyLoss(class_weights=class_weights)
        logger.info("Using WeightedCrossEntropyLoss")

    elif loss_type == "label_smoothing":
        smoothing = loss_cfg.get("smoothing", 0.1)
        num_classes = loss_cfg.get("num_classes", 4)
        loss = LabelSmoothingLoss(smoothing=smoothing, num_classes=num_classes)
        logger.info(f"Using LabelSmoothingLoss with smoothing={smoothing}")

    elif loss_type == "pairwise_confusion":
        # Adenocarcinoma (0) <-> Squamous (2) confusion penalty
        pairs = loss_cfg.get("confusion_pairs", [(0, 2), (2, 0)])
        penalty_weight = loss_cfg.get("penalty_weight", 0.5)
        base_loss = None
        if class_weights is not None:
            base_loss = nn.CrossEntropyLoss(weight=class_weights)
        loss = PairwiseConfusionLoss(confusion_pairs=pairs, penalty_weight=penalty_weight, base_loss=base_loss)
        logger.info(f"Using PairwiseConfusionLoss with penalty={penalty_weight}")

    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Choose from: cross_entropy, focal, weighted_ce, label_smoothing, pairwise_confusion"
        )

    return loss
