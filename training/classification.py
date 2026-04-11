from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from utils.common import AverageMeter, autocast_if_available, save_model_checkpoint


"""Classification stage helpers built on top of ViT/MAE encoder weights."""


def load_mae_encoder_weights_into_classifier(
    classifier_model: nn.Module,
    mae_checkpoint_path: str | Path | None,
) -> nn.Module:
    """Load compatible ViT encoder weights from a MAE checkpoint into classifier."""

    if mae_checkpoint_path is None:
        return classifier_model

    checkpoint_path = Path(mae_checkpoint_path)
    if not checkpoint_path.exists():
        return classifier_model

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    mae_state_dict = checkpoint.get("model_state_dict", checkpoint)

    classifier_state = classifier_model.state_dict()
    transferable_state = {}
    for key, value in mae_state_dict.items():
        # Transfer only ViT encoder parameters with matching tensor shapes.
        if not key.startswith("vit."):
            continue
        if key in classifier_state and classifier_state[key].shape == value.shape:
            transferable_state[key] = value

    classifier_state.update(transferable_state)
    classifier_model.load_state_dict(classifier_state)
    return classifier_model


def train_classifier_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> tuple[float, float]:
    """Run one classifier training epoch and return (loss, accuracy)."""

    model.train()
    criterion = criterion or nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    correct = 0
    total = 0

    for batch in loader:
        images, labels = batch[:2]
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast_if_available(device):
            outputs = model(pixel_values=images)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        loss_meter.update(loss.item(), labels.size(0))

    accuracy = correct / max(total, 1)
    return loss_meter.avg, accuracy


@torch.no_grad()
def evaluate_classifier_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[float, float]:
    """Run one classifier validation epoch and return (loss, accuracy)."""

    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    correct = 0
    total = 0

    for batch in loader:
        images, labels = batch[:2]
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast_if_available(device):
            outputs = model(pixel_values=images)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = criterion(logits, labels)

        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        loss_meter.update(loss.item(), labels.size(0))

    accuracy = correct / max(total, 1)
    return loss_meter.avg, accuracy


def set_classifier_train_mode(
    model: nn.Module,
    mode: str = "end_to_end",
    *,
    unfreeze_last_blocks: int = 1,
    unfreeze_vit_layernorm: bool = True,
) -> int:
    """Configure trainable parameters for classifier fine-tuning strategy.

    Modes:
    - end_to_end: train every parameter.
    - partial: freeze all, then unfreeze classifier head, last ViT blocks,
      and optionally final ViT layer norm.
    """

    normalized_mode = mode.lower().strip()
    if normalized_mode not in {"end_to_end", "partial"}:
        raise ValueError("mode must be one of: end_to_end, partial")

    if normalized_mode == "end_to_end":
        for param in model.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze classification head.
        if hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad = True

        # Unfreeze last N transformer blocks when available.
        if hasattr(model, "vit") and hasattr(model.vit, "encoder") and hasattr(model.vit.encoder, "layer"):
            total_blocks = len(model.vit.encoder.layer)
            n = max(0, min(unfreeze_last_blocks, total_blocks))
            if n > 0:
                for block in model.vit.encoder.layer[-n:]:
                    for param in block.parameters():
                        param.requires_grad = True

        # Optionally unfreeze final layer norm.
        if unfreeze_vit_layernorm and hasattr(model, "vit") and hasattr(model.vit, "layernorm"):
            for param in model.vit.layernorm.parameters():
                param.requires_grad = True

    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable_params


class ViTClassifierTrainer:
    """Object-oriented trainer for MAE-encoder-based image classification."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        *,
        scaler: torch.cuda.amp.GradScaler | None = None,
        criterion: nn.Module | None = None,
    ) -> None:
        # Criterion can be swapped for experimentation (label smoothing, focal, etc.).
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler
        self.criterion = criterion or nn.CrossEntropyLoss()

    def train_epoch(self, loader) -> tuple[float, float]:
        return train_classifier_epoch(
            model=self.model,
            loader=loader,
            optimizer=self.optimizer,
            device=self.device,
            criterion=self.criterion,
            scaler=self.scaler,
        )

    def evaluate_epoch(self, loader) -> tuple[float, float]:
        return evaluate_classifier_epoch(
            model=self.model,
            loader=loader,
            device=self.device,
            criterion=self.criterion,
        )

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        epoch: int,
        metrics: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        # Reuse common format for easier resume and analysis scripts.
        save_model_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            metrics=metrics,
            config=config,
        )
