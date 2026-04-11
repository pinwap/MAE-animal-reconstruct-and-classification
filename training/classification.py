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
