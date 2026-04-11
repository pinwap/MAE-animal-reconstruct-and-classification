from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from training.common import AverageMeter, autocast_if_available, save_model_checkpoint


"""MAE-specific training/evaluation helpers and OOP trainer wrapper."""


def set_mae_mask_ratio(model: torch.nn.Module, mask_ratio: float) -> None:
    """Apply masking ratio to MAE config in one place."""

    if hasattr(model, "config"):
        model.config.mask_ratio = mask_ratio


def train_mae_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mask_ratio: float = 0.75,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> float:
    """Run one MAE training epoch and return mean reconstruction loss."""

    model.train()
    # Keep mask ratio explicit each epoch so resumed runs remain consistent.
    set_mae_mask_ratio(model, mask_ratio)
    meter = AverageMeter()

    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast_if_available(device):
            outputs = model(pixel_values=images)
            loss = outputs.loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        meter.update(loss.item(), images.size(0))

    return meter.avg


@torch.no_grad()
def evaluate_mae_epoch(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    mask_ratio: float = 0.75,
) -> float:
    """Run one MAE validation epoch and return mean reconstruction loss."""

    model.eval()
    # Validation uses the same masking policy for fair comparison.
    set_mae_mask_ratio(model, mask_ratio)
    meter = AverageMeter()

    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        with autocast_if_available(device):
            outputs = model(pixel_values=images)
            loss = outputs.loss
        meter.update(loss.item(), images.size(0))

    return meter.avg


@torch.no_grad()
def reconstruct_mae_images(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Reconstruct images from MAE outputs, unpatchifying when supported."""

    outputs = model(pixel_values=images)
    reconstructions = outputs.logits
    if hasattr(model, "unpatchify"):
        try:
            reconstructions = model.unpatchify(reconstructions)
        except Exception:
            pass
    return reconstructions


class MAETrainer:
    """Object-oriented trainer for MAE continual pre-training and validation."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        *,
        mask_ratio: float = 0.75,
        scaler: torch.cuda.amp.GradScaler | None = None,
    ) -> None:
        # State is injected to keep this class framework-agnostic and testable.
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.mask_ratio = mask_ratio
        self.scaler = scaler

    def train_epoch(self, loader) -> float:
        return train_mae_epoch(
            model=self.model,
            loader=loader,
            optimizer=self.optimizer,
            device=self.device,
            mask_ratio=self.mask_ratio,
            scaler=self.scaler,
        )

    def evaluate_epoch(self, loader) -> float:
        return evaluate_mae_epoch(
            model=self.model,
            loader=loader,
            device=self.device,
            mask_ratio=self.mask_ratio,
        )

    @torch.no_grad()
    def reconstruct(self, images: torch.Tensor) -> torch.Tensor:
        return reconstruct_mae_images(self.model, images)

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        epoch: int,
        metrics: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        # Delegate to shared serializer so all task checkpoints have same format.
        save_model_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            metrics=metrics,
            config=config,
        )
