from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from utils.common import AverageMeter, autocast_if_available, save_model_checkpoint


"""U-Net baseline training helpers for masked image reconstruction."""


def apply_patch_mask(images: torch.Tensor, mask_ratio: float = 0.75, patch_size: int = 16):
    """Apply random patch masking to create a 75%-masked reconstruction input."""

    if images.dim() != 4:
        raise ValueError("Expected a BCHW tensor")

    batch_size, channels, height, width = images.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("Image size must be divisible by patch_size")

    grid_h = height // patch_size
    grid_w = width // patch_size
    num_patches = grid_h * grid_w
    num_masked = max(1, int(round(num_patches * mask_ratio)))

    masked_images = images.clone()
    patch_masks = torch.zeros((batch_size, num_patches), device=images.device, dtype=torch.bool)

    for batch_index in range(batch_size):
        # Randomly choose which patches are hidden for this sample.
        permutation = torch.randperm(num_patches, device=images.device)
        masked_indices = permutation[:num_masked]
        patch_masks[batch_index, masked_indices] = True

        reshaped = masked_images[batch_index].reshape(
            channels,
            grid_h,
            patch_size,
            grid_w,
            patch_size,
        )
        reshaped = reshaped.permute(1, 3, 0, 2, 4).contiguous()
        flattened = reshaped.view(num_patches, channels, patch_size, patch_size)
        flattened[masked_indices] = 0.0
        reshaped = flattened.view(grid_h, grid_w, channels, patch_size, patch_size)
        reshaped = reshaped.permute(2, 0, 3, 1, 4).contiguous()
        masked_images[batch_index] = reshaped.view(channels, height, width)

    return masked_images, patch_masks.view(batch_size, grid_h, grid_w)


def train_unet_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mask_ratio: float = 0.75,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> float:
    """Run one U-Net training epoch on masked-to-original reconstruction."""

    model.train()
    meter = AverageMeter()

    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        masked_images, _ = apply_patch_mask(images, mask_ratio=mask_ratio)
        optimizer.zero_grad(set_to_none=True)

        with autocast_if_available(device):
            predictions = model(masked_images)
            loss = F.mse_loss(predictions, images)

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
def evaluate_unet_epoch(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    mask_ratio: float = 0.75,
) -> float:
    """Run one U-Net validation epoch and return mean MSE."""

    model.eval()
    meter = AverageMeter()

    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        masked_images, _ = apply_patch_mask(images, mask_ratio=mask_ratio)
        with autocast_if_available(device):
            predictions = model(masked_images)
            loss = F.mse_loss(predictions, images)
        meter.update(loss.item(), images.size(0))

    return meter.avg


class UNetReconstructionTrainer:
    """Object-oriented trainer for U-Net reconstruction baseline."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        *,
        mask_ratio: float = 0.75,
        scaler: torch.cuda.amp.GradScaler | None = None,
    ) -> None:
        # Keep all train-time dependencies explicit for cleaner orchestration.
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.mask_ratio = mask_ratio
        self.scaler = scaler

    def train_epoch(self, loader) -> float:
        return train_unet_epoch(
            model=self.model,
            loader=loader,
            optimizer=self.optimizer,
            device=self.device,
            mask_ratio=self.mask_ratio,
            scaler=self.scaler,
        )

    def evaluate_epoch(self, loader) -> float:
        return evaluate_unet_epoch(
            model=self.model,
            loader=loader,
            device=self.device,
            mask_ratio=self.mask_ratio,
        )

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        epoch: int,
        metrics: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        save_model_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            metrics=metrics,
            config=config,
        )
