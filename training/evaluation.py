from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from training.classification import evaluate_classifier_epoch
from training.mae_trainer import reconstruct_mae_images
from training.unet import apply_patch_mask
from utils.visualization import save_comparison_figure


"""Evaluation utilities that compare reconstruction quality and classification metrics."""


@torch.no_grad()
def compare_reconstruction_on_batch(
    mae_model: torch.nn.Module,
    unet_model: torch.nn.Module,
    batch,
    device: torch.device,
    mask_ratio: float = 0.75,
    output_path: str | Path | None = None,
) -> dict[str, float]:
    """Compare MAE and U-Net MSE on one batch and optionally save sample figure."""

    mae_model.eval()
    unet_model.eval()

    images = batch[0].to(device, non_blocking=True)
    masked_images, _ = apply_patch_mask(images, mask_ratio=mask_ratio)

    mae_outputs = mae_model(pixel_values=images)
    mae_loss = mae_outputs.loss.item()
    mae_recon = reconstruct_mae_images(mae_model, images)

    unet_predictions = unet_model(masked_images)
    unet_loss = F.mse_loss(unet_predictions, images).item()

    if output_path is not None:
        save_comparison_figure(
            original=images[0],
            masked=masked_images[0],
            mae_reconstruction=mae_recon[0].detach(),
            unet_reconstruction=unet_predictions[0].detach(),
            output_path=output_path,
        )

    return {"mae_mse": mae_loss, "unet_mse": unet_loss}


@torch.no_grad()
def evaluate_classifier(model: torch.nn.Module, loader, device: torch.device) -> dict[str, float]:
    """Wrapper that returns classifier metrics in a JSON-friendly dictionary."""

    loss, accuracy = evaluate_classifier_epoch(model=model, loader=loader, device=device)
    return {"loss": loss, "accuracy": accuracy}
