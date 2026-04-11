from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data.animals10 import IMAGENET_MEAN, IMAGENET_STD


"""Image visualization helpers for qualitative reconstruction comparison outputs."""


def denormalize_image(tensor: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    """Undo ImageNet normalization so tensors can be displayed correctly."""

    if tensor.dim() != 3:
        raise ValueError("Expected a CHW tensor")
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return (tensor * std_tensor + mean_tensor).clamp(0.0, 1.0)


def tensor_to_numpy_image(tensor: torch.Tensor):
    """Convert CHW tensor to HWC numpy image for matplotlib."""

    if tensor.dim() != 3:
        raise ValueError("Expected a CHW tensor")
    return denormalize_image(tensor).detach().cpu().permute(1, 2, 0).numpy()


def save_comparison_figure(
    original: torch.Tensor,
    masked: torch.Tensor,
    mae_reconstruction: torch.Tensor,
    unet_reconstruction: torch.Tensor,
    output_path: str | Path,
    title: str = "Reconstruction Comparison",
) -> None:
    """Save side-by-side figure: original, masked, MAE output, U-Net output."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = [original, masked, mae_reconstruction, unet_reconstruction]
    labels = ["Original", "Masked", "MAE", "U-Net"]

    plt.figure(figsize=(14, 4))
    for index, (image, label) in enumerate(zip(images, labels), start=1):
        plt.subplot(1, 4, index)
        plt.imshow(tensor_to_numpy_image(image))
        plt.title(label)
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
