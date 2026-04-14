from __future__ import annotations

import json
import random
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.animals10 import IMAGENET_MEAN, IMAGENET_STD


"""
set_seed, get_device, and mixed precision
average meter
"""


class AverageMeter:
    """Track running average for scalar metrics like loss and accuracy."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n


class EarlyStopping:
    """Track validation metric and stop when no meaningful improvement appears."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min") -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: float | None = None
        self.bad_epochs = 0

    def step(self, current_score: float) -> bool:
        """Return True when training should stop early."""

        if self.best_score is None:
            self.best_score = current_score
            self.bad_epochs = 0
            return False

        if self.mode == "min":
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_score
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def set_seed(seed: int = 42) -> None:
    """Set all random seeds used in this project for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "auto") -> torch.device:
    """Pick runtime device. Use auto-detection unless user overrides it."""

    if preferred != "auto":
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextmanager
def autocast_if_available(device: torch.device) -> Iterator[None]:
    """Enable mixed precision only when CUDA is available."""

    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        yield


def create_grad_scaler(device: torch.device) -> torch.cuda.amp.GradScaler | None:
    """Create GradScaler for CUDA training, return None for CPU."""

    if device.type == "cuda":
        return torch.cuda.amp.GradScaler()
    return None


def save_checkpoint(path: str | Path, **payload: Any) -> None:
    """Persist a raw PyTorch checkpoint payload to disk."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True) 
    torch.save(payload, checkpoint_path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint dictionary from disk."""

    return torch.load(Path(path), map_location=map_location)


def to_serializable(value: Any) -> Any:
    """Convert objects (Path/dataclass/containers) to JSON-safe values."""

    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def save_json(path: str | Path, payload: Any) -> None:
    """Write pretty JSON with unicode support for logs/metrics."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2, ensure_ascii=False)


def save_model_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Save a consistent checkpoint format for any training task."""

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics or {},
        "config": config or {},
    }
    save_checkpoint(path, **payload)

# ---------------visualization ---------------

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
