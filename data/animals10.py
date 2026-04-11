from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


"""Animals-10 dataset utilities.

This module discovers classes from folders, creates deterministic train/val splits,
and builds normalized dataloaders shared by all training stages.
"""


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DatasetSplit:
    """Container for class index mapping and split sample lists."""

    class_to_idx: dict[str, int]
    train_samples: list[tuple[Path, int]]
    val_samples: list[tuple[Path, int]]


def discover_class_directories(root: str | Path) -> list[Path]:
    """Return sorted class folders under dataset root."""

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")

    class_dirs = [path for path in sorted(root_path.iterdir()) if path.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class subdirectories found under: {root_path}")
    return class_dirs


def build_class_to_idx(root: str | Path) -> dict[str, int]:
    """Create deterministic class name -> index mapping."""

    class_dirs = discover_class_directories(root)
    return {directory.name: index for index, directory in enumerate(class_dirs)}


def _list_images(class_dir: Path) -> list[Path]:
    """Recursively collect all supported image files in one class folder."""

    return [path for path in sorted(class_dir.rglob("*")) if path.suffix.lower() in IMAGE_EXTENSIONS]


def build_split(root: str | Path, val_fraction: float = 0.2, seed: int = 42) -> DatasetSplit:
    """Build reproducible per-class train/validation split."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    class_dirs = discover_class_directories(root)
    class_to_idx = {directory.name: index for index, directory in enumerate(class_dirs)}
    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []

    for class_name, class_index in class_to_idx.items():
        class_dir = Path(root) / class_name
        image_paths = _list_images(class_dir)
        if not image_paths:
            continue

        rng = random.Random(seed + class_index)
        rng.shuffle(image_paths)

        val_count = max(1, int(len(image_paths) * val_fraction)) if len(image_paths) > 1 else 0
        if val_count >= len(image_paths):
            val_count = max(0, len(image_paths) - 1)

        val_subset = image_paths[:val_count]
        train_subset = image_paths[val_count:]
        if not train_subset:
            train_subset = image_paths
            val_subset = image_paths[:0]

        train_samples.extend((path, class_index) for path in train_subset)
        val_samples.extend((path, class_index) for path in val_subset)

    if not train_samples:
        raise ValueError(f"No training samples found under: {root}")
    if not val_samples:
        raise ValueError(f"No validation samples found under: {root}")

    return DatasetSplit(class_to_idx=class_to_idx, train_samples=train_samples, val_samples=val_samples)


def build_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> tuple[T.Compose, T.Compose]:
    """Build ImageNet-normalized transforms used for MAE and classifier."""

    transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform, transform


class Animals10Dataset(Dataset):
    """Torch dataset wrapper over (image_path, label) samples."""

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        class_to_idx: dict[str, int],
        transform: Callable | None = None,
        return_path: bool = False,
    ) -> None:
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        # Read lazily to keep memory usage small, then apply transform pipeline.
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.return_path:
            return image, label, str(image_path)
        return image, label


def build_dataloaders(
    root: str | Path,
    batch_size: int = 32,
    image_size: int = DEFAULT_IMAGE_SIZE,
    val_fraction: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    return_path: bool = False,
    train_transform: Callable | None = None,
    val_transform: Callable | None = None,
) -> tuple[DataLoader, DataLoader, DatasetSplit]:
    """Create train/validation dataloaders and return split metadata."""

    split = build_split(root=root, val_fraction=val_fraction, seed=seed)
    default_train_transform, default_val_transform = build_transforms(image_size=image_size)
    if train_transform is None:
        train_transform = default_train_transform
    if val_transform is None:
        val_transform = default_val_transform
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_dataset = Animals10Dataset(
        samples=split.train_samples,
        class_to_idx=split.class_to_idx,
        transform=train_transform,
        return_path=return_path,
    )
    val_dataset = Animals10Dataset(
        samples=split.val_samples,
        class_to_idx=split.class_to_idx,
        transform=val_transform,
        return_path=return_path,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, split


def count_samples_by_class(samples: Iterable[tuple[Path, int]], idx_to_class: dict[int, str]) -> dict[str, int]:
    """Summarize split composition for reporting/debugging."""

    counts = {class_name: 0 for class_name in idx_to_class.values()}
    for _, label in samples:
        class_name = idx_to_class[label]
        counts[class_name] += 1
    return counts
