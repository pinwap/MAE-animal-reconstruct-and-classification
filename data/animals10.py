from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import WeightedRandomSampler


"""Animals-10 dataset utilities.

This module discovers classes from folders, creates deterministic train/val splits,
and builds normalized dataloaders shared by all training stages.
"""

# เนื่องจากเราจะ transfer learning จากโมเดลที่ถูกฝึกบน ImageNet เลยต้องตั้งค่าภาพให้เหมือนกับที่โมเดลถูกฝึกมา
# ต้อง normalize x' = (x - mean)\std ให้ distribution ของภาพใกล้เคียง
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406) 
IMAGENET_STD = (0.229, 0.224, 0.225)
ANIMALS10_IT_TO_EN = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel",
}
ANIMALS10_EXPECTED_EN_CLASSES = set(ANIMALS10_IT_TO_EN.values())


@dataclass(frozen=True)
class DatasetSplit:
    """Container for class index mapping and split sample lists."""

    class_to_idx: dict[str, int]
    train_samples: list[tuple[Path, int]]
    val_samples: list[tuple[Path, int]]


def discover_class_directories(root: str | Path) -> list[Path]:
    """Return โฟลเดอร์ที่เป็น class ต่างๆ ที่อยู่ภายใต้ root directory เรียงตามลำดับตัวอักษร."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")

    class_dirs = [path for path in sorted(root_path.iterdir()) if path.is_dir()] # ตรวจสอบว่าเป็น directory หรือไม่ เพราะแต่ละ class จะถูกเก็บใน folder ของมันเอง
    if not class_dirs:
        raise ValueError(f"No class subdirectories found under: {root_path}")
    return class_dirs


def build_class_to_idx(root: str | Path) -> dict[str, int]:
    """สร้าง mapping จากชื่อ class (ชื่อ folder) ไปเป็นตัวเลข index เรียงตามลำดับตัวอักษรของ folder เช่น cat: 0, dog: 1, ..."""

    class_dirs = discover_class_directories(root)
    return {directory.name: index for index, directory in enumerate(class_dirs)} 


def _list_images(class_dir: Path) -> list[Path]:
    """ส่งกลับ list ของ path ของรูปภาพทั้งหมดที่อยู่ใน folder ของ class นั้นๆ : โดยจะตรวจสอบนามสกุลของไฟล์ว่าตรงกับที่กำหนดไว้ใน IMAGE_EXTENSIONS หรือไม่ และเรียงตามลำดับตัวอักษรของชื่อไฟล์ด้วย"""
    
    return [path for path in sorted(class_dir.rglob("*")) if path.suffix.lower() in IMAGE_EXTENSIONS]


def build_split(root: str | Path, val_fraction: float = 0.2, seed: int = 42) -> DatasetSplit:
    """Build reproducible per-class train/validation split.
    
    """

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    class_to_idx = build_class_to_idx(root) 
    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []

    for class_name, class_index in class_to_idx.items():
        class_dir = Path(root) / class_name # เขียนแบบนี้มันจะใส่เครื่องหมาย / path ให้เอง เช่น windowใช้ \ แต่ linux ใช้ / 
        image_paths = _list_images(class_dir) # สร้าง list ของ path ของรูปภาพทั้งหมดที่อยู่ใน folder ของ class นั้นๆ
        if not image_paths:
            continue

        rng = random.Random(seed + class_index)
        rng.shuffle(image_paths)

        # 20% ของรูปทั้งหมดในคลาสนี้ คิดเป็นกี่รูป (เพื่อให้แบ่งไปเป็นชุด Val)
        val_count = max(1, int(len(image_paths) * val_fraction)) if len(image_paths) > 1 else 0
        if val_count >= len(image_paths): # ถ้า val_count มากกว่าหรือเท่ากับจำนวนรูปทั้งหมดในคลาสนี้
            val_count = max(0, len(image_paths) - 1) # len - 1 เพื่อให้แน่ใจว่าอย่างน้อยจะมีตัวอย่างหนึ่งในชุด Train

        val_subset = image_paths[:val_count] # แบ่งจำนวน val_count แรกไปเป็นชุด Val
        train_subset = image_paths[val_count:] # ที่เหลือไปเป็นชุด Train
        if not train_subset: # ถ้า train_subset ว่างเปล่า (เกิดขึ้นเมื่อมีรูปเดียวในคลาสนี้และ val_fraction ถูกตั้งไว้สูงเกินไป) ให้ย้ายรูปนั้นไปเป็นชุด Train แทน
            train_subset = image_paths
            val_subset = image_paths[:0] # val_subset เป็น เซตว่างไป

        # ใส่กล่องกลางของรูปภาพและ label
        train_samples.extend((path, class_index) for path in train_subset)
        val_samples.extend((path, class_index) for path in val_subset)

    if not train_samples:
        raise ValueError(f"No training samples found under: {root}")
    if not val_samples:
        raise ValueError(f"No validation samples found under: {root}")

    return DatasetSplit(class_to_idx=class_to_idx, train_samples=train_samples, val_samples=val_samples)


def map_animals10_labels_to_english(
    class_to_idx_it: dict[str, int],
) -> tuple[dict[str, int], dict[int, str]]:
    """Map Animals-10 folder names (Italian) to English labels with validation."""

    dataset_classes_it = set(class_to_idx_it.keys())
    unknown_it = sorted(dataset_classes_it - set(ANIMALS10_IT_TO_EN.keys()))
    missing_it = sorted(set(ANIMALS10_IT_TO_EN.keys()) - dataset_classes_it)
    if unknown_it or missing_it:
        raise ValueError(
            f"Unexpected class folders. unknown={unknown_it}, missing={missing_it}. "
            "Please check DATA_ROOT points to Animals-10 raw-img."
        )

    class_to_idx_en = {
        ANIMALS10_IT_TO_EN[class_name]: class_index
        for class_name, class_index in class_to_idx_it.items()
    }
    if set(class_to_idx_en.keys()) != ANIMALS10_EXPECTED_EN_CLASSES:
        raise ValueError("English class mapping is incomplete or incorrect.")

    idx_to_class_en = {index: english for english, index in class_to_idx_en.items()}
    return class_to_idx_en, idx_to_class_en


def _build_weighted_sampler(train_samples: list[tuple[Path, int]]) -> WeightedRandomSampler:
    """Build weighted sampler to reduce class imbalance effect during training."""

    class_counts: dict[int, int] = {}
    for _, label in train_samples:
        class_counts[label] = class_counts.get(label, 0) + 1 # ดูว่าเคยนับคลาส label นี้ไปกี่ครั้งแล้ว ถ้ายังไม่เคยนับเลย ให้ถือว่าค่าปัจจุบันคือ 0

    if not class_counts:
        raise ValueError("Cannot build weighted sampler with empty train_samples")

    # ให้น้ำหนักกับแต่ละตัวอย่างในชุด Train ที่เป็นสัดส่วนผกผันกับจำนวนตัวอย่างในคลาสนั้นๆ เช่น ถ้าในคลาสนี้มีตัวอย่างน้อย ก็จะให้น้ำหนักมากขึ้น เพื่อให้โมเดลมีโอกาสหยิบได้คลาสนั้นมากขึ้นตอนฝึก
    sample_weights = [1.0 / class_counts[label] for _, label in train_samples] 
    weights_tensor = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weights=weights_tensor, num_samples=len(weights_tensor), replacement=True)


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

        if self.return_path: # ถ้า return_path เป็น True จะส่งกลับ tuple ที่มี image, label และ path ของรูปภาพด้วย แต่ถ้าเป็น False จะส่งกลับแค่ image กับ label เท่านั้น
            return image, label, str(image_path)
        return image, label

"""data loader ไว้แบ่ง batch, shuffle, และจัดการกับการโหลดข้อมูลใน background เพื่อให้ GPU ไม่ต้องรอข้อมูลจาก CPU นานเกินไป ซึ่งจะช่วยเพิ่มประสิทธิภาพในการฝึกโมเดลได้มากขึ้น"""
def build_dataloaders(
    root: str | Path,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    seed: int = 42,
    num_workers: int = 0, # ถ้า num_workers > 0 จะใช้ subprocess ในการโหลดข้อมูล เปิด Multi-processing ให้ CPU หลายๆ แกนช่วยกันไปอ่านรูปจากดิสก์มารอไว้ ซึ่งจะช่วยเพิ่มความเร็วในการโหลดข้อมูล 
                        #แต่ก็อาจทำให้เกิดปัญหาในบางระบบปฏิบัติการหรือบางสถานการณ์ได้ เช่น ใน Windows ดังนั้นควรตั้งค่า num_workers เป็น 0 ใน Windows เพื่อหลีกเลี่ยงปัญหานี้
    pin_memory: bool | None = None, # ควรตั้งค่า pin_memory เป็น True เมื่อใช้ GPU และเป็น False หรือ None เมื่อใช้ CPU เพราะถ้า pin_memory เป็น True จะทำให้ DataLoader ส่งข้อมูลไปยัง GPU ได้เร็วขึ้น แต่ก็จะใช้หน่วยความจำมากขึ้น
    return_path: bool = False,
    train_transform: Callable | None = None,
    val_transform: Callable | None = None,
    use_weighted_sampler: bool = False, # true = ใช้ weighted sampler ให้น้ำหนักคลาสที่มี train samples น้อยกว่า คือจะสุ่มหยิบภาพนั้นซ้ำๆ เพื่อช่วยลดผลกระทบของ class imbalance ในการฝึกโมเดล
) -> tuple[DataLoader, DataLoader, DatasetSplit]:
    """Create train/validation dataloaders and return split metadata."""

    if train_transform is None or val_transform is None:
        raise ValueError("train_transform and val_transform are required")

    split = build_split(root=root, val_fraction=val_fraction, seed=seed)
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

    train_sampler = _build_weighted_sampler(split.train_samples) if use_weighted_sampler else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
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
