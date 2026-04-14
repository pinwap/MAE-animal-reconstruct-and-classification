from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTMAEForPreTraining

from utils.common import AverageMeter, autocast_if_available, save_model_checkpoint


"""MAE-specific training/evaluation helpers and OOP trainer wrapper."""


DEFAULT_MAE_MODEL_NAME = "facebook/vit-mae-base"


def load_mae_processor(model_name: str = DEFAULT_MAE_MODEL_NAME) -> ViTImageProcessor:
    """Load Hugging Face image processor for MAE preprocessing."""

    return ViTImageProcessor.from_pretrained(model_name)


def load_mae_model(model_name: str = DEFAULT_MAE_MODEL_NAME, mask_ratio: float = 0.75) -> ViTMAEForPreTraining:
    """Load MAE model and set masking ratio on config when available."""

    model = ViTMAEForPreTraining.from_pretrained(model_name)
    set_mae_mask_ratio(model, mask_ratio)
    return model


def prepare_mae_input(image: Image.Image, processor: ViTImageProcessor) -> torch.Tensor:
    """Convert a PIL image to model-ready pixel tensor."""

    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"]


@torch.no_grad() 
def reconstruct_image(
    image: Image.Image,
    model: ViTMAEForPreTraining | None = None,
    processor: ViTImageProcessor | None = None,
) -> torch.Tensor:
    """Return a reconstruction tensor for a single PIL image."""

    processor = processor or load_mae_processor()
    model = model or load_mae_model()
    pixel_values = prepare_mae_input(image, processor)
    outputs = model(pixel_values=pixel_values)
    reconstruction = outputs.logits
    #ถ้าโมเดลมีฟังก์ชัน unpatchify ให้ลองเรียกใช้เพื่อแปลง reconstruction จาก patchified form กลับเป็นรูปภาพเต็มๆ แต่ถ้าเกิดข้อผิดพลาดขึ้นก็ให้ข้ามไปและคืนค่า reconstruction ในรูปแบบ patchified เดิม
    if hasattr(model, "unpatchify"):
        try:
            reconstruction = model.unpatchify(reconstruction)
        except Exception:
            pass
    return reconstruction.squeeze(0)


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
    scheduler: Any | None = None,
) -> float:
    """Run one MAE training epoch and return mean reconstruction loss."""

    model.train()
    # Keep mask ratio explicit each epoch so resumed runs remain consistent.
    set_mae_mask_ratio(model, mask_ratio)
    meter = AverageMeter()

    for batch in loader:
        images = batch[0].to(device, non_blocking=True) #ดึงรูปทีละ batch จาก dataloader แล้วส่งไปที่ device โดยใช้ non_blocking=True เพื่อให้การส่งข้อมูลเป็นแบบ asynchronous ซึ่งจะช่วยเพิ่มประสิทธิภาพในการฝึกโมเดลได้มากขึ้น
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

        if scheduler is not None:
            scheduler.step()

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

    def train_epoch(self, loader, scheduler: Any | None = None) -> float:
        return train_mae_epoch(
            model=self.model,
            loader=loader,
            optimizer=self.optimizer,
            device=self.device,
            mask_ratio=self.mask_ratio,
            scaler=self.scaler,
            scheduler=scheduler,
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
