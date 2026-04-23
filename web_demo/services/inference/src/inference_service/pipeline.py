from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from transformers import ViTMAEForPreTraining

from .constants import (
    IDX_TO_CLASS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    NUM_PATCHES,
    NUM_PATCHES_SIDE,
    PATCH_SIZE,
)
from .loaders import load_classifier, load_mae, load_unet
from .models import MAEClassifier, UNet

recon_transform = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


@dataclass
class Models:
    mae: ViTMAEForPreTraining
    unet: UNet
    classifier: MAEClassifier
    device: torch.device


@dataclass
class RunResult:
    masked_input: Image.Image
    mae_recon: Image.Image
    unet_recon: Image.Image
    mae_mse: float
    unet_mse: float
    better_model: str
    predictions: list[tuple[str, float]]


def load_all(weight_dir: Path, device: torch.device) -> Models:
    return Models(
        mae        = load_mae(weight_dir / "mae_reconstruction.pt", device),
        unet       = load_unet(weight_dir / "unet_best.pt", device),
        classifier = load_classifier(weight_dir / "mae_cls_best.pth", device),
        device     = device,
    )


def warmup(models: Models) -> None:
    """Run a dummy forward pass on each model to prime kernels / allocator."""
    x = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=models.device)
    with torch.no_grad():
        models.mae.config.mask_ratio = 0.0
        models.mae(pixel_values=x)
        models.unet(x)
        models.classifier(x)


def denormalize(t: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=t.dtype, device=t.device).view(-1, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  dtype=t.dtype, device=t.device).view(-1, 1, 1)
    if t.dim() == 4:
        mean, std = mean.unsqueeze(0), std.unsqueeze(0)
    return (t * std + mean).clamp(0.0, 1.0)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = denormalize(t.detach().cpu()).squeeze(0).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255.0).round().astype("uint8"))


def _build_mask_image(masked_indices: list[int], batch_size: int, device: torch.device) -> torch.Tensor:
    mask_img = torch.zeros(batch_size, 1, IMG_SIZE, IMG_SIZE, device=device)
    for idx in masked_indices:
        r = idx // NUM_PATCHES_SIDE
        c = idx % NUM_PATCHES_SIDE
        mask_img[:, :, r * PATCH_SIZE:(r + 1) * PATCH_SIZE,
                       c * PATCH_SIZE:(c + 1) * PATCH_SIZE] = 1.0
    return mask_img


@torch.no_grad()
def _mae_reconstruct(model: ViTMAEForPreTraining, x: torch.Tensor, masked_indices: list[int]) -> torch.Tensor:
    valid = [i for i in masked_indices if 0 <= i < NUM_PATCHES]
    if not valid:
        return x

    model.config.mask_ratio = len(valid) / NUM_PATCHES
    noise = torch.zeros(x.shape[0], NUM_PATCHES, device=x.device)
    for idx in valid:
        noise[0, idx] = 1.0

    logits = model(pixel_values=x, noise=noise).logits
    pred   = model.unpatchify(logits)

    mask_img = _build_mask_image(valid, x.shape[0], x.device)
    return x * (1.0 - mask_img) + pred * mask_img


@torch.no_grad()
def _unet_inpaint(model: UNet, x: torch.Tensor, masked_indices: list[int]) -> torch.Tensor:
    valid = [i for i in masked_indices if 0 <= i < NUM_PATCHES]
    x_masked = x.clone()
    for idx in valid:
        r = idx // NUM_PATCHES_SIDE
        c = idx % NUM_PATCHES_SIDE
        x_masked[:, :, r * PATCH_SIZE:(r + 1) * PATCH_SIZE,
                       c * PATCH_SIZE:(c + 1) * PATCH_SIZE] = 0.0
    return model(x_masked)


def _masked_mse(original: torch.Tensor, recon: torch.Tensor, masked_indices: list[int]) -> float:
    valid = [i for i in masked_indices if 0 <= i < NUM_PATCHES]
    if not valid:
        return 0.0
    orig_dn  = denormalize(original)
    recon_dn = denormalize(recon)
    mask_img = _build_mask_image(valid, original.shape[0], original.device)
    diff_sq  = (recon_dn - orig_dn) ** 2
    return round(float(((diff_sq * mask_img).sum() / mask_img.sum()).item()), 6)


@torch.no_grad()
def _classify_topk(classifier: MAEClassifier, pixel_values: torch.Tensor, k: int) -> list[tuple[str, float]]:
    probs = F.softmax(classifier(pixel_values), dim=-1).squeeze(0)
    top_probs, top_idx = probs.topk(min(k, probs.numel()))
    return [(IDX_TO_CLASS[int(i)], round(float(p), 4)) for p, i in zip(top_probs, top_idx)]


def _apply_black_patches(x: torch.Tensor, masked_indices: list[int]) -> torch.Tensor:
    out = x.clone()
    for idx in [i for i in masked_indices if 0 <= i < NUM_PATCHES]:
        r = idx // NUM_PATCHES_SIDE
        c = idx % NUM_PATCHES_SIDE
        out[:, :, r * PATCH_SIZE:(r + 1) * PATCH_SIZE,
                  c * PATCH_SIZE:(c + 1) * PATCH_SIZE] = 0.0
    return out


def run_once(
    models: Models,
    image: Image.Image,
    masked_indices: list[int],
    topk: int = 3,
) -> RunResult:
    """In-memory inference. No disk I/O. Returns PIL images + metrics."""
    image_rgb = image.convert("RGB")
    x = recon_transform(image_rgb).unsqueeze(0).to(models.device)

    mae_recon  = _mae_reconstruct(models.mae, x, masked_indices)
    unet_recon = _unet_inpaint(models.unet, x, masked_indices)

    mae_mse  = _masked_mse(x, mae_recon,  masked_indices)
    unet_mse = _masked_mse(x, unet_recon, masked_indices)

    predictions = _classify_topk(models.classifier, mae_recon, k=topk)

    masked_input = _apply_black_patches(x, masked_indices)

    return RunResult(
        masked_input = tensor_to_pil(masked_input),
        mae_recon    = tensor_to_pil(mae_recon),
        unet_recon   = tensor_to_pil(unet_recon),
        mae_mse      = mae_mse,
        unet_mse     = unet_mse,
        better_model = "mae" if mae_mse <= unet_mse else "unet",
        predictions  = predictions,
    )
