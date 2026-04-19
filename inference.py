from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from transformers import ViTMAEConfig, ViTMAEForPreTraining, ViTConfig, ViTModel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMG_SIZE      = 224
PATCH_SIZE    = 16
NUM_PATCHES_SIDE = IMG_SIZE // PATCH_SIZE   # 14
NUM_PATCHES      = NUM_PATCHES_SIDE ** 2   # 196

ANIMALS10_EN = [
    "dog", "horse", "elephant", "butterfly", "chicken",
    "cat", "cow", "sheep", "spider", "squirrel",
]
ANIMALS10_IT_SORTED = [
    "cane", "cavallo", "elefante", "farfalla", "gallina",
    "gatto", "mucca", "pecora", "ragno", "scoiattolo",
]
IDX_TO_CLASS = {i: en for i, en in enumerate(ANIMALS10_EN)}


# ---------------------------------------------------------------------------
# Model architectures
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """U-Net (base=64, in/out=3) matching weight/unet_best.pt."""

    def __init__(self) -> None:
        super().__init__()
        b = 64
        self.inc   = DoubleConv(3, b)
        self.down1 = Down(b,      b * 2)
        self.down2 = Down(b * 2,  b * 4)
        self.down3 = Down(b * 4,  b * 8)
        self.down4 = Down(b * 8,  b * 16)
        self.up1   = Up(b * 16 + b * 8, b * 8)
        self.up2   = Up(b * 8  + b * 4, b * 4)
        self.up3   = Up(b * 4  + b * 2, b * 2)
        self.up4   = Up(b * 2  + b,     b)
        self.outc  = OutConv(b, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)


class MAEClassifier(nn.Module):
    """ViT-Base encoder + MLP head matching weight/mae_cls_best.pth."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        cfg = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=IMG_SIZE,
            patch_size=PATCH_SIZE,
            num_channels=3,
        )
        self.encoder = ViTModel(cfg, add_pooling_layer=False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        cls_token = self.encoder(pixel_values=pixel_values).last_hidden_state[:, 0]
        return self.classifier(cls_token)


# ---------------------------------------------------------------------------
# Weight loaders
# ---------------------------------------------------------------------------
def load_mae(weight_path: Path, device: torch.device) -> ViTMAEForPreTraining:
    cfg = ViTMAEConfig.from_pretrained("facebook/vit-mae-base")
    cfg.mask_ratio = 0.0
    model = ViTMAEForPreTraining(cfg)
    ckpt  = torch.load(weight_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.config.mask_ratio = 0.0
    return model.to(device).eval()


def load_unet(weight_path: Path, device: torch.device) -> UNet:
    model = UNet()
    state = torch.load(weight_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state), strict=True)
    return model.to(device).eval()


def load_classifier(weight_path: Path, device: torch.device) -> MAEClassifier:
    model = MAEClassifier(num_classes=10)
    state = torch.load(weight_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state), strict=False)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Transforms / helpers
# ---------------------------------------------------------------------------
recon_transform = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


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
    """Return a [B,1,H,W] float mask; 1.0 at masked patch positions, 0.0 elsewhere."""
    mask_img = torch.zeros(batch_size, 1, IMG_SIZE, IMG_SIZE, device=device)
    for idx in masked_indices:
        r = idx // NUM_PATCHES_SIDE
        c = idx % NUM_PATCHES_SIDE
        mask_img[:, :, r * PATCH_SIZE:(r + 1) * PATCH_SIZE,
                       c * PATCH_SIZE:(c + 1) * PATCH_SIZE] = 1.0
    return mask_img


# ---------------------------------------------------------------------------
# Core inference functions
# ---------------------------------------------------------------------------
@torch.no_grad()
def mae_reconstruct(
    model: ViTMAEForPreTraining,
    x: torch.Tensor,
    masked_indices: list[int],
) -> torch.Tensor:
    """Run MAE with deterministic patch masking and return a composite image.

    Visible patches retain original pixels; masked patches use the decoder
    prediction.  ``masked_indices`` are flat patch indices (0-based, row-major
    over the 14×14 grid).
    """
    valid = [i for i in masked_indices if 0 <= i < NUM_PATCHES]
    if not valid:
        return x  # nothing to reconstruct

    model.config.mask_ratio = len(valid) / NUM_PATCHES

    # Build noise: masked patches get 1.0 (sorted last → removed), others 0.0.
    noise = torch.zeros(x.shape[0], NUM_PATCHES, device=x.device)
    for idx in valid:
        noise[0, idx] = 1.0

    logits = model(pixel_values=x, noise=noise).logits   # [B, N, patch_dim]
    pred   = model.unpatchify(logits)                    # [B, 3, H, W]

    mask_img = _build_mask_image(valid, x.shape[0], x.device)
    return x * (1.0 - mask_img) + pred * mask_img


@torch.no_grad()
def unet_inpaint(
    model: UNet,
    x: torch.Tensor,
    masked_indices: list[int],
) -> torch.Tensor:
    """Zero-out the masked patches in normalized space and run UNet inpainting."""
    valid = [i for i in masked_indices if 0 <= i < NUM_PATCHES]
    x_masked = x.clone()
    for idx in valid:
        r = idx // NUM_PATCHES_SIDE
        c = idx % NUM_PATCHES_SIDE
        x_masked[:, :, r * PATCH_SIZE:(r + 1) * PATCH_SIZE,
                       c * PATCH_SIZE:(c + 1) * PATCH_SIZE] = 0.0
    return model(x_masked)


def masked_mse(
    original: torch.Tensor,
    recon: torch.Tensor,
    masked_indices: list[int],
) -> float:
    """MSE over masked patches only, computed in denormalized [0, 1] space."""
    valid = [i for i in masked_indices if 0 <= i < NUM_PATCHES]
    if not valid:
        return 0.0
    orig_dn  = denormalize(original)
    recon_dn = denormalize(recon)
    mask_img = _build_mask_image(valid, original.shape[0], original.device)
    diff_sq  = (recon_dn - orig_dn) ** 2
    return round(float(((diff_sq * mask_img).sum() / mask_img.sum()).item()), 6)


@torch.no_grad()
def classify_topk(
    classifier: MAEClassifier,
    pixel_values: torch.Tensor,
    k: int = 3,
) -> list[tuple[str, float]]:
    probs = F.softmax(classifier(pixel_values), dim=-1).squeeze(0)
    top_probs, top_idx = probs.topk(k)
    return [(IDX_TO_CLASS[int(i)], round(float(p), 4)) for p, i in zip(top_probs, top_idx)]


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------
@dataclass
class InferenceResult:
    image_path:      Path
    mae_output_path: Path
    unet_output_path: Path
    mae_mse:         float
    unet_mse:        float
    top_predictions: list[tuple[str, float]]


def run_inference(
    image_path: Path,
    masked_indices: list[int],
    weight_dir: Path,
    output_dir: Path,
    device: torch.device,
    topk: int = 3,
) -> InferenceResult:
    """
    Load models, run MAE + UNet reconstruction on ``image_path`` with
    ``masked_indices``, compute per-model MSE, classify via MAE output,
    and save results under ``output_dir``.

    Parameters
    ----------
    image_path:     Path to the source image (any format PIL can read).
    masked_indices: Flat patch indices to mask (0-based, row-major, 14×14 grid).
    weight_dir:     Directory containing the three .pt / .pth weight files.
    output_dir:     Directory where output images are written (created if needed).
    device:         Torch device.
    topk:           Number of top predictions to return.
    """
    mae        = load_mae(weight_dir / "mae_reconstruction.pt", device)
    unet       = load_unet(weight_dir / "unet_best.pt", device)
    classifier = load_classifier(weight_dir / "mae_cls_best.pth", device)

    image = Image.open(image_path).convert("RGB")
    x     = recon_transform(image).unsqueeze(0).to(device)

    mae_recon  = mae_reconstruct(mae, x, masked_indices)
    unet_recon = unet_inpaint(unet, x, masked_indices)

    mae_mse_val  = masked_mse(x, mae_recon,  masked_indices)
    unet_mse_val = masked_mse(x, unet_recon, masked_indices)

    predictions = classify_topk(classifier, mae_recon, k=topk)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    masked_out_path = output_dir / f"{stem}_masked_input.png"
    mae_out_path    = output_dir / f"{stem}_mae_recon.png"
    unet_out_path   = output_dir / f"{stem}_unet_recon.png"

    # Save masked input: zero out the masked patches on the denormalized image
    masked_input = x.clone()
    for idx in [i for i in masked_indices if 0 <= i < NUM_PATCHES]:
        r = idx // NUM_PATCHES_SIDE
        c = idx % NUM_PATCHES_SIDE
        masked_input[:, :, r * PATCH_SIZE:(r + 1) * PATCH_SIZE,
                           c * PATCH_SIZE:(c + 1) * PATCH_SIZE] = 0.0
    tensor_to_pil(masked_input).save(masked_out_path)
    tensor_to_pil(mae_recon).save(mae_out_path)
    tensor_to_pil(unet_recon).save(unet_out_path)

    return InferenceResult(
        image_path       = image_path,
        mae_output_path  = mae_out_path,
        unet_output_path = unet_out_path,
        mae_mse          = mae_mse_val,
        unet_mse         = unet_mse_val,
        top_predictions  = predictions,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_mask(value: str) -> list[int]:
    """Accept comma-separated patch indices, e.g. '0,1,15,196'."""
    if not value.strip():
        return []
    return [int(v) for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MAE + U-Net inpainting and classification (standalone CLI).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Mask patches 10,11,12 and reconstruct:
  python inference.py --image cat.jpg --mask 10,11,12

  # Use all defaults (no masking → MSE = 0, identity reconstruction):
  python inference.py --image dog.png
""",
    )
    parser.add_argument("--image",      required=True,  type=Path, help="Path to input image.")
    parser.add_argument("--mask",       default="",     type=str,
                        help="Comma-separated flat patch indices to mask (0-based, 14×14 grid).")
    parser.add_argument("--weight-dir", default=Path("weight"), type=Path)
    parser.add_argument("--output-dir", default=Path("inference_outputs"), type=Path)
    parser.add_argument("--topk",       default=3,      type=int)
    parser.add_argument("--device",     default="auto", type=str,
                        help="'auto', 'cpu', 'cuda', or 'mps'.")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    masked_indices = _parse_mask(args.mask)

    print(f"Device         : {device}")
    print(f"Image          : {args.image}")
    print(f"Masked patches : {len(masked_indices)} / {NUM_PATCHES}")

    result = run_inference(
        image_path      = args.image,
        masked_indices  = masked_indices,
        weight_dir      = args.weight_dir,
        output_dir      = args.output_dir,
        device          = device,
        topk            = args.topk,
    )

    print(f"\nMasked input    → {args.output_dir}/{result.image_path.stem}_masked_input.png")
    print(f"MAE  recon      → {result.mae_output_path}  (MSE: {result.mae_mse:.6f})")
    print(f"UNet recon      → {result.unet_output_path}  (MSE: {result.unet_mse:.6f})")
    better = "MAE" if result.mae_mse <= result.unet_mse else "UNet"
    print(f"Better model (lower MSE): {better}")

    print(f"\nTop-{args.topk} predictions (from MAE reconstruction):")
    for rank, (label, prob) in enumerate(result.top_predictions, start=1):
        print(f"  {rank}. {label:12s}  {prob * 100:6.2f}%")


if __name__ == "__main__":
    main()
