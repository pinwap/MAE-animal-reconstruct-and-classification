from __future__ import annotations

import base64
import io
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms as T
from transformers import ViTMAEConfig, ViTMAEForPreTraining, ViTConfig, ViTModel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224
WEIGHT_DIR = Path(__file__).resolve().parents[2] / "weight"

ANIMALS10_IT_SORTED = [
    "cane", "cavallo", "elefante", "farfalla", "gallina",
    "gatto", "mucca", "pecora", "ragno", "scoiattolo",
]
ANIMALS10_EN = [
    "dog", "horse", "elephant", "butterfly", "chicken",
    "cat", "cow", "sheep", "spider", "squirrel",
]
IDX_TO_CLASS = {i: en for i, en in enumerate(ANIMALS10_EN)}


# ---------------------------------------------------------------------------
# Model architectures
# ---------------------------------------------------------------------------
class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class _OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _UNet(nn.Module):
    """U-Net inferred from weight/unet_best.pt (base=64, in/out=3)."""

    def __init__(self) -> None:
        super().__init__()
        b = 64
        self.inc = _DoubleConv(3, b)
        self.down1 = _Down(b, b * 2)
        self.down2 = _Down(b * 2, b * 4)
        self.down3 = _Down(b * 4, b * 8)
        self.down4 = _Down(b * 8, b * 16)
        self.up1 = _Up(b * 16 + b * 8, b * 8)
        self.up2 = _Up(b * 8 + b * 4, b * 4)
        self.up3 = _Up(b * 4 + b * 2, b * 2)
        self.up4 = _Up(b * 2 + b, b)
        self.outc = _OutConv(b, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class _MAEClassifier(nn.Module):
    """ViT-Base encoder + MLP head inferred from weight/mae_cls_best.pth.

    Head layout (indices match saved state dict):
      0: LayerNorm(768)  1: Linear(768,512)  2: GELU  3: Dropout
      4: Linear(512,256) 5: GELU             6: Dropout  7: Linear(256,10)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        cfg = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=IMG_SIZE,
            patch_size=16,
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
def _load_mae(device: torch.device) -> ViTMAEForPreTraining:
    cfg = ViTMAEConfig.from_pretrained("facebook/vit-mae-base")
    cfg.mask_ratio = 0.0
    model = ViTMAEForPreTraining(cfg)
    ckpt = torch.load(WEIGHT_DIR / "mae_reconstruction.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.config.mask_ratio = 0.0
    return model.to(device).eval()


def _load_unet(device: torch.device) -> _UNet:
    model = _UNet()
    state = torch.load(WEIGHT_DIR / "unet_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state), strict=True)
    return model.to(device).eval()


def _load_classifier(device: torch.device) -> _MAEClassifier:
    model = _MAEClassifier(num_classes=10)
    state = torch.load(WEIGHT_DIR / "mae_cls_best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state), strict=False)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
_recon_transform = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def _denormalize(t: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=t.dtype, device=t.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=t.dtype, device=t.device).view(-1, 1, 1)
    if t.dim() == 4:
        mean, std = mean.unsqueeze(0), std.unsqueeze(0)
    return (t * std + mean).clamp(0.0, 1.0)


def _tensor_to_b64(t: torch.Tensor) -> str:
    arr = _denormalize(t.detach().cpu()).squeeze(0).permute(1, 2, 0).numpy()
    pil = Image.fromarray((arr * 255.0).round().astype("uint8"))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@torch.no_grad()
def _mae_reconstruct(
    model: ViTMAEForPreTraining,
    x: torch.Tensor,
    masked_indices: list[int],
) -> torch.Tensor:
    """Run MAE with the given patch mask and return a composite image.

    Visible patches use the original pixels; masked patches use the
    MAE decoder's prediction.
    """
    patch_size = model.config.patch_size  # 16
    num_patches_side = IMG_SIZE // patch_size  # 14
    num_patches = num_patches_side ** 2  # 196
    batch_size = x.shape[0]

    valid_mask = [i for i in masked_indices if 0 <= i < num_patches]
    if not valid_mask:
        # Nothing to reconstruct – return the original tensor.
        return x

    mask_ratio = len(valid_mask) / num_patches
    model.config.mask_ratio = mask_ratio

    # Build noise tensor: masked patches → 1.0 (removed), others → 0.0 (kept).
    # ViTMAE keeps the patches with the *smallest* noise values (argsort ascending).
    noise = torch.zeros(batch_size, num_patches, device=x.device)
    for idx in valid_mask:
        noise[0, idx] = 1.0

    outputs = model(pixel_values=x, noise=noise)
    logits = outputs.logits  # [B, num_patches, patch_size^2 * 3]

    # Unpatchify → full reconstructed image in normalized pixel space.
    pred = model.unpatchify(logits)  # [B, 3, H, W]

    # Build a pixel-space binary mask (1 = masked patch, 0 = visible patch).
    cell = patch_size
    mask_img = torch.zeros(batch_size, 1, IMG_SIZE, IMG_SIZE, device=x.device)
    for idx in valid_mask:
        r = idx // num_patches_side
        c = idx % num_patches_side
        mask_img[:, :, r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = 1.0

    # Composite: keep original pixels where visible, use prediction where masked.
    return x * (1.0 - mask_img) + pred * mask_img


@torch.no_grad()
def _unet_inpaint(model: _UNet, x: torch.Tensor, masked_indices: list[int]) -> torch.Tensor:
    """Apply black-square mask to the image and run UNet inpainting."""
    patch_size = 16
    num_patches_side = IMG_SIZE // patch_size
    num_patches = num_patches_side ** 2

    x_masked = x.clone()
    for idx in masked_indices:
        if 0 <= idx < num_patches:
            r = idx // num_patches_side
            c = idx % num_patches_side
            x_masked[:, :, r * patch_size:(r + 1) * patch_size, c * patch_size:(c + 1) * patch_size] = 0.0

    return model(x_masked)


def _masked_mse(
    original: torch.Tensor,
    recon: torch.Tensor,
    masked_indices: list[int],
) -> float:
    """MSE between original and reconstruction over masked patches only (denormalized [0,1])."""
    patch_size = 16
    num_patches_side = IMG_SIZE // patch_size
    num_patches = num_patches_side ** 2

    valid_mask = [i for i in masked_indices if 0 <= i < num_patches]
    if not valid_mask:
        return 0.0

    orig_dn = _denormalize(original)
    recon_dn = _denormalize(recon)

    mask_img = torch.zeros_like(orig_dn)
    for idx in valid_mask:
        r = idx // num_patches_side
        c = idx % num_patches_side
        mask_img[:, :, r * patch_size:(r + 1) * patch_size, c * patch_size:(c + 1) * patch_size] = 1.0

    diff_sq = (recon_dn - orig_dn) ** 2
    mse = (diff_sq * mask_img).sum() / mask_img.sum()
    return round(float(mse.item()), 6)


@torch.no_grad()
def _topk(classifier: _MAEClassifier, recon: torch.Tensor, k: int = 3) -> list[dict]:
    probs = F.softmax(classifier(recon), dim=-1).squeeze(0)
    top_probs, top_idx = probs.topk(k)
    return [
        {"label": IDX_TO_CLASS[int(i)], "confidence": round(float(p), 4)}
        for p, i in zip(top_probs, top_idx)
    ]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="MAE Animal Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mae_model: ViTMAEForPreTraining | None = None
unet_model: _UNet | None = None
cls_model: _MAEClassifier | None = None


@app.on_event("startup")
def _load_models() -> None:
    global mae_model, unet_model, cls_model
    print(f"Loading models on {device} ...")
    mae_model = _load_mae(device)
    unet_model = _load_unet(device)
    cls_model = _load_classifier(device)
    print("Models ready.")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class Prediction(BaseModel):
    label: str
    confidence: float


class InferenceResponse(BaseModel):
    mae_reconstruction: str   # base64 PNG
    unet_reconstruction: str  # base64 PNG
    mae_mse: float            # MSE over masked patches (denormalized)
    unet_mse: float           # MSE over masked patches (denormalized)
    top_predictions: list[Prediction]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "device": str(device)}


@app.post("/infer", response_model=InferenceResponse)
async def infer(
    file: UploadFile = File(...),
    mask: str = Form(""),
) -> InferenceResponse:
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {exc}")

    # Parse comma-separated masked patch indices sent from the frontend.
    masked_indices: list[int] = []
    if mask:
        try:
            masked_indices = [int(i) for i in mask.split(",") if i.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid mask format: expected comma-separated integers.")

    x = _recon_transform(image).unsqueeze(0).to(device)

    mae_recon = _mae_reconstruct(mae_model, x, masked_indices)
    unet_recon = _unet_inpaint(unet_model, x, masked_indices)
    predictions = _topk(cls_model, mae_recon, k=3)

    mae_mse = _masked_mse(x, mae_recon, masked_indices)
    unet_mse = _masked_mse(x, unet_recon, masked_indices)

    return InferenceResponse(
        mae_reconstruction=_tensor_to_b64(mae_recon),
        unet_reconstruction=_tensor_to_b64(unet_recon),
        mae_mse=mae_mse,
        unet_mse=unet_mse,
        top_predictions=[Prediction(**p) for p in predictions],
    )
