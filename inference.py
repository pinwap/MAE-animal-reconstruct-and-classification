from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from PIL import Image
import numpy as np


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_MAE_MODEL_NAME = "facebook/vit-mae-base"

IDX_TO_CLASS = {
    0: "dog",
    1: "horse",
    2: "elephant",
    3: "butterfly",
    4: "chicken",
    5: "cat",
    6: "cow",
    7: "sheep",
    8: "spider",
    9: "squirrel",
}


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(kernel_size=2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
        self.up1 = Up(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.up2 = Up(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.up3 = Up(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up4 = Up(base_channels * 2 + base_channels, base_channels)
        self.outc = OutConv(base_channels, out_channels)

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


class MAEWithClassifier(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.0, model_name: str = DEFAULT_MAE_MODEL_NAME) -> None:
        super().__init__()
        from transformers import ViTMAEModel

        self.encoder = ViTMAEModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_token)


def _extract_state_dict(checkpoint_or_state: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint_or_state, dict) and "model_state_dict" in checkpoint_or_state:
        model_state_dict = checkpoint_or_state["model_state_dict"]
        if isinstance(model_state_dict, dict):
            return model_state_dict
    if isinstance(checkpoint_or_state, dict):
        return checkpoint_or_state
    raise TypeError("Checkpoint must be a state_dict or checkpoint dict")


def _maybe_strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if all(key.startswith(prefix) for key in state_dict.keys()):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def _load_mae_reconstruction_model(
    mae_checkpoint_path: str | Path,
    device: torch.device,
    model_name: str,
    mask_ratio: float,
) -> nn.Module:
    from transformers import ViTMAEForPreTraining

    model = ViTMAEForPreTraining.from_pretrained(model_name)
    model.config.mask_ratio = mask_ratio

    checkpoint_raw = torch.load(mae_checkpoint_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint_raw)
    state_dict = _maybe_strip_prefix(state_dict, "module.")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[MAE] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    model.to(device).eval()
    return model


def _load_unet_model(unet_checkpoint_path: str | Path, device: torch.device) -> UNet:
    model = UNet()
    checkpoint_raw = torch.load(unet_checkpoint_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint_raw)
    state_dict = _maybe_strip_prefix(state_dict, "module.")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[UNet] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    model.to(device).eval()
    return model


def _load_classifier_model(
    cls_checkpoint_path: str | Path,
    device: torch.device,
    model_name: str,
    num_classes: int,
    dropout: float,
) -> MAEWithClassifier:
    model = MAEWithClassifier(num_classes=num_classes, dropout=dropout, model_name=model_name)

    checkpoint_raw = torch.load(cls_checkpoint_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint_raw)
    state_dict = _maybe_strip_prefix(state_dict, "module.")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[Classifier] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    model.to(device).eval()
    return model


def _normalize_to_pil(tensor_bchw_or_chw: torch.Tensor) -> Image.Image:
    if tensor_bchw_or_chw.dim() == 4:
        tensor = tensor_bchw_or_chw[0]
    elif tensor_bchw_or_chw.dim() == 3:
        tensor = tensor_bchw_or_chw
    else:
        raise ValueError("Expected CHW or BCHW tensor")

    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    denormalized = (tensor * std + mean).clamp(0.0, 1.0)
    image_np = (denormalized.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(image_np)


class AnimalPipelineInference:
    def __init__(
        self,
        mae_recon_path: str | Path,
        cls_path: str | Path,
        unet_recon_path: str | Path | None = None,
        model_name: str = DEFAULT_MAE_MODEL_NAME,
        mask_ratio: float = 0.0,
        num_classes: int = 10,
        classifier_dropout: float = 0.0,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mask_ratio = mask_ratio
        from transformers import ViTImageProcessor

        self.processor = ViTImageProcessor.from_pretrained(model_name)

        print(f"Loading inference models on {self.device}...")
        self.mae_recon = _load_mae_reconstruction_model(
            mae_checkpoint_path=mae_recon_path,
            device=self.device,
            model_name=model_name,
            mask_ratio=mask_ratio,
        )
        self.classifier = _load_classifier_model(
            cls_checkpoint_path=cls_path,
            device=self.device,
            model_name=model_name,
            num_classes=num_classes,
            dropout=classifier_dropout,
        )

        self.unet_recon: UNet | None = None
        if unet_recon_path is not None and Path(unet_recon_path).exists():
            self.unet_recon = _load_unet_model(unet_checkpoint_path=unet_recon_path, device=self.device)
        elif unet_recon_path is not None:
            print(f"UNet checkpoint not found at {unet_recon_path}. UNet stage will be skipped.")

    def _prepare_input(self, image_path: str | Path) -> tuple[Image.Image, torch.Tensor]:
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)
        return image, pixel_values

    @torch.no_grad()
    def process(self, image_path: str | Path, output_dir: str | Path = "static") -> dict[str, Any]:
        _, input_tensor = self._prepare_input(image_path)

        self.mae_recon.config.mask_ratio = self.mask_ratio
        mae_outputs = self.mae_recon(pixel_values=input_tensor)
        mae_recon = mae_outputs.logits
        if hasattr(self.mae_recon, "unpatchify"):
            try:
                mae_recon = self.mae_recon.unpatchify(mae_recon)
            except Exception:
                pass

        unet_recon: torch.Tensor | None = None
        if self.unet_recon is not None:
            unet_recon = self.unet_recon(input_tensor)

        cls_logits = self.classifier(mae_recon)
        probabilities = torch.softmax(cls_logits, dim=1).squeeze(0)
        top3_prob, top3_idx = torch.topk(probabilities, 3)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        mae_image_path = output_path / "mae_result.jpg"
        mae_image = _normalize_to_pil(mae_recon)
        mae_image.save(mae_image_path)

        unet_image_path: Path | None = None
        if unet_recon is not None:
            unet_image_path = output_path / "unet_result.jpg"
            _normalize_to_pil(unet_recon).save(unet_image_path)

        top_predictions = []
        for rank in range(3):
            index = int(top3_idx[rank].item())
            probability = float(top3_prob[rank].item() * 100.0)
            top_predictions.append(
                {
                    "rank": rank + 1,
                    "class_index": index,
                    "class_name": IDX_TO_CLASS.get(index, f"class_{index}"),
                    "probability": round(probability, 4),
                }
            )

        result: dict[str, Any] = {
            "input_path": str(image_path),
            "mask_ratio": self.mask_ratio,
            "reconstruction_paths": {
                "mae": str(mae_image_path),
                "unet": str(unet_image_path) if unet_image_path is not None else None,
            },
            "classification": {
                "source_model": "mae_reconstructed",
                "top_3_predictions": top_predictions,
            },
        }
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-file inference for MAE reconstruction + classification")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--mae-ckpt", type=str, default="checkpoint/mae_reconstruct.pt", help="MAE reconstruction checkpoint")
    parser.add_argument("--cls-ckpt", type=str, default="checkpoint/mae_cls_best.pth", help="Classifier checkpoint")
    parser.add_argument("--unet-ckpt", type=str, default="checkpoint/unet_final.pt", help="Optional UNet checkpoint")
    parser.add_argument("--output-dir", type=str, default="static", help="Directory to save output images")
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.0,
        help="Additional MAE internal mask ratio; keep 0.0 when input is already masked",
    )
    args = parser.parse_args()

    unet_path: str | None = args.unet_ckpt if Path(args.unet_ckpt).exists() else None

    pipeline = AnimalPipelineInference(
        mae_recon_path=args.mae_ckpt,
        cls_path=args.cls_ckpt,
        unet_recon_path=unet_path,
        mask_ratio=args.mask_ratio,
    )
    output = pipeline.process(args.image, output_dir=args.output_dir)
    print(json.dumps(output, indent=2, ensure_ascii=False))
