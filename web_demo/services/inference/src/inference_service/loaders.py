from __future__ import annotations

from pathlib import Path

import torch
from transformers import ViTMAEConfig, ViTMAEForPreTraining

from .models import MAEClassifier, UNet


def load_mae(weight_path: Path, device: torch.device) -> ViTMAEForPreTraining:
    # Use local config to avoid network dependency during startup.
    cfg = ViTMAEConfig()
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


def resolve_device(preference: str = "auto") -> torch.device:
    """Pick a torch device from a preference string.

    macOS note: `mps` is skipped in `auto` mode because ViTMAE attention ops
    fall back unreliably on MPS; users who want it must set DEVICE=mps explicitly.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(preference)
