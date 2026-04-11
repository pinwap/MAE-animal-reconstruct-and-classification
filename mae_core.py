from __future__ import annotations

from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTMAEForPreTraining


"""Stable MAE utility functions.

Keep this file lightweight and reusable for both notebook experiments and CLI runs.
"""


DEFAULT_MAE_MODEL_NAME = "facebook/vit-mae-base"


def load_mae_processor(model_name: str = DEFAULT_MAE_MODEL_NAME) -> ViTImageProcessor:
    """Load Hugging Face image processor for MAE preprocessing."""

    return ViTImageProcessor.from_pretrained(model_name)


def load_mae_model(model_name: str = DEFAULT_MAE_MODEL_NAME, mask_ratio: float = 0.75) -> ViTMAEForPreTraining:
    """Load MAE model and set masking ratio on config when available."""

    model = ViTMAEForPreTraining.from_pretrained(model_name)
    if hasattr(model, "config"):
        model.config.mask_ratio = mask_ratio
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
    if hasattr(model, "unpatchify"):
        try:
            reconstruction = model.unpatchify(reconstruction)
        except Exception:
            pass
    return reconstruction.squeeze(0)


if __name__ == "__main__":
    # Useful quick check when running this file directly.
    print("MAE core helpers are ready.")