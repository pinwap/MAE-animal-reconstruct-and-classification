from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from .constants import NUM_PATCHES
from .loaders import resolve_device
from .pipeline import load_all, run_once, warmup


def _parse_mask(value: str) -> list[int]:
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
  inference-cli --image cat.jpg --mask 10,11,12
  inference-cli --image dog.png
""",
    )
    parser.add_argument("--image",      required=True, type=Path)
    parser.add_argument("--mask",       default="",    type=str)
    parser.add_argument("--weight-dir", default=Path("weights"), type=Path)
    parser.add_argument("--output-dir", default=Path("inference_outputs"), type=Path)
    parser.add_argument("--topk",       default=3,     type=int)
    parser.add_argument("--device",     default="auto", type=str)
    args = parser.parse_args()

    device = resolve_device(args.device)
    masked_indices = _parse_mask(args.mask)

    print(f"Device         : {device}")
    print(f"Image          : {args.image}")
    print(f"Masked patches : {len(masked_indices)} / {NUM_PATCHES}")

    models = load_all(args.weight_dir, device)
    warmup(models)

    image = Image.open(args.image)
    result = run_once(models, image, masked_indices, topk=args.topk)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    masked_path = args.output_dir / f"{stem}_masked_input.png"
    mae_path    = args.output_dir / f"{stem}_mae_recon.png"
    unet_path   = args.output_dir / f"{stem}_unet_recon.png"
    result.masked_input.save(masked_path)
    result.mae_recon.save(mae_path)
    result.unet_recon.save(unet_path)

    print(f"\nMasked input    → {masked_path}")
    print(f"MAE  recon      → {mae_path}  (MSE: {result.mae_mse:.6f})")
    print(f"UNet recon      → {unet_path}  (MSE: {result.unet_mse:.6f})")
    print(f"Better model (lower MSE): {result.better_model.upper()}")

    print(f"\nTop-{args.topk} predictions (from MAE reconstruction):")
    for rank, (label, prob) in enumerate(result.predictions, start=1):
        print(f"  {rank}. {label:12s}  {prob * 100:6.2f}%")


if __name__ == "__main__":
    main()
