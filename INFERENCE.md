# Inference Manual

Standalone CLI for MAE + U-Net patch inpainting and animal classification.

---

## Requirements

Dependencies are managed by **uv** via `pyproject.toml`. No manual install needed.

```
uv run inference.py ...
```

---

## Weight files

Place the three weight files under `weight/` (default) before running:

| File | Description |
|------|-------------|
| `weight/mae_reconstruction.pt` | ViT-MAE-Base encoder + decoder (inpainting) |
| `weight/unet_best.pt` | U-Net inpainting model |
| `weight/mae_cls_best.pth` | ViT-Base encoder + MLP classification head |

---

## Usage

```
uv run inference.py --image <path> [options]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | *(required)* | Path to the input image (any PIL-readable format) |
| `--mask` | `""` (none) | Comma-separated flat patch indices to mask (see [Patch grid](#patch-grid)) |
| `--weight-dir` | `weight/` | Directory containing the three weight files |
| `--output-dir` | `inference_outputs/` | Directory where output images are saved (created if needed) |
| `--topk` | `3` | Number of top classification predictions to print |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |

---

## Patch grid

The input image is resized and center-cropped to **224 × 224** pixels, then divided into a **14 × 14** grid of **16 × 16 px** patches (196 total).

Patch indices are **flat, row-major** starting at 0:

```
 0   1   2  ...  13
14  15  16  ...  27
...
182 183 ...     195
```

To mask a patch at row `r`, column `c` (0-based):

```
index = r * 14 + c
```

### Example — mask a 4 × 6 block in the center (rows 5–8, cols 4–9)

```
r=5: 74 75 76 77 78 79
r=6: 88 89 90 91 92 93
r=7: 102 103 104 105 106 107
r=8: 116 117 118 119 120 121
```

```bash
uv run inference.py --image data/dog.jpg \
  --mask 74,75,76,77,78,79,88,89,90,91,92,93,102,103,104,105,106,107,116,117,118,119,120,121
```

---

## Output files

Three PNG images are written to `--output-dir`:

| File | Contents |
|------|----------|
| `<stem>_masked_input.png` | Preprocessed 224 × 224 image with masked patches zeroed to black |
| `<stem>_mae_recon.png` | MAE reconstruction (original pixels visible + decoder fills masked patches) |
| `<stem>_unet_recon.png` | U-Net reconstruction (inpaints from black-square input) |

---

## Console output

```
Device         : cpu
Image          : data/dog.jpg
Masked patches : 24 / 196

Masked input    → inference_outputs/dog_masked_input.png
MAE  recon      → inference_outputs/dog_mae_recon.png  (MSE: 0.011391)
UNet recon      → inference_outputs/dog_unet_recon.png  (MSE: 0.015320)
Better model (lower MSE): MAE

Top-3 predictions (from MAE reconstruction):
  1. dog            100.00%
  2. cow              0.00%
  3. cat              0.00%
```

**MSE** is computed over masked patches only, in denormalized [0, 1] pixel space — lower is better.
**Classification** always uses the MAE reconstruction as input.

---

## How each model handles the mask

| Model | What it receives | How masking works |
|-------|-----------------|-------------------|
| **MAE** | Original clean image | A `noise` tensor causes ViTMAE to *drop* the masked patch tokens before encoding — the encoder never sees them. The decoder predicts them from context. The final image composites original pixels (visible) + decoder output (masked). |
| **U-Net** | Original image with masked patches zeroed to black | The model learns to inpaint from surrounding context. It explicitly sees the black squares as its input signal. |

---

## Supported classes

The classifier recognises **10 Animals-10 categories**:

`dog` · `horse` · `elephant` · `butterfly` · `chicken` · `cat` · `cow` · `sheep` · `spider` · `squirrel`

---

## Examples

```bash
# Mask patches at the top-left corner
uv run inference.py --image data/dog.jpg --mask 0,1,2,14,15,16,28,29,30

# Mask ~30% of the image randomly (first 60 patches)
uv run inference.py --image data/dog.jpg --mask $(seq 0 59 | tr '\n' ',' | sed 's/,$//')

# Use GPU if available
uv run inference.py --image data/dog.jpg --mask 60,61,74,75 --device cuda

# Custom output directory and top-5 predictions
uv run inference.py --image photo.jpg --mask 50,51,64,65 --output-dir results/ --topk 5
```

---

## Using as a Python module

```python
from pathlib import Path
import torch
from inference import run_inference

result = run_inference(
    image_path     = Path("data/dog.jpg"),
    masked_indices = [60, 61, 62, 74, 75, 76, 88, 89, 90],
    weight_dir     = Path("weight"),
    output_dir     = Path("inference_outputs"),
    device         = torch.device("cpu"),
    topk           = 3,
)

print(result.mae_mse)          # float — MSE over masked patches
print(result.unet_mse)         # float
print(result.top_predictions)  # list of (label, confidence) tuples
print(result.mae_output_path)  # Path to saved MAE reconstruction PNG
print(result.unet_output_path) # Path to saved UNet reconstruction PNG
```
