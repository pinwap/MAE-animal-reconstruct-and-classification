# Inference Service

FastAPI service + CLI for MAE + U-Net patch inpainting and Animals-10 classification.

## Quickstart

```bash
# Create venv + install deps (run from repo root: `make install-inference`)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .

# Place weight files under ./weights/
#   weights/mae_reconstruction.pt
#   weights/unet_best.pt
#   weights/mae_cls_best.pth

# Run the HTTP service
.venv/bin/uvicorn inference_service.main:app --host 0.0.0.0 --port 8000

# Or run the CLI (same behavior as before)
.venv/bin/inference-cli --image path/to/dog.jpg --mask 74,75,88,89
```

Environment variables (`.env.example`):

| Var          | Default   | Description                               |
|--------------|-----------|-------------------------------------------|
| `DEVICE`     | `auto`    | `auto` / `cpu` / `cuda` / `mps`           |
| `WEIGHT_DIR` | `weights` | Directory containing the 3 weight files   |
| `PORT`       | `8000`    | HTTP port                                  |

> **macOS note:** `auto` picks `cuda` or `cpu`. `mps` is supported but opt-in — ViTMAE attention ops have been unreliable on MPS in some torch releases.

## HTTP API

### `GET /health`
```json
{ "status": "ok", "device": "cuda", "models_loaded": true }
```

### `POST /infer`

Request:
```json
{
  "image_base64": "<raw base64 PNG/JPEG, no data: prefix>",
  "masked_indices": [74, 75, 88, 89],
  "topk": 3
}
```

Response:
```json
{
  "masked_input_base64": "<PNG>",
  "mae_recon_base64":    "<PNG>",
  "unet_recon_base64":   "<PNG>",
  "mae_mse":   0.011391,
  "unet_mse":  0.015320,
  "better_model": "mae",
  "predictions": [
    {"label": "dog", "confidence": 0.9987}
  ],
  "device": "cuda",
  "latency_ms": 842
}
```

## Patch grid

Input is resized→256 (bicubic), center-cropped to 224×224, divided into a 14×14 grid of 16×16 patches (196 total, flat row-major indexing).

```
index = row * 14 + col
```

## CLI

```
.venv/bin/inference-cli --image <path> [options]
```

| Argument       | Default                | Description                                    |
|----------------|------------------------|------------------------------------------------|
| `--image`      | *(required)*           | Path to input image                            |
| `--mask`       | `""`                   | Comma-separated flat patch indices             |
| `--weight-dir` | `weights`              | Directory with weight files                    |
| `--output-dir` | `inference_outputs`    | Output directory (created if needed)           |
| `--topk`       | `3`                    | Top-k predictions                              |
| `--device`     | `auto`                 | `auto` / `cpu` / `cuda` / `mps`                |

Writes three PNGs: `<stem>_masked_input.png`, `<stem>_mae_recon.png`, `<stem>_unet_recon.png`.

## Classes (Animals-10)

`dog`, `horse`, `elephant`, `butterfly`, `chicken`, `cat`, `cow`, `sheep`, `spider`, `squirrel`.
