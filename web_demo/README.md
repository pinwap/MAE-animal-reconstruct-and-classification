# APP_DEEP

MAE + U-Net patch inpainting and Animals-10 classification — full-stack demo.

```
APP_DEEP/
├── apps/web/               Next.js 16 frontend (upload → mask → inspect)
└── services/inference/     FastAPI service + CLI wrapping PyTorch models
```

## Prerequisites

- Node.js 20+ and npm
- Python 3.10+ (deps installed into a local `.venv` via `make install`)
- Three weight files under `services/inference/weights/`:
  - `mae_reconstruction.pt` — ViT-MAE base encoder + decoder
  - `unet_best.pt`          — U-Net inpainting
  - `mae_cls_best.pth`      — ViT-Base encoder + MLP classifier

## Quickstart

```bash
make install          # npm install + python venv + pip install

# Terminal 1 — inference service
make dev-inference    # http://localhost:8000

# Terminal 2 — web app
make dev-web          # http://localhost:3000
```

Then open http://localhost:3000 and walk through the wizard.

### Configuration

`apps/web/.env.local`:
```
INFERENCE_SERVICE_URL=http://localhost:8000
```

`services/inference/.env`:
```
DEVICE=auto            # auto | cpu | cuda | mps
WEIGHT_DIR=weights
PORT=8000
```

> **macOS:** `auto` picks CPU (CUDA/MPS fallbacks are opt-in). Use `DEVICE=mps` only if you've confirmed your torch build handles ViTMAE ops.

## CLI

```bash
make infer IMAGE=path/to/dog.jpg MASK=74,75,88,89
```

Or directly:
```bash
services/inference/.venv/bin/inference-cli \
  --image path/to/dog.jpg --mask 74,75,88,89
```

Writes `<stem>_masked_input.png`, `<stem>_mae_recon.png`, `<stem>_unet_recon.png` to `inference_outputs/`.

## Architecture

```
Browser
  │  canvas → resize+crop 224 → base64 PNG
  ▼
Next.js /api/model          (proxy, 120s timeout)
  │
  ▼
FastAPI /infer              (models loaded once at startup)
  │
  ├─ MAE reconstruct   (drops masked tokens, decoder fills)
  ├─ U-Net inpaint     (sees black squares, inpaints)
  └─ Classifier        (top-k from MAE recon)
```

Both reconstructions are returned along with their masked-MSE so the UI can compare.
