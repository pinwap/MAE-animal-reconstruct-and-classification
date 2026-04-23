from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .loaders import resolve_device
from .pipeline import Models, load_all, run_once, warmup
from .schemas import HealthResponse, InferRequest, InferResponse, Prediction
from .utils import b64_to_pil, pil_to_b64_png

logger = logging.getLogger("inference_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_STATE: dict[str, Models | None] = {"models": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    weight_dir = Path(os.environ.get("WEIGHT_DIR", "weights"))
    device_pref = os.environ.get("DEVICE", "auto")
    device = resolve_device(device_pref)
    logger.info("loading models from %s on device=%s", weight_dir, device)
    t0 = time.perf_counter()
    models = load_all(weight_dir, device)
    logger.info("warmup...")
    warmup(models)
    logger.info("ready in %.1fs", time.perf_counter() - t0)
    _STATE["models"] = models
    try:
        yield
    finally:
        _STATE["models"] = None


app = FastAPI(title="MAE+UNet Inference Service", version="0.1.0", lifespan=lifespan)


def _models() -> Models:
    m = _STATE["models"]
    if m is None:
        raise HTTPException(status_code=503, detail="models not loaded")
    return m


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    m = _STATE["models"]
    return HealthResponse(
        status="ok" if m is not None else "loading",
        device=str(m.device) if m is not None else "unknown",
        models_loaded=m is not None,
    )


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest) -> InferResponse:
    models = _models()
    try:
        image = b64_to_pil(req.image_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not decode image_base64: {exc}") from exc

    t0 = time.perf_counter()
    result = await asyncio.to_thread(run_once, models, image, req.masked_indices, req.topk)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    return InferResponse(
        masked_input_base64 = pil_to_b64_png(result.masked_input),
        mae_recon_base64    = pil_to_b64_png(result.mae_recon),
        unet_recon_base64   = pil_to_b64_png(result.unet_recon),
        mae_mse             = result.mae_mse,
        unet_mse            = result.unet_mse,
        better_model        = result.better_model,
        predictions         = [Prediction(label=l, confidence=c) for l, c in result.predictions],
        device              = str(models.device),
        latency_ms          = latency_ms,
    )
