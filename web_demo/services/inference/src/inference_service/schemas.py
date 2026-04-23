from __future__ import annotations

from pydantic import BaseModel, Field


class InferRequest(BaseModel):
    image_base64: str = Field(..., description="Raw base64 PNG/JPEG bytes (no data: prefix).")
    masked_indices: list[int] = Field(default_factory=list, description="Flat patch indices over a 14x14 grid.")
    topk: int = Field(3, ge=1, le=10)


class Prediction(BaseModel):
    label: str
    confidence: float


class InferResponse(BaseModel):
    masked_input_base64: str
    mae_recon_base64: str
    unet_recon_base64: str
    mae_mse: float
    unet_mse: float
    better_model: str
    predictions: list[Prediction]
    device: str
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: bool
