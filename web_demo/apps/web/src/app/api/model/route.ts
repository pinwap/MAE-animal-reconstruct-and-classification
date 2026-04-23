export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const INFERENCE_URL = process.env.INFERENCE_SERVICE_URL ?? "http://localhost:8000";
const INFERENCE_TIMEOUT_MS = 120_000;

interface ProxyRequestBody {
  imageBase64?: string;
  maskedPatches?: number[];
  topk?: number;
}

interface UpstreamResponse {
  masked_input_base64: string;
  mae_recon_base64: string;
  unet_recon_base64: string;
  mae_mse: number;
  unet_mse: number;
  better_model: "mae" | "unet";
  predictions: { label: string; confidence: number }[];
  device: string;
  latency_ms: number;
}

export async function POST(request: Request) {
  let body: ProxyRequestBody;
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const { imageBase64, maskedPatches, topk } = body;
  if (!imageBase64 || !Array.isArray(maskedPatches)) {
    return Response.json(
      { error: "Missing imageBase64 or maskedPatches" },
      { status: 400 },
    );
  }

  try {
    const upstream = await fetch(`${INFERENCE_URL}/infer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_base64: imageBase64,
        masked_indices: maskedPatches,
        topk: topk ?? 3,
      }),
      signal: AbortSignal.timeout(INFERENCE_TIMEOUT_MS),
    });

    if (!upstream.ok) {
      const text = await upstream.text();
      return Response.json(
        { error: "Inference service error", status: upstream.status, detail: text },
        { status: 502 },
      );
    }

    const data = (await upstream.json()) as UpstreamResponse;

    return Response.json({
      maskedInputBase64: data.masked_input_base64,
      maeReconBase64:    data.mae_recon_base64,
      unetReconBase64:   data.unet_recon_base64,
      maeMse:            data.mae_mse,
      unetMse:           data.unet_mse,
      betterModel:       data.better_model,
      predictions:       data.predictions,
      device:            data.device,
      latencyMs:         data.latency_ms,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    const isTimeout = err instanceof Error && err.name === "TimeoutError";
    return Response.json(
      { error: isTimeout ? "Inference service timed out" : "Inference service unreachable", detail: message },
      { status: isTimeout ? 504 : 502 },
    );
  }
}
