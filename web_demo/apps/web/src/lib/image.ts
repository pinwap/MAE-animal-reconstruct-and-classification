export const MODEL_INPUT_SIZE = 224;

/**
 * Resize + center-crop an image to a square canvas matching the backend's
 * preprocessing (shorter edge → `resizeEdge`, then center-crop to `outSize`).
 * Default values mirror the Python pipeline: resize 256 bicubic, crop 224.
 */
export function squareCropCanvas(
  source: CanvasImageSource & { width: number; height: number },
  outSize = MODEL_INPUT_SIZE,
  resizeEdge = 256,
): HTMLCanvasElement {
  const sw = (source as HTMLImageElement | HTMLCanvasElement).width;
  const sh = (source as HTMLImageElement | HTMLCanvasElement).height;
  const scale = resizeEdge / Math.min(sw, sh);
  const rw = Math.round(sw * scale);
  const rh = Math.round(sh * scale);

  const scaled = document.createElement("canvas");
  scaled.width = rw;
  scaled.height = rh;
  scaled.getContext("2d")!.drawImage(source, 0, 0, rw, rh);

  const out = document.createElement("canvas");
  out.width = outSize;
  out.height = outSize;
  const ctx = out.getContext("2d")!;
  ctx.imageSmoothingQuality = "high";
  const sx = Math.max(0, Math.floor((rw - outSize) / 2));
  const sy = Math.max(0, Math.floor((rh - outSize) / 2));
  ctx.drawImage(scaled, sx, sy, outSize, outSize, 0, 0, outSize, outSize);
  return out;
}

export function canvasToBase64Png(canvas: HTMLCanvasElement): string {
  const dataUrl = canvas.toDataURL("image/png");
  const comma = dataUrl.indexOf(",");
  return comma >= 0 ? dataUrl.slice(comma + 1) : dataUrl;
}
