import { Sample } from "@/types";

export const MAE_SAMPLES: Sample[] = [
  { id: "cat",      name: "Cat",      photo: "/samples/cat.jpeg",     palette: ["#c9a273", "#8f6a3d", "#5e4528", "#2b2017"], accent: "#f2d9a4" },
  { id: "dog",      name: "Dog",      photo: "/samples/dog.jpeg",     palette: ["#d9c297", "#b09065", "#6d5332", "#352820"], accent: "#f4e4bd" },
  { id: "hourse",   name: "Horse",    photo: "/samples/hourse.png",   palette: ["#d6b38a", "#8b6a45", "#4a3522", "#1f1710"], accent: "#ead0a6" },
  { id: "squirrel", name: "Squirrel", photo: "/samples/squirrel.png", palette: ["#cf9a63", "#8b5a30", "#4d2e18", "#1c110a"], accent: "#f0cfa0" },
];

/**
 * Load the real photo from `sample.photo` onto the canvas.
 * Falls back to the procedural gradient if the file is missing.
 */
export function renderSample(canvas: HTMLCanvasElement, sample: Sample): Promise<void> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return resolve();
      const W = canvas.width;
      const H = canvas.height;
      ctx.clearRect(0, 0, W, H);
      const scale = Math.max(W / img.width, H / img.height);
      const dw = img.width * scale;
      const dh = img.height * scale;
      ctx.drawImage(img, (W - dw) / 2, (H - dh) / 2, dw, dh);
      resolve();
    };
    img.onerror = () => {
      drawSampleProcedural(canvas, sample);
      resolve();
    };
    img.src = sample.photo;
  });
}

export function drawSampleProcedural(canvas: HTMLCanvasElement, sample: Sample): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const W = canvas.width;
  const H = canvas.height;
  const [c1, c2, c3, c4] = sample.palette;

  const g = ctx.createLinearGradient(0, 0, W, H);
  g.addColorStop(0, c1);
  g.addColorStop(0.55, c2);
  g.addColorStop(1, c3);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);

  ctx.globalAlpha = 0.14;
  for (let y = 0; y < H; y += 3) {
    ctx.fillStyle = y % 6 === 0 ? c4 : c1;
    ctx.fillRect(0, y, W, 1);
  }
  ctx.globalAlpha = 1;

  const cx = W * 0.52;
  const cy = H * 0.56;
  const rx = W * 0.36;
  const ry = H * 0.32;
  const blob = ctx.createRadialGradient(cx - rx * 0.2, cy - ry * 0.25, 4, cx, cy, rx * 1.2);
  blob.addColorStop(0, sample.accent);
  blob.addColorStop(0.5, c2);
  blob.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = blob;
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "rgba(0,0,0,0.45)";
  ctx.fillRect(8, H - 22, 120, 14);
  ctx.fillStyle = "rgba(255,255,255,0.85)";
  ctx.font = '9px "JetBrains Mono", ui-monospace, monospace';
  ctx.textBaseline = "middle";
  ctx.fillText("[sample] " + sample.id, 12, H - 15);
}

export const drawSample = drawSampleProcedural;
