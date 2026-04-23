"use client";

import { useEffect, useRef, useState } from "react";
import { ImageSource, Sample } from "@/types";
import { MAE_SAMPLES, renderSample } from "@/lib/samples";
import { squareCropCanvas } from "@/lib/image";

interface UploadStepProps {
  onImageReady: (src: ImageSource) => void;
}

function createImageCanvas(w: number, h: number): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}

export default function UploadStep({ onImageReady }: UploadStepProps) {
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function handleSample(sample: Sample) {
    const raw = createImageCanvas(448, 448);
    await renderSample(raw, sample);
    const canvas = squareCropCanvas(raw);
    onImageReady({ kind: "sample", id: sample.id, name: sample.name, canvas });
  }

  function handleUpload(file: File) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const canvas = squareCropCanvas(img);
        onImageReady({ kind: "upload", id: "upload", name: file.name, canvas });
      };
      img.src = e.target?.result as string;
    };
    reader.readAsDataURL(file);
  }

  function openFilePicker() {
    fileInputRef.current?.click();
  }

  return (
    <div
      className="fade-in"
      style={{
        background: "var(--bg-elev)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-lg)",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          padding: "14px 20px",
          borderBottom: "1px solid var(--border)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 16,
        }}
      >
        <h2 style={{ fontSize: 19, fontWeight: 600, margin: 0, letterSpacing: "-0.005em" }}>
          Input image
        </h2>
        <div style={{ color: "var(--fg-muted)", fontSize: 12, fontFamily: "var(--mono)" }}>
          step 01 / 03 · expected 224×224 RGB
        </div>
      </div>

      <div style={{ padding: 20 }}>
        <div
          role="button"
          tabIndex={0}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith("image/")) handleUpload(file);
          }}
          onClick={openFilePicker}
          onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openFilePicker(); } }}
          style={{
            border: `1.5px dashed ${dragging ? "var(--accent)" : "var(--border-strong)"}`,
            borderRadius: "var(--radius-lg)",
            padding: "48px 24px",
            textAlign: "center",
            background: dragging ? "var(--accent-soft)" : "var(--bg-sunken)",
            cursor: "pointer",
            transition: "border-color .15s, background .15s",
          }}
        >
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              width: 36,
              height: 36,
              border: "1.5px solid var(--fg-muted)",
              borderRadius: 6,
              color: "var(--fg-muted)",
            }}
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M12 19V5" />
              <path d="M5 12l7-7 7 7" />
            </svg>
          </div>
          <h3 style={{ margin: "12px 0 4px", fontSize: 15, fontWeight: 600 }}>
            Drop an animal photo, or click to browse
          </h3>
          <p
            style={{
              margin: 0,
              color: "var(--fg-muted)",
              fontFamily: "var(--mono)",
              fontSize: 11,
            }}
          >
            JPG · PNG · WebP · max 10MB · resized to 224×224
          </p>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleUpload(file);
              e.target.value = "";
            }}
          />
        </div>

        <div style={{ marginTop: 32 }}>
          <h4
            style={{
              fontFamily: "var(--mono)",
              fontSize: 11,
              fontWeight: 500,
              color: "var(--fg-muted)",
              textTransform: "uppercase",
              letterSpacing: "0.05em",
              margin: "0 0 12px",
            }}
          >
            ── Sample Images ──
          </h4>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(6, 1fr)",
              gap: 12,
            }}
          >
            {MAE_SAMPLES.map((sample) => (
              <SampleCard key={sample.id} sample={sample} onClick={() => handleSample(sample)} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function SampleCard({ sample, onClick }: { sample: Sample; onClick: () => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (canvasRef.current) {
      canvasRef.current.width = 160;
      canvasRef.current.height = 160;
      renderSample(canvasRef.current, sample);
    }
  }, [sample]);

  return (
    <div
      onClick={onClick}
      style={{
        border: "1px solid var(--border)",
        borderRadius: "var(--radius)",
        background: "var(--bg-sunken)",
        overflow: "hidden",
        cursor: "pointer",
        transition: "transform .15s, border-color .15s",
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLDivElement).style.borderColor = "var(--accent)";
        (e.currentTarget as HTMLDivElement).style.transform = "translateY(-1px)";
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLDivElement).style.borderColor = "var(--border)";
        (e.currentTarget as HTMLDivElement).style.transform = "none";
      }}
    >
      <canvas
        ref={canvasRef}
        style={{ display: "block", width: "100%", aspectRatio: "1/1" }}
      />
      <div
        style={{
          padding: "8px 10px",
          fontFamily: "var(--mono)",
          fontSize: 10,
          color: "var(--fg-muted)",
          display: "flex",
          justifyContent: "space-between",
          borderTop: "1px solid var(--border)",
        }}
      >
        <span style={{ color: "var(--fg)" }}>{sample.id}</span>
        <span>{sample.name}</span>
      </div>
    </div>
  );
}
