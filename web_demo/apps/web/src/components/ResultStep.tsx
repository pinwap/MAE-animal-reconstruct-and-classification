"use client";

import { useEffect, useRef } from "react";
import { ImageSource, InferenceResponse } from "@/types";
import { displayName } from "@/lib/classes";

const PATCH_N = 14;
const TOTAL_PATCHES = PATCH_N * PATCH_N;

interface ResultStepProps {
  imageSource: ImageSource;
  masked: Set<number>;
  result: InferenceResponse;
  onEditMask: () => void;
  onNewImage: () => void;
}

export default function ResultStep({
  imageSource,
  masked,
  result,
  onEditMask,
  onNewImage,
}: ResultStepProps) {
  const originalRef = useRef<HTMLCanvasElement>(null);
  const maskPct = Math.round((masked.size / TOTAL_PATCHES) * 100);

  useEffect(() => {
    const src = imageSource.canvas;
    if (originalRef.current) {
      originalRef.current.width = src.width;
      originalRef.current.height = src.height;
      originalRef.current.getContext("2d")!.drawImage(src, 0, 0);
    }
  }, [imageSource]);

  const maeBetter = result.betterModel === "mae";

  return (
    <div
      className="fade-in"
      style={{
        display: "grid",
        gridTemplateColumns: "2fr 1fr",
        gap: 20,
      }}
    >
      {/* Reconstruction panel */}
      <div
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
          }}
        >
          <h2 style={{ fontSize: 19, fontWeight: 600, margin: 0 }}>Reconstruction</h2>
          <div style={{ color: "var(--fg-muted)", fontSize: 12, fontFamily: "var(--mono)" }}>
            step 03 / 03 · {masked.size}/{TOTAL_PATCHES} patches masked
          </div>
        </div>

        <div style={{ padding: 20 }}>
          {/* 2x2 quad */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 12 }}>
            <ImageTile label="Original" tag="x">
              <canvas
                ref={originalRef}
                style={{ display: "block", width: "100%", height: "100%" }}
              />
            </ImageTile>
            <ImageTile label="Masked input" tag="x ⊙ M">
              <Img b64={result.maskedInputBase64} />
            </ImageTile>
            <ImageTile
              label="MAE reconstruction"
              tag="x̂ MAE"
              metric={`MSE ${result.maeMse.toFixed(6)}`}
              highlight={maeBetter}
            >
              <Img b64={result.maeReconBase64} />
            </ImageTile>
            <ImageTile
              label="U-Net reconstruction"
              tag="x̂ UNet"
              metric={`MSE ${result.unetMse.toFixed(6)}`}
              highlight={!maeBetter}
            >
              <Img b64={result.unetReconBase64} />
            </ImageTile>
          </div>

          {/* Action bar */}
          <div
            style={{
              marginTop: 20,
              paddingTop: 20,
              borderTop: "1px solid var(--border)",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              gap: 12,
            }}
          >
            <div style={{ fontFamily: "var(--mono)", fontSize: 14, color: "var(--fg-muted)" }}>
              better model (lower masked MSE):{" "}
              <span style={{ color: "var(--accent)", fontWeight: 700, fontSize: 16 }}>
                {result.betterModel.toUpperCase()}
              </span>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <Btn ghost onClick={onEditMask}>← Edit mask</Btn>
              <Btn onClick={onNewImage}>New image</Btn>
            </div>
          </div>
        </div>
      </div>

      {/* Classifier panel */}
      <div
        style={{
          background: "var(--bg-elev)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-lg)",
          overflow: "hidden",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <div
          style={{
            padding: "14px 20px",
            borderBottom: "1px solid var(--border)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <h2 style={{ fontSize: 13, fontWeight: 600, margin: 0 }}>
            Classification · top-{result.predictions.length}
          </h2>
          <div style={{ color: "var(--fg-muted)", fontSize: 12, fontFamily: "var(--mono)" }}>
            softmax
          </div>
        </div>

        <div style={{ flex: 1 }}>
          {result.predictions.map((pred, i) => {
            const isTop = i === 0;
            return (
              <div
                key={pred.label}
                style={{
                  display: "grid",
                  gridTemplateColumns: "24px 1fr auto",
                  gap: 12,
                  alignItems: "center",
                  padding: "10px 20px",
                  borderBottom: i < result.predictions.length - 1 ? "1px solid var(--border)" : "none",
                }}
              >
                <div
                  style={{
                    fontFamily: "var(--mono)",
                    fontSize: 10,
                    color: isTop ? "var(--accent)" : "var(--fg-subtle)",
                    fontWeight: isTop ? 600 : 400,
                  }}
                >
                  {String(i + 1).padStart(2, "0")}
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  <div style={{ fontSize: 13, fontWeight: 500 }}>{displayName(pred.label)}</div>
                  <div
                    style={{
                      fontFamily: "var(--mono)",
                      fontSize: 10,
                      color: "var(--fg-subtle)",
                    }}
                  >
                    {pred.label}
                  </div>
                  <div
                    style={{
                      height: 4,
                      background: "var(--border)",
                      borderRadius: 2,
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        height: "100%",
                        width: `${Math.max(0, Math.min(1, pred.confidence)) * 100}%`,
                        background: isTop ? "var(--accent)" : "var(--fg-muted)",
                        transition: "width .4s cubic-bezier(.2,.9,.2,1)",
                      }}
                    />
                  </div>
                </div>
                <div
                  style={{
                    fontFamily: "var(--mono)",
                    fontSize: 12,
                    fontVariantNumeric: "tabular-nums",
                    minWidth: 48,
                    textAlign: "right",
                  }}
                >
                  {(pred.confidence * 100).toFixed(2)}%
                </div>
              </div>
            );
          })}
        </div>

        {/* Metrics */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(4, 1fr)",
            borderTop: "1px solid var(--border)",
          }}
        >
          {[
            { k: "Device",  v: result.device },
            { k: "Latency", v: `${result.latencyMs}ms` },
            { k: "Mask",    v: `${maskPct}%` },
            { k: "Better",  v: result.betterModel.toUpperCase() },
          ].map(({ k, v }, i, arr) => (
            <div
              key={k}
              style={{
                padding: "12px 20px",
                borderRight: i < arr.length - 1 ? "1px solid var(--border)" : "none",
              }}
            >
              <div
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: 10,
                  color: "var(--fg-muted)",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                }}
              >
                {k}
              </div>
              <div
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: 14,
                  marginTop: 2,
                  fontVariantNumeric: "tabular-nums",
                }}
              >
                {v}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function ImageTile({
  label,
  tag,
  metric,
  highlight,
  children,
}: {
  label: string;
  tag: string;
  metric?: string;
  highlight?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div
        style={{
          fontFamily: "var(--mono)",
          fontSize: 12,
          color: "var(--fg-muted)",
          textTransform: "uppercase",
          letterSpacing: "0.05em",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <span
          style={{
            color: highlight ? "var(--accent)" : "var(--fg-muted)",
            fontWeight: highlight ? 700 : 500,
          }}
        >
          {label}
          {highlight && " ★"}
        </span>
        <span
          style={{
            background: "var(--bg-sunken)",
            color: "var(--fg-muted)",
            padding: "2px 7px",
            borderRadius: 3,
            fontSize: 11,
            border: "1px solid var(--border)",
          }}
        >
          {tag}
        </span>
      </div>
      <div
        style={{
          aspectRatio: "1/1",
          background: "var(--bg-sunken)",
          borderRadius: "var(--radius)",
          overflow: "hidden",
          position: "relative",
          border: `1px solid ${highlight ? "var(--accent)" : "var(--border)"}`,
        }}
      >
        {children}
      </div>
      {metric && (
        <div
          style={{
            fontFamily: "var(--mono)",
            fontSize: 14,
            fontWeight: highlight ? 700 : 500,
            fontVariantNumeric: "tabular-nums",
            color: highlight ? "var(--accent)" : "var(--fg)",
          }}
        >
          {metric}
        </div>
      )}
    </div>
  );
}

function Img({ b64 }: { b64: string }) {
  return (
    <img
      src={`data:image/png;base64,${b64}`}
      alt=""
      style={{ display: "block", width: "100%", height: "100%", imageRendering: "pixelated" }}
    />
  );
}

function Btn({
  children,
  onClick,
  ghost,
}: {
  children: React.ReactNode;
  onClick?: () => void;
  ghost?: boolean;
}) {
  const base: React.CSSProperties = {
    fontFamily: "var(--mono)",
    fontSize: 12,
    padding: "8px 14px",
    borderRadius: "var(--radius)",
    border: ghost ? "1px solid var(--border)" : "1px solid var(--border-strong)",
    background: ghost ? "transparent" : "var(--bg-elev)",
    color: ghost ? "var(--fg-muted)" : "var(--fg)",
    cursor: "pointer",
    display: "inline-flex",
    alignItems: "center",
    gap: 8,
    transition: "background .12s, border-color .12s, color .12s",
  };
  return (
    <button onClick={onClick} style={base}>
      {children}
    </button>
  );
}
