"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ImageSource } from "@/types";

const PATCH_N = 14;
const TOTAL_PATCHES = PATCH_N * PATCH_N;
const CAP = Math.floor(TOTAL_PATCHES * 0.75); // 147

interface MaskStepProps {
  imageSource: ImageSource;
  masked: Set<number>;
  onMaskedChange: (masked: Set<number>) => void;
  onBack: () => void;
  onRun: () => void;
}

export default function MaskStep({
  imageSource,
  masked,
  onMaskedChange,
  onBack,
  onRun,
}: MaskStepProps) {
  const sourceCanvasRef = useRef<HTMLCanvasElement>(null);
  const draggingRef = useRef(false);
  const dragModeRef = useRef<"add" | "remove">("add");

  const maskedPct = Math.round((masked.size / TOTAL_PATCHES) * 100);
  const isCapped = masked.size >= CAP;

  // Draw source image
  useEffect(() => {
    if (sourceCanvasRef.current) {
      const ctx = sourceCanvasRef.current.getContext("2d");
      if (ctx) {
        sourceCanvasRef.current.width = imageSource.canvas.width;
        sourceCanvasRef.current.height = imageSource.canvas.height;
        ctx.drawImage(imageSource.canvas, 0, 0);
      }
    }
  }, [imageSource]);

  // Keyboard shortcuts
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Enter" && masked.size > 0) onRun();
      if (e.key === "Escape") onBack();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [masked.size, onRun, onBack]);

  function togglePatch(idx: number, mode: "add" | "remove") {
    const next = new Set(masked);
    if (mode === "add") {
      if (next.size >= CAP || next.has(idx)) return;
      next.add(idx);
    } else {
      next.delete(idx);
    }
    onMaskedChange(next);
  }

  function setRandomMask(ratio: number) {
    const target = Math.min(Math.floor(TOTAL_PATCHES * ratio), CAP);
    const indices = Array.from({ length: TOTAL_PATCHES }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    onMaskedChange(new Set(indices.slice(0, target)));
  }

  function handleInvert() {
    const inv = new Set<number>();
    for (let i = 0; i < TOTAL_PATCHES; i++) {
      if (!masked.has(i)) inv.add(i);
    }
    while (inv.size > CAP) {
      inv.delete(inv.values().next().value!);
    }
    onMaskedChange(inv);
  }

  function getIdxFromPointer(e: React.PointerEvent<HTMLDivElement>): number | null {
    const rect = e.currentTarget.getBoundingClientRect();
    const col = Math.floor(((e.clientX - rect.left) / rect.width) * PATCH_N);
    const row = Math.floor(((e.clientY - rect.top) / rect.height) * PATCH_N);
    if (col < 0 || col >= PATCH_N || row < 0 || row >= PATCH_N) return null;
    return row * PATCH_N + col;
  }

  function handleGridPointerDown(e: React.PointerEvent<HTMLDivElement>) {
    e.currentTarget.setPointerCapture(e.pointerId);
    draggingRef.current = true;
    const idx = getIdxFromPointer(e);
    if (idx === null) return;
    const shouldRemove = masked.has(idx) || e.shiftKey;
    dragModeRef.current = shouldRemove ? "remove" : "add";
    togglePatch(idx, dragModeRef.current);
  }

  function handleGridPointerMove(e: React.PointerEvent<HTMLDivElement>) {
    if (!draggingRef.current) return;
    const idx = getIdxFromPointer(e);
    if (idx === null) return;
    togglePatch(idx, dragModeRef.current);
  }

  function handlePointerUp() {
    draggingRef.current = false;
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
      {/* Panel head */}
      <div
        style={{
          padding: "14px 20px",
          borderBottom: "1px solid var(--border)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <h2 style={{ fontSize: 19, fontWeight: 600, margin: 0, letterSpacing: "-0.005em" }}>
          Mask patches
        </h2>
        <div style={{ color: "var(--fg-muted)", fontSize: 12, fontFamily: "var(--mono)" }}>
          step 02 / 03 · grid {PATCH_N}×{PATCH_N} · patch 16px
        </div>
      </div>

      <div style={{ padding: 20 }}>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 280px",
            gap: 20,
          }}
        >
          {/* Canvas area */}
          <div>
            <div
              style={{
                position: "relative",
                background: "var(--bg-sunken)",
                borderRadius: "var(--radius)",
                overflow: "hidden",
                aspectRatio: "1/1",
                userSelect: "none",
                touchAction: "none",
              }}
            >
              <canvas
                ref={sourceCanvasRef}
                style={{ display: "block", width: "100%", height: "100%" }}
              />
              {/* Patch grid */}
              <div
                onPointerDown={handleGridPointerDown}
                onPointerMove={handleGridPointerMove}
                onPointerUp={handlePointerUp}
                style={{
                  position: "absolute",
                  inset: 0,
                  display: "grid",
                  gridTemplate: `repeat(${PATCH_N}, 1fr) / repeat(${PATCH_N}, 1fr)`,
                  cursor: "crosshair",
                }}
              >
                {Array.from({ length: TOTAL_PATCHES }, (_, i) => (
                  <div
                    key={i}
                    style={{
                      borderRight: "1px solid rgba(255,255,255,0.06)",
                      borderBottom: "1px solid rgba(255,255,255,0.06)",
                      background: masked.has(i) ? "var(--mask)" : undefined,
                      transition: "background .08s",
                    }}
                    onMouseEnter={(e) => {
                      if (!masked.has(i)) {
                        (e.currentTarget as HTMLDivElement).style.background = "var(--mask-hover)";
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!masked.has(i)) {
                        (e.currentTarget as HTMLDivElement).style.background = "transparent";
                      }
                    }}
                  />
                ))}
              </div>
              {/* Overlay label */}
              <div
                style={{
                  position: "absolute",
                  top: 10,
                  left: 10,
                  fontFamily: "var(--mono)",
                  fontSize: 10,
                  color: "rgba(255,255,255,0.85)",
                  background: "rgba(0,0,0,0.55)",
                  padding: "4px 8px",
                  borderRadius: 4,
                  backdropFilter: "blur(4px)",
                  pointerEvents: "none",
                }}
              >
                {imageSource.name} · {imageSource.id}
              </div>
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
              <div
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: 11,
                  color: "var(--fg-muted)",
                }}
              >
                click or drag to toggle · shift-drag to erase
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <Btn ghost onClick={onBack}>← Back</Btn>
                <Btn primary disabled={masked.size === 0} onClick={onRun}>
                  Run MAE{" "}
                  <span
                    style={{
                      fontSize: 10,
                      color: "var(--accent)",
                      background: "var(--accent-soft)",
                      padding: "1px 4px",
                      borderRadius: 3,
                      border: "1px solid var(--accent)",
                    }}
                  >
                    ↵
                  </span>
                </Btn>
              </div>
            </div>
          </div>

          {/* Sidebar */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {/* Mask ratio stat */}
            <div
              style={{
                border: `1px solid ${isCapped ? "var(--warn)" : "var(--border)"}`,
                borderRadius: "var(--radius)",
                padding: 14,
                background: "var(--bg-sunken)",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontFamily: "var(--mono)",
                  fontSize: 10,
                  color: "var(--fg-muted)",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  marginBottom: 10,
                }}
              >
                <span>Mask ratio</span>
                <span style={{ color: "var(--fg-subtle)" }}>cap 75%</span>
              </div>
              <div
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: 24,
                  fontWeight: 500,
                  letterSpacing: "-0.02em",
                  lineHeight: 1,
                  color: isCapped ? "var(--warn)" : "var(--fg)",
                }}
              >
                {maskedPct}
                <span style={{ fontSize: 14, color: "var(--fg-muted)", marginLeft: 2 }}>%</span>
              </div>
              <div
                style={{
                  marginTop: 10,
                  height: 4,
                  background: "var(--border)",
                  borderRadius: 2,
                  overflow: "hidden",
                  position: "relative",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${maskedPct}%`,
                    background: isCapped ? "var(--warn)" : "var(--accent)",
                    transition: "width .15s",
                  }}
                />
                <div
                  style={{
                    position: "absolute",
                    top: -2,
                    bottom: -2,
                    width: 2,
                    background: "var(--warn)",
                    left: "75%",
                  }}
                />
              </div>
              {isCapped && (
                <div
                  style={{
                    fontFamily: "var(--mono)",
                    fontSize: 10,
                    color: "var(--warn)",
                    marginTop: 8,
                  }}
                >
                  ! Cap reached — unmask to add more
                </div>
              )}
            </div>

            {/* Patch count stat */}
            <div
              style={{
                border: "1px solid var(--border)",
                borderRadius: "var(--radius)",
                padding: 14,
                background: "var(--bg-sunken)",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontFamily: "var(--mono)",
                  fontSize: 10,
                  color: "var(--fg-muted)",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  marginBottom: 10,
                }}
              >
                <span>Patches</span>
                <span style={{ color: "var(--fg-subtle)" }}>of {TOTAL_PATCHES}</span>
              </div>
              <div
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: 24,
                  fontWeight: 500,
                  letterSpacing: "-0.02em",
                  lineHeight: 1,
                }}
              >
                {masked.size}
              </div>
            </div>

            {/* Random mask slider */}
            <div>
              <label
                style={{
                  display: "block",
                  fontFamily: "var(--mono)",
                  fontSize: 10,
                  color: "var(--fg-muted)",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  marginBottom: 8,
                }}
              >
                Random mask
              </label>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <input
                  type="range"
                  min={0}
                  max={75}
                  step={1}
                  value={Math.min(maskedPct, 75)}
                  onChange={(e) => setRandomMask(Number(e.target.value) / 100)}
                  style={{ flex: 1, accentColor: "var(--accent)" }}
                />
                <span
                  style={{
                    fontFamily: "var(--mono)",
                    fontSize: 12,
                    minWidth: 34,
                    textAlign: "right",
                  }}
                >
                  {Math.min(maskedPct, 75)}%
                </span>
              </div>
            </div>

            {/* Action buttons */}
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <Btn
                onClick={() => {
                  const cur = masked.size / TOTAL_PATCHES;
                  setRandomMask(Math.min(cur > 0 ? cur : 0.6, 0.75));
                }}
              >
                Shuffle
              </Btn>
              <Btn onClick={() => onMaskedChange(new Set())}>Clear</Btn>
              <Btn onClick={handleInvert}>Invert</Btn>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Btn({
  children,
  onClick,
  disabled,
  primary,
  ghost,
}: {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  primary?: boolean;
  ghost?: boolean;
}) {
  const base: React.CSSProperties = {
    fontFamily: "var(--mono)",
    fontSize: 12,
    padding: "8px 14px",
    borderRadius: "var(--radius)",
    border: "1px solid var(--border-strong)",
    background: "var(--bg-elev)",
    color: "var(--fg)",
    cursor: disabled ? "not-allowed" : "pointer",
    display: "inline-flex",
    alignItems: "center",
    gap: 8,
    opacity: disabled ? 0.45 : 1,
    transition: "background .12s, border-color .12s, color .12s",
  };
  if (primary) {
    base.background = "var(--accent)";
    base.color = "var(--accent-fg)";
    base.borderColor = "var(--accent)";
  }
  if (ghost) {
    base.background = "transparent";
    base.borderColor = "var(--border)";
    base.color = "var(--fg-muted)";
  }
  return (
    <button onClick={disabled ? undefined : onClick} style={base}>
      {children}
    </button>
  );
}
