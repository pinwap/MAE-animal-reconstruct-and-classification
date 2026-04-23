"use client";

import { useEffect, useRef, useState } from "react";
import { ImageSource, InferenceResponse } from "@/types";
import { canvasToBase64Png } from "@/lib/image";

const STAGES = [
  "› encoding image",
  "› sending to inference service",
  "› running MAE reconstruction",
  "› running U-Net inpainting",
  "› classifying",
];

interface InferenceLoaderProps {
  imageSource: ImageSource;
  masked: Set<number>;
  onDone: (result: InferenceResponse) => void;
  onError: (message: string) => void;
}

export default function InferenceLoader({ imageSource, masked, onDone, onError }: InferenceLoaderProps) {
  const [stage, setStage] = useState(0);
  const cancelledRef = useRef(false);

  useEffect(() => {
    cancelledRef.current = false;

    const tick = window.setInterval(() => {
      setStage((s) => (s < STAGES.length - 1 ? s + 1 : s));
    }, 600);

    (async () => {
      try {
        const imageBase64 = canvasToBase64Png(imageSource.canvas);
        const maskedPatches = Array.from(masked);

        const res = await fetch("/api/model", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ imageBase64, maskedPatches }),
        });

        if (cancelledRef.current) return;

        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body?.error ?? `HTTP ${res.status}`);
        }

        const data = (await res.json()) as InferenceResponse;
        window.clearInterval(tick);
        setStage(STAGES.length);
        if (!cancelledRef.current) onDone(data);
      } catch (err) {
        window.clearInterval(tick);
        if (!cancelledRef.current) {
          onError(err instanceof Error ? err.message : String(err));
        }
      }
    })();

    return () => {
      cancelledRef.current = true;
      window.clearInterval(tick);
    };
  }, [imageSource, masked, onDone, onError]);

  const progress = (stage / STAGES.length) * 100;
  const currentLog = STAGES[Math.min(stage, STAGES.length - 1)];

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.5)",
        display: "grid",
        placeItems: "center",
        backdropFilter: "blur(2px)",
        zIndex: 50,
      }}
    >
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 14,
          alignItems: "center",
          background: "var(--bg-elev)",
          padding: "20px 28px",
          borderRadius: "var(--radius)",
          border: "1px solid var(--border)",
          fontFamily: "var(--mono)",
          fontSize: 11,
          color: "var(--fg)",
          minWidth: 280,
        }}
      >
        <div>Running MAE + U-Net + classifier</div>
        <div
          style={{
            width: "100%",
            height: 3,
            background: "var(--border)",
            borderRadius: 2,
            overflow: "hidden",
          }}
        >
          <div
            style={{
              height: "100%",
              width: `${progress}%`,
              background: "var(--accent)",
              transition: "width .3s linear",
            }}
          />
        </div>
        <div
          style={{
            fontSize: 10,
            color: "var(--fg-muted)",
            alignSelf: "flex-start",
            minHeight: 14,
          }}
        >
          {currentLog}
        </div>
      </div>
    </div>
  );
}
