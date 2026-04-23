"use client";

import { AppStep } from "@/types";

interface StepperProps {
  step: AppStep;
  canGoToMask: boolean;
  canGoToResult: boolean;
  onGoTo: (step: AppStep) => void;
}

const STEPS = ["Upload", "Mask", "Reconstruct"];

export default function Stepper({ step, canGoToMask, canGoToResult, onGoTo }: StepperProps) {
  function handleClick(i: number) {
    if (i === 0) onGoTo(0);
    else if (i === 1 && canGoToMask) onGoTo(1);
    else if (i === 2 && canGoToResult) onGoTo(2);
  }

  const stateOf = (i: number) => (i < step ? "done" : i === step ? "active" : "idle");

  return (
    <div
      style={{
        maxWidth: 1240,
        margin: "0 auto",
        padding: "24px 32px 8px",
        display: "flex",
        alignItems: "center",
      }}
    >
      {STEPS.map((label, i) => {
        const s = stateOf(i);
        return (
          <div key={i} style={{ display: "contents" }}>
            <div
              onClick={() => handleClick(i)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                color: s === "active" ? "var(--fg)" : s === "done" ? "var(--fg-muted)" : "var(--fg-subtle)",
                fontFamily: "var(--mono)",
                fontSize: 12,
                cursor: "pointer",
                userSelect: "none",
              }}
            >
              <div
                style={{
                  width: 22,
                  height: 22,
                  borderRadius: "50%",
                  border: s === "active"
                    ? "1px solid var(--fg)"
                    : s === "done"
                    ? "1px solid var(--accent)"
                    : "1px solid var(--border-strong)",
                  display: "grid",
                  placeItems: "center",
                  fontSize: 10,
                  background: s === "active"
                    ? "var(--fg)"
                    : s === "done"
                    ? "var(--accent)"
                    : "var(--bg-elev)",
                  color: s === "active"
                    ? "var(--bg-elev)"
                    : s === "done"
                    ? "var(--accent-fg)"
                    : "inherit",
                }}
              >
                {s === "done" ? "✓" : String(i + 1).padStart(2, "0")}
              </div>
              <div>{label}</div>
            </div>
            {i < STEPS.length - 1 && (
              <div
                style={{
                  flex: "0 0 48px",
                  height: 1,
                  background: i < step ? "var(--accent)" : "var(--border)",
                  margin: "0 12px",
                }}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
