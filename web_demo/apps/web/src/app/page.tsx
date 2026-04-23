"use client";

import { useCallback, useEffect, useState } from "react";
import { AppStep, ImageSource, InferenceResponse } from "@/types";
import Header from "@/components/Header";
import Stepper from "@/components/Stepper";
import UploadStep from "@/components/UploadStep";
import MaskStep from "@/components/MaskStep";
import ResultStep from "@/components/ResultStep";
import InferenceLoader from "@/components/InferenceLoader";

export default function Home() {
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [step, setStep] = useState<AppStep>(0);
  const [imageSource, setImageSource] = useState<ImageSource | null>(null);
  const [masked, setMasked] = useState<Set<number>>(new Set());
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const saved = localStorage.getItem("mae-theme") as "light" | "dark" | null;
    if (saved) {
      setTheme(saved);
      document.documentElement.setAttribute("data-theme", saved);
    }
  }, []);

  function toggleTheme() {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("mae-theme", next);
  }

  function goTo(s: AppStep) {
    setStep(s);
  }

  function handleImageReady(src: ImageSource) {
    setImageSource(src);
    setMasked(new Set());
    setResult(null);
    goTo(1);
  }

  function handleRun() {
    if (masked.size === 0 || running) return;
    setError(null);
    setRunning(true);
  }

  const handleInferenceDone = useCallback((r: InferenceResponse) => {
    setResult(r);
    setRunning(false);
    goTo(2);
  }, []);

  const handleInferenceError = useCallback((msg: string) => {
    setRunning(false);
    setError(msg);
  }, []);

  function handleEditMask() {
    goTo(1);
  }

  function handleNewImage() {
    setImageSource(null);
    setMasked(new Set());
    setResult(null);
    setError(null);
    setRunning(false);
    goTo(0);
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        background: "var(--bg)",
      }}
    >
      <Header
        theme={theme}
        onToggleTheme={toggleTheme}
        onGoToUpload={handleNewImage}
      />

      <Stepper
        step={step}
        canGoToMask={!!imageSource}
        canGoToResult={!!imageSource && !!result}
        onGoTo={goTo}
      />

      <main
        style={{
          flex: 1,
          maxWidth: 1240,
          margin: "0 auto",
          padding: "24px 32px 64px",
          width: "100%",
        }}
      >
        {error && (
          <div
            role="alert"
            style={{
              marginBottom: 16,
              padding: "10px 14px",
              border: "1px solid var(--warn, #c0392b)",
              background: "rgba(192, 57, 43, 0.08)",
              borderRadius: "var(--radius)",
              fontFamily: "var(--mono)",
              fontSize: 12,
              color: "var(--warn, #c0392b)",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              gap: 12,
            }}
          >
            <span>Inference failed: {error}</span>
            <button
              onClick={() => setError(null)}
              style={{
                background: "transparent",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius)",
                fontFamily: "var(--mono)",
                fontSize: 11,
                padding: "4px 10px",
                color: "var(--fg-muted)",
                cursor: "pointer",
              }}
            >
              dismiss
            </button>
          </div>
        )}

        {step === 0 && <UploadStep onImageReady={handleImageReady} />}

        {step === 1 && imageSource && (
          <MaskStep
            imageSource={imageSource}
            masked={masked}
            onMaskedChange={setMasked}
            onBack={() => goTo(0)}
            onRun={handleRun}
          />
        )}

        {step === 2 && imageSource && result && (
          <ResultStep
            imageSource={imageSource}
            masked={masked}
            result={result}
            onEditMask={handleEditMask}
            onNewImage={handleNewImage}
          />
        )}
      </main>

      {running && imageSource && (
        <InferenceLoader
          imageSource={imageSource}
          masked={masked}
          onDone={handleInferenceDone}
          onError={handleInferenceError}
        />
      )}
    </div>
  );
}
