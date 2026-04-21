import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const GRID_SIZE = 14;
const TOTAL_CELLS = GRID_SIZE * GRID_SIZE;
const MAX_MASKED = Math.floor(TOTAL_CELLS * 0.75); // 147

type Prediction = {
  label: string;
  confidence: number;
};

type InferenceResponse = {
  mae_reconstruction: string;
  unet_reconstruction: string;
  mae_mse: number;
  unet_mse: number;
  top_predictions: Prediction[];
};

type HealthState = "checking" | "ready" | "error";
type ResultView = "mae" | "unet" | "original";

type FileMeta = {
  name: string;
  sizeKB: number;
  width: number;
  height: number;
};

function formatKB(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function App() {
  const [health, setHealth] = useState<HealthState>("checking");
  const [device, setDevice] = useState<string>("—");

  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [fileMeta, setFileMeta] = useState<FileMeta | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [masked, setMasked] = useState<Set<number>>(() => new Set());
  const [dragOver, setDragOver] = useState(false);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [view, setView] = useState<ResultView>("mae");

  const [focusTile, setFocusTile] = useState<string | null>(null);

  // painting state
  const paintingRef = useRef(false);
  const paintModeRef = useRef<"add" | "remove">("add");

  // ---------------- health check ----------------
  useEffect(() => {
    let cancelled = false;
    fetch("/health")
      .then((r) =>
        r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)),
      )
      .then((j: { status: string; device: string }) => {
        if (cancelled) return;
        setHealth(j.status === "ok" ? "ready" : "error");
        setDevice(j.device || "—");
      })
      .catch(() => {
        if (!cancelled) setHealth("error");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // ---------------- upload handling ----------------
  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please upload an image file.");
      return;
    }
    const url = URL.createObjectURL(file);
    const probe = new Image();
    probe.onload = () => {
      setImageUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return url;
      });
      setFileMeta({
        name: file.name,
        sizeKB: file.size,
        width: probe.naturalWidth,
        height: probe.naturalHeight,
      });
      setMasked(new Set());
      setResult(null);
      setError(null);
    };
    probe.onerror = () => {
      URL.revokeObjectURL(url);
      setError("Could not decode image.");
    };
    probe.src = url;
  }, []);

  const onFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
    if (e.target) e.target.value = "";
  };

  const removeFile = () => {
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    setImageUrl(null);
    setFileMeta(null);
    setMasked(new Set());
    setResult(null);
  };

  // ---------------- paint ----------------
  const setMaskedCell = useCallback((idx: number, on: boolean) => {
    setMasked((prev) => {
      if (on) {
        if (prev.has(idx)) return prev;
        if (prev.size >= MAX_MASKED) return prev;
        const next = new Set(prev);
        next.add(idx);
        return next;
      } else {
        if (!prev.has(idx)) return prev;
        const next = new Set(prev);
        next.delete(idx);
        return next;
      }
    });
  }, []);

  const onCellPointerDown =
    (idx: number, isMasked: boolean) => (e: React.PointerEvent) => {
      if (!imageUrl) return;
      paintingRef.current = true;
      paintModeRef.current = isMasked ? "remove" : "add";
      setMaskedCell(idx, paintModeRef.current === "add");
      e.preventDefault();
      (e.target as HTMLElement).releasePointerCapture?.(e.pointerId);
    };

  const onCellPointerEnter = (idx: number) => () => {
    if (!paintingRef.current || !imageUrl) return;
    setMaskedCell(idx, paintModeRef.current === "add");
  };

  useEffect(() => {
    const up = () => {
      paintingRef.current = false;
    };
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
    return () => {
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
    };
  }, []);

  const clearMask = useCallback(() => {
    setMasked(new Set());
    setResult(null);
    setError(null);
  }, []);

  const invertMask = useCallback(() => {
    setMasked((prev) => {
      const inv = new Set<number>();
      for (let i = 0; i < TOTAL_CELLS; i++) {
        if (!prev.has(i) && inv.size < MAX_MASKED) inv.add(i);
      }
      return inv;
    });
  }, []);

  // ---------------- reconstruct ----------------
  const runReconstruct = useCallback(async () => {
    if (!imageUrl || masked.size === 0 || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    const t0 = performance.now();

    try {
      // Send the original image + mask indices to the backend.
      // The backend applies the mask internally so MAE can properly reconstruct.
      const imgBlob = await fetch(imageUrl).then((r) => r.blob());

      const form = new FormData();
      form.append("file", imgBlob, "image.png");
      form.append("mask", Array.from(masked).join(","));

      const resp = await fetch("/infer", { method: "POST", body: form });
      if (!resp.ok) {
        const txt = await resp.text().catch(() => "");
        throw new Error(`HTTP ${resp.status} ${txt}`.trim());
      }
      const data: InferenceResponse = await resp.json();
      setResult(data);
      setView("mae");
      setLatency((performance.now() - t0) / 1000);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [imageUrl, masked, loading]);

  // ---------------- keyboard ----------------
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.code === "Space" || e.key === " ") {
        e.preventDefault();
        if (masked.size > 0 && !loading) runReconstruct();
      } else if (e.key === "Escape") {
        clearMask();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [masked.size, loading, runReconstruct, clearMask]);

  // ---------------- derived ----------------
  const n = masked.size;
  const pctTotal = (n / TOTAL_CELLS) * 100;
  const pctMax = n / MAX_MASKED;
  const numClass = pctMax >= 1 ? "over" : pctMax >= 0.9 ? "warn" : "";
  const fillClass = pctMax >= 0.9 && pctMax < 1 ? "warn" : "";
  const meterWidth = Math.min(pctMax * 100, 100);

  const resultSrc = useMemo(() => {
    if (!result) return null;
    if (view === "mae")
      return `data:image/png;base64,${result.mae_reconstruction}`;
    if (view === "unet")
      return `data:image/png;base64,${result.unet_reconstruction}`;
    return imageUrl;
  }, [result, view, imageUrl]);

  const tileClass = (key: string) => {
    if (!focusTile) return "tile";
    return focusTile === key ? "tile focus-active" : "tile dimmed";
  };

  const maskCells = useMemo(() => Array.from({ length: TOTAL_CELLS }), []);

  const reconstructDisabled = n === 0 || !imageUrl || loading;

  return (
    <div className="app">
      {/* ---------- Top nav ---------- */}
      <nav className="nav">
        <div className="brand">
          <div className="glyph" aria-hidden="true">
            <i />
            <i />
            <i />
            <i />
            <i />
            <i />
            <i />
            <i />
            <i />
          </div>
          <div>
            <span className="name">
              reconstruct<span className="dot">.</span>
            </span>
            <span className="tag mono">Research Project</span>
          </div>
        </div>
        <div className="nav-meta">
          <span>
            <span className={`dot${health === "error" ? " err" : ""}`} />
            {health === "ready"
              ? "model ready"
              : health === "error"
                ? "backend offline"
                : "checking…"}
          </span>
          <span>device: {device}</span>
          <a href="https://github.com/pinwap/MAE-animal-reconstruct-and-classification">
            github ↗
          </a>
        </div>
      </nav>

      {/* ---------- Title ---------- */}
      <header className="title-row">
        <div>
          <h1>Mask &amp; reconstruct</h1>
          <div className="subtitle">
            Upload an image, mask any cells in the <b>14 × 14</b> grid (up to{" "}
            <b>75%</b>), then reconstruct. Compare original, masked input, and
            the model's prediction side-by-side.
          </div>
        </div>
        <div className="quick-meta">
          <div className="qm">
            <div className="l">grid</div>
            <div className="v">14 × 14</div>
          </div>
          <div className="qm">
            <div className="l">cells</div>
            <div className="v">196</div>
          </div>
          <div className="qm">
            <div className="l">max masked</div>
            <div className="v">147 (75%)</div>
          </div>
          <div className="qm">
            <div className="l">latency</div>
            <div className="v">{latency ? `${latency.toFixed(2)} s` : "—"}</div>
          </div>
        </div>
      </header>

      {/* ---------- 3-up grid ---------- */}
      <section className="canvas-grid">
        {/* ===== Tile 1 — Source ===== */}
        <article
          className={tileClass("source")}
          onMouseEnter={() => setFocusTile("source")}
          onMouseLeave={() => setFocusTile(null)}
        >
          <div className="tile-head">
            <div className="left">
              <span className="step-badge">1</span>
              <span className="tile-title">Source</span>
            </div>
            <span className="tile-tag">original</span>
          </div>

          {imageUrl ? (
            <div className="img-frame">
              <img className="photo" src={imageUrl} alt="" />
            </div>
          ) : (
            <div
              className={`upload-empty${dragOver ? " drag-over" : ""}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                const f = e.dataTransfer.files?.[0];
                if (f) handleFile(f);
              }}
            >
              <div className="icon">↑</div>
              <div>
                <div className="big">Drop an image</div>
                <div className="small">
                  PNG, JPG · center-cropped to 224 × 224
                </div>
              </div>
            </div>
          )}

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={onFileInputChange}
          />

          {fileMeta && (
            <div className="file-chip">
              <div>
                <div className="name">{fileMeta.name}</div>
                <div className="meta">
                  {fileMeta.width} × {fileMeta.height} ·{" "}
                  {formatKB(fileMeta.sizeKB)}
                </div>
              </div>
              <span className="x" title="remove" onClick={removeFile}>
                ✕
              </span>
            </div>
          )}

          <div className="upload-actions">
            <button
              className="btn"
              style={{ flex: 1, justifyContent: "center" }}
              onClick={() => fileInputRef.current?.click()}
            >
              ↑ {imageUrl ? "Replace" : "Upload"}
            </button>
            {imageUrl && (
              <button className="btn" onClick={removeFile}>
                Clear
              </button>
            )}
          </div>
        </article>

        {/* ===== Tile 2 — Mask ===== */}
        <article
          className={tileClass("mask")}
          onMouseEnter={() => setFocusTile("mask")}
          onMouseLeave={() => setFocusTile(null)}
        >
          <div className="tile-head">
            <div className="left">
              <span className="step-badge">2</span>
              <span className="tile-title">Mask cells</span>
            </div>
            <span className="tile-tag">click + drag</span>
          </div>

          <div className={`img-frame${!imageUrl ? " placeholder" : ""}`}>
            {imageUrl && <img className="photo" src={imageUrl} alt="" />}
            {!imageUrl && (
              <div className="ph-caption">
                <div className="name">upload to start masking</div>
              </div>
            )}
            <div className={`grid-overlay${!imageUrl ? " disabled" : ""}`}>
              {maskCells.map((_, i) => {
                const isMasked = masked.has(i);
                return (
                  <div
                    key={i}
                    className={`cell${isMasked ? " masked" : ""}`}
                    onPointerDown={onCellPointerDown(i, isMasked)}
                    onPointerEnter={onCellPointerEnter(i)}
                  />
                );
              })}
            </div>
          </div>

          <div className="tile-toolbar">
            <div className="counter-block">
              <div>
                <div className={`counter-num ${numClass}`}>
                  <span>{n}</span>
                  <span className="of"> / {MAX_MASKED}</span>
                </div>
              </div>
              <div className="counter-pct">
                <b>{pctTotal.toFixed(0)}%</b> &nbsp;·&nbsp; limit 75%
              </div>
            </div>
            <div className="meter">
              <div
                className={`fill ${fillClass}`}
                style={{ width: `${meterWidth}%` }}
              />
              <div className="limit-mark" title="75% limit" />
            </div>
            <div className="actions">
              <button
                className="btn"
                onClick={clearMask}
                disabled={n === 0 && !result}
              >
                Clear <span className="kbd">⎋</span>
              </button>
              <button
                className="btn"
                onClick={invertMask}
                disabled={!imageUrl}
                title="invert mask"
              >
                Invert
              </button>
              <div style={{ flex: 1 }} />
              <button
                className="btn primary"
                onClick={runReconstruct}
                disabled={reconstructDisabled}
              >
                <span>{loading ? "Running…" : "↻ Reconstruct"}</span>
                <span className="kbd">␣</span>
              </button>
            </div>
          </div>
        </article>

        {/* ===== Tile 3 — Result ===== */}
        <article
          className={tileClass("result")}
          onMouseEnter={() => setFocusTile("result")}
          onMouseLeave={() => setFocusTile(null)}
        >
          <div className="tile-head">
            <div className="left">
              <span className="step-badge">3</span>
              <span className="tile-title">Reconstruction</span>
            </div>
            <span className="tile-tag">
              {loading
                ? "running…"
                : result
                  ? "complete"
                  : error
                    ? "error"
                    : "awaiting input"}
            </span>
          </div>

          <div
            className={`img-frame${!result && !loading ? " result-empty" : ""}${loading ? " loading" : ""}`}
          >
            {resultSrc && <img className="photo" src={resultSrc} alt="" />}

            {!result && !loading && (
              <div className="ph-caption">
                <div className="name">
                  {error
                    ? "error · try again"
                    : "run reconstruct to see output"}
                </div>
                <div className="meta">
                  {error ?? "side-by-side with source above"}
                </div>
              </div>
            )}

            {loading && (
              <div className="ph-caption">
                <div className="name">reconstructing…</div>
                <div className="meta">running mae-b/16 + u-net</div>
              </div>
            )}

            {result && (
              <>
                <div className="result-stat">
                  <span className={`dot${error ? " err" : ""}`} />
                  complete · {latency?.toFixed(2)}s · top-1{" "}
                  {result.top_predictions[0]?.label}
                </div>
                <div className="result-swap" role="tablist">
                  <button
                    data-view="mae"
                    className={view === "mae" ? "active" : ""}
                    onClick={() => setView("mae")}
                  >
                    MAE
                  </button>
                  <button
                    data-view="unet"
                    className={view === "unet" ? "active" : ""}
                    onClick={() => setView("unet")}
                  >
                    U-Net
                  </button>
                  <button
                    data-view="original"
                    className={view === "original" ? "active" : ""}
                    onClick={() => setView("original")}
                  >
                    Original
                  </button>
                </div>
              </>
            )}
          </div>

          <div className="result-meta">
            <div className="kv">
              <span className="k">model</span>
              <span className="v">mae-b/16</span>
            </div>
            <div className="kv">
              <span className="k">latency</span>
              <span className="v">
                {latency ? `${latency.toFixed(2)} s` : "—"}
              </span>
            </div>
            <div className="kv">
              <span className="k">masked</span>
              <span className="v">{result ? `${n}/${TOTAL_CELLS}` : "—"}</span>
            </div>
            <div className="kv">
              <span className="k">view</span>
              <span className="v">{result ? view : "—"}</span>
            </div>
            <div className="kv">
              <span className="k">mae mse</span>
              <span
                className={`v mse${result && result.mae_mse <= result.unet_mse ? " best" : ""}`}
              >
                {result ? result.mae_mse.toFixed(4) : "—"}
              </span>
            </div>
            <div className="kv">
              <span className="k">unet mse</span>
              <span
                className={`v mse${result && result.unet_mse < result.mae_mse ? " best" : ""}`}
              >
                {result ? result.unet_mse.toFixed(4) : "—"}
              </span>
            </div>
          </div>

          {result && result.top_predictions.length > 0 && (
            <div className="predictions">
              <div className="predictions-head">top-3 predictions</div>
              {result.top_predictions.slice(0, 3).map((p, i) => (
                <div className="pred-row" key={p.label}>
                  <span className="rank">#{i + 1}</span>
                  <div className="bar-wrap">
                    <div
                      className="bar"
                      style={{ width: `${p.confidence * 100}%` }}
                    />
                    <div className="label-inner">
                      <span className="label">{p.label}</span>
                      <span className="conf">
                        {(p.confidence * 100).toFixed(2)}%
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </article>
      </section>

      {/* ---------- Footer ---------- */}
      <footer className="footer">
        <div>
          14 × 14 patch grid · 16 px patches · max 75% masked (147 cells)
        </div>
        <div className="kbds">
          <span className="k">
            paint <kbd>drag</kbd>
          </span>
          <span className="k">
            erase <kbd>drag-over-masked</kbd>
          </span>
          <span className="k">
            reconstruct <kbd>space</kbd>
          </span>
          <span className="k">
            clear <kbd>esc</kbd>
          </span>
        </div>
      </footer>
    </div>
  );
}

export default App;
