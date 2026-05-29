/**
 * ExportButton.tsx
 *
 * Drop-in export button for the Lecture Annotation Tool.
 *
 * Usage:
 *   <ExportButton segments={annotatedSegments} />
 *
 * Where `annotatedSegments` is your array of AnnotatedSegment objects
 * (see exportWhisper.ts for the interface).
 *
 * The component handles:
 *  - Loading JSZip lazily (no bundle-time dependency)
 *  - WAV conversion progress
 *  - Download trigger
 *  - Error display
 */

import { useState, useCallback } from "react";
import { buildWhisperExportZip, downloadBlob, AnnotatedSegment } from "../utils/exportWhisper";

// ---- Types ----------------------------------------------------------------

interface ExportButtonProps {
  segments: AnnotatedSegment[];
  /** Optional dataset name override (default: "whisper_dataset") */
  datasetName?: string;
  /** Disable the button externally (e.g. while recording) */
  disabled?: boolean;
}

type ExportPhase = "idle" | "converting" | "zipping" | "done" | "error";

// ---- Component ------------------------------------------------------------

export function ExportButton({ segments, datasetName, disabled }: ExportButtonProps) {
  const [phase, setPhase] = useState<ExportPhase>("idle");
  const [progress, setProgress] = useState(0); // 0–100
  const [errorMsg, setErrorMsg] = useState("");

  const handleExport = useCallback(async () => {
    if (segments.length === 0) {
      setErrorMsg("No annotated segments to export.");
      setPhase("error");
      return;
    }

    setPhase("converting");
    setProgress(0);
    setErrorMsg("");

    try {
      // Lazy-load JSZip so it doesn't bloat the initial bundle
      const { default: JSZip } = await import("jszip");

      // We patch buildWhisperExportZip to report progress via a wrapped
      // segments array that calls setProgress after each WAV conversion.
      // We do this by building a tiny progress-reporting shim here.
      const total = segments.length;
      let done = 0;

      // Wrap each segment so we can hook into WAV conversion progress
      // (the real work happens inside buildWhisperExportZip, so we report
      //  progress after the call returns each segment's wav blob)
      const progressSegments = segments.map((seg) => ({
        ...seg,
        // Intercept the audioBlob read by replacing it with a Proxy-like approach:
        // We'll just increment after each segment is processed via a post-process hook.
        _onConverted: () => {
          done++;
          setProgress(Math.round((done / total) * 80)); // reserve 80% for conversion
        },
      }));

      // Since the export util doesn't natively call _onConverted, we simulate
      // progress by running a lightweight pre-pass here:
      setPhase("converting");

      // Small async tick so React re-renders the "converting" state
      await new Promise((r) => setTimeout(r, 0));

      // Simulate incremental progress during the zip build (real work is async)
      const progressInterval = setInterval(() => {
        setProgress((p) => (p < 75 ? p + 3 : p));
      }, 150);

      setPhase("zipping");
      const zipBlob = await buildWhisperExportZip(segments, JSZip, {
        datasetName: datasetName ?? "whisper_dataset",
      });

      clearInterval(progressInterval);
      setProgress(100);
      setPhase("done");

      const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
      downloadBlob(zipBlob, `${datasetName ?? "whisper_dataset"}_${ts}.zip`);

      // Reset after 3 seconds so the button is reusable
      setTimeout(() => {
        setPhase("idle");
        setProgress(0);
      }, 3000);
    } catch (err: any) {
      setPhase("error");
      setErrorMsg(err?.message ?? "Export failed.");
    }
  }, [segments, datasetName]);

  // ---- Render ---------------------------------------------------------------

  const isProcessing = phase === "converting" || phase === "zipping";
  const isDisabled = disabled || isProcessing || segments.length === 0;

  return (
    <div className="export-btn-wrapper">
      <button
        onClick={handleExport}
        disabled={isDisabled}
        className={`export-btn export-btn--${phase}`}
        title={
          segments.length === 0
            ? "No segments to export yet"
            : `Export ${segments.length} segment${segments.length !== 1 ? "s" : ""} as Whisper dataset`
        }
      >
        {/* Icon */}
        <span className="export-btn__icon" aria-hidden="true">
          {phase === "done" ? "✓" : phase === "error" ? "✗" : isProcessing ? "⟳" : "⬇"}
        </span>

        {/* Label */}
        <span className="export-btn__label">
          {phase === "converting" && "Converting audio…"}
          {phase === "zipping" && "Building zip…"}
          {phase === "done" && "Downloaded!"}
          {phase === "error" && "Export failed"}
          {phase === "idle" && (
            <>
              Export Dataset
              {segments.length > 0 && (
                <span className="export-btn__count">{segments.length}</span>
              )}
            </>
          )}
        </span>
      </button>

      {/* Progress bar (visible during processing) */}
      {isProcessing && (
        <div className="export-progress" role="progressbar" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100}>
          <div className="export-progress__fill" style={{ width: `${progress}%` }} />
          <span className="export-progress__label">{progress}%</span>
        </div>
      )}

      {/* Error message */}
      {phase === "error" && errorMsg && (
        <p className="export-error" role="alert">
          {errorMsg}
          <button
            className="export-error__dismiss"
            onClick={() => { setPhase("idle"); setErrorMsg(""); }}
          >
            Dismiss
          </button>
        </p>
      )}

      {/* What gets exported — shown as a tooltip-style hint */}
      {phase === "idle" && segments.length > 0 && (
        <p className="export-hint">
          → <code>whisper_dataset.zip</code> containing <code>metadata.jsonl</code> + {segments.length} <code>.wav</code> file{segments.length !== 1 ? "s" : ""}
        </p>
      )}
    </div>
  );
}

export default ExportButton;
