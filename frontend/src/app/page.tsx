"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import ResultPanel, { ResultSkeleton } from "@/components/ResultPanel";
import UploadDropzone from "@/components/UploadDropzone";
import { ApiError, predictGalaxy, validateFile } from "@/lib/api";
import type { PredictResponse } from "@/lib/types";

/**
 * The screen's state, as a discriminated union.
 *
 * This is the closest React gets to a sealed record hierarchy, and it buys the
 * same thing: illegal states cannot be represented. There is no way to be
 * loading and showing an error at once, or to render a result without the
 * preview it belongs to, because those combinations do not typecheck.
 */
type State =
  | { kind: "idle" }
  | { kind: "loading"; previewUrl: string; fileName: string }
  | {
      kind: "success";
      previewUrl: string;
      fileName: string;
      result: PredictResponse;
    }
  | { kind: "error"; previewUrl: string | null; message: string };

export default function Home() {
  const [state, setState] = useState<State>({ kind: "idle" });

  // An object URL is a browser-held reference to the file's bytes. It is not
  // garbage collected when the string goes out of scope — it lives until
  // revoked or the document unloads. Tracking the current one in a ref lets us
  // revoke the previous URL whenever a new file replaces it.
  const previewUrlRef = useRef<string | null>(null);

  // Lets a new upload cancel one already in flight. Without this, a slow first
  // request can resolve *after* a fast second one and overwrite the newer
  // result with stale data.
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    // Runs on unmount only — releases whatever preview is current at that point.
    return () => {
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
      abortRef.current?.abort();
    };
  }, []);

  const handleFile = useCallback(async (file: File) => {
    abortRef.current?.abort();

    const problem = validateFile(file);
    if (problem) {
      setState({ kind: "error", previewUrl: null, message: problem });
      return;
    }

    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    const previewUrl = URL.createObjectURL(file);
    previewUrlRef.current = previewUrl;

    setState({ kind: "loading", previewUrl, fileName: file.name });

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const result = await predictGalaxy(file, controller.signal);
      setState({ kind: "success", previewUrl, fileName: file.name, result });
    } catch (err) {
      // A cancelled request is not a failure — a newer upload superseded it, and
      // that newer call owns the state now.
      if (err instanceof DOMException && err.name === "AbortError") return;

      setState({
        kind: "error",
        previewUrl,
        message:
          err instanceof ApiError
            ? err.message
            : "Something went wrong while analysing that image.",
      });
    }
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = null;
    setState({ kind: "idle" });
  }, []);

  const isBusy = state.kind === "loading";

  return (
    <main className="mx-auto flex w-full max-w-4xl flex-1 flex-col px-6 py-12 sm:py-20">
      <header className="mb-12 sm:mb-16">
        <div className="flex items-center gap-3">
          <span
            aria-hidden="true"
            className="h-2 w-2 rounded-full bg-gradient-to-br from-cyan-300 to-violet-400 shadow-[0_0_12px_2px_rgba(34,211,238,0.5)]"
          />
          <h1 className="font-mono text-[13px] uppercase tracking-[0.3em] text-white/80">
            AstralAI
          </h1>
        </div>
        <p className="mt-5 max-w-xl text-[22px] leading-snug font-light tracking-tight text-white/90 sm:text-[26px]">
          Galaxy morphology classification,{" "}
          <span className="text-white/50">with its reasoning shown.</span>
        </p>
        <p className="mt-3 max-w-lg text-[13px] leading-relaxed text-white/35">
          A ResNet18 model trained on Galaxy10 DECaLS sorts an image into one of
          ten morphological classes, and Grad-CAM reveals the regions it based
          that call on.
        </p>
      </header>

      <section className="flex-1">
        {state.kind === "idle" && <UploadDropzone onFile={handleFile} />}

        {state.kind === "loading" && (
          <div className="space-y-6">
            <StatusBar fileName={state.fileName} busy />
            <ResultSkeleton previewUrl={state.previewUrl} />
          </div>
        )}

        {state.kind === "success" && (
          <div className="space-y-6">
            <StatusBar fileName={state.fileName} onReset={reset} />
            <ResultPanel
              previewUrl={state.previewUrl}
              result={state.result}
            />
          </div>
        )}

        {state.kind === "error" && (
          <div className="space-y-6">
            <div
              role="alert"
              className="rounded-2xl border border-red-400/25 bg-red-400/[0.06] px-5 py-4"
            >
              <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-red-300/70">
                Analysis failed
              </p>
              <p className="mt-2 text-[14px] leading-relaxed text-red-100/85">
                {state.message}
              </p>
            </div>
            <UploadDropzone onFile={handleFile} disabled={isBusy} />
          </div>
        )}
      </section>

      <footer className="mt-16 border-t border-white/[0.07] pt-6">
        <p className="font-mono text-[11px] tracking-wide text-white/20">
          ResNet18 · Galaxy10 DECaLS · ~72% validation accuracy across 10 classes
        </p>
      </footer>
    </main>
  );
}

function StatusBar({
  fileName,
  busy = false,
  onReset,
}: {
  fileName: string;
  busy?: boolean;
  onReset?: () => void;
}) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-white/[0.07] pb-4">
      <div className="flex min-w-0 items-center gap-2.5">
        {busy && (
          <span
            aria-hidden="true"
            className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-cyan-400 motion-reduce:animate-none"
          />
        )}
        <p className="truncate font-mono text-[12px] text-white/45">
          {busy ? "Analysing" : "Analysed"} <span className="text-white/70">{fileName}</span>
        </p>
      </div>

      {onReset && (
        <button
          type="button"
          onClick={onReset}
          className="shrink-0 rounded-lg px-3 py-1.5 font-mono text-[11px] uppercase tracking-[0.15em] text-white/45 transition-colors hover:bg-white/[0.05] hover:text-white/80 motion-reduce:transition-none"
        >
          New image
        </button>
      )}

      {busy && (
        <span className="sr-only" role="status">
          Running inference on {fileName}
        </span>
      )}
    </div>
  );
}
