"use client";

import { heatmapSrc } from "@/lib/api";
import { LOW_CONFIDENCE_THRESHOLD, type PredictResponse } from "@/lib/types";

/** Shared frame so the two images and the loading skeleton align exactly. */
function ImageFrame({
  label,
  caption,
  children,
}: {
  label: string;
  caption: string;
  children: React.ReactNode;
}) {
  return (
    <figure className="flex flex-col gap-3">
      <figcaption className="flex items-baseline justify-between gap-3">
        <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-white/45">
          {label}
        </span>
        <span className="text-[11px] text-white/25">{caption}</span>
      </figcaption>
      <div className="aspect-square overflow-hidden rounded-2xl border border-white/10 bg-black/40">
        {children}
      </div>
    </figure>
  );
}

/** Placeholder shown while inference runs, sized identically to the result. */
export function ResultSkeleton({ previewUrl }: { previewUrl: string }) {
  return (
    <div className="space-y-8">
      <div className="grid gap-6 sm:grid-cols-2">
        <ImageFrame label="Original" caption="your upload">
          {/* eslint-disable-next-line @next/next/no-img-element -- blob: URL from
              URL.createObjectURL; next/image cannot optimise it and there is
              nothing to optimise. */}
          <img
            src={previewUrl}
            alt="The galaxy image you uploaded"
            className="h-full w-full object-cover"
          />
        </ImageFrame>

        <ImageFrame label="Grad-CAM" caption="computing…">
          <div className="h-full w-full animate-pulse bg-white/[0.04] motion-reduce:animate-none" />
        </ImageFrame>
      </div>

      <div className="space-y-3">
        <div className="h-3 w-24 animate-pulse rounded bg-white/[0.06] motion-reduce:animate-none" />
        <div className="h-8 w-56 animate-pulse rounded bg-white/[0.08] motion-reduce:animate-none" />
        <div className="h-1.5 w-full animate-pulse rounded-full bg-white/[0.06] motion-reduce:animate-none" />
      </div>
    </div>
  );
}

export default function ResultPanel({
  previewUrl,
  result,
}: {
  previewUrl: string;
  result: PredictResponse;
}) {
  const percent = result.confidence * 100;
  const isLowConfidence = result.confidence < LOW_CONFIDENCE_THRESHOLD;

  return (
    <div className="space-y-8">
      <div className="grid gap-6 sm:grid-cols-2">
        <ImageFrame label="Original" caption="your upload">
          {/* eslint-disable-next-line @next/next/no-img-element -- blob: URL from
              URL.createObjectURL; next/image cannot optimise it. */}
          <img
            src={previewUrl}
            alt="The galaxy image you uploaded"
            className="h-full w-full object-cover"
          />
        </ImageFrame>

        <ImageFrame label="Grad-CAM" caption="what the model looked at">
          {/* eslint-disable-next-line @next/next/no-img-element -- data: URI built
              from base64 in the API response; next/image cannot optimise it. */}
          <img
            src={heatmapSrc(result.heatmap)}
            alt={`Grad-CAM heatmap highlighting the regions that led the model to predict ${result.predicted_class}`}
            className="h-full w-full object-cover"
          />
        </ImageFrame>
      </div>

      <div className="space-y-4">
        <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2">
          <div className="space-y-1.5">
            <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-white/45">
              Predicted morphology
            </p>
            <h2 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
              {result.predicted_class}
            </h2>
          </div>

          <div className="text-right">
            <p className="font-mono text-3xl font-medium tabular-nums text-white sm:text-4xl">
              {percent.toFixed(1)}
              <span className="text-xl text-white/40">%</span>
            </p>
            <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-white/45">
              confidence
            </p>
          </div>
        </div>

        <div
          className="h-1.5 w-full overflow-hidden rounded-full bg-white/[0.07]"
          role="meter"
          aria-valuenow={Number(percent.toFixed(1))}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={`Model confidence in the ${result.predicted_class} classification`}
        >
          <div
            className={[
              "h-full rounded-full transition-[width] duration-700 ease-out motion-reduce:transition-none",
              isLowConfidence
                ? "bg-gradient-to-r from-amber-500/70 to-amber-300"
                : "bg-gradient-to-r from-cyan-500 to-violet-400",
            ].join(" ")}
            style={{ width: `${percent}%` }}
          />
        </div>

        {isLowConfidence && (
          <p className="flex items-start gap-2.5 rounded-xl border border-amber-400/20 bg-amber-400/[0.05] px-4 py-3 text-[13px] leading-relaxed text-amber-200/80">
            <span aria-hidden="true" className="mt-px select-none">
              ⚠
            </span>
            <span>
              Low confidence. The model is unsure between several morphologies
              here — treat this classification as a suggestion rather than an
              answer.
            </span>
          </p>
        )}

        <p className="max-w-prose text-[13px] leading-relaxed text-white/40">
          The heatmap shows which regions of the image most influenced the
          prediction — warm areas contributed most. If the heat sits on the
          galaxy&rsquo;s structure rather than on empty sky or image artefacts,
          the model is reasoning about the right thing.
        </p>
      </div>
    </div>
  );
}
