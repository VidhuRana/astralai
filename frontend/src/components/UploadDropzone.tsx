"use client";

import { useCallback, useRef, useState } from "react";

/**
 * Drag-and-drop / click-to-browse file picker.
 *
 * Built around a real <input type="file">, with the drop target layered on top
 * as an enhancement. A div-only dropzone looks fine and is unusable by keyboard;
 * this way Tab reaches the input and Enter opens the picker for free.
 */
export default function UploadDropzone({
  onFile,
  disabled = false,
}: {
  onFile: (file: File) => void;
  disabled?: boolean;
}) {
  const [isDragging, setIsDragging] = useState(false);

  // dragenter/dragleave fire for every child element the pointer crosses, so a
  // plain boolean flickers as you move over the icon or the text. Counting
  // enters minus leaves is the standard fix.
  const dragDepth = useRef(0);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragDepth.current += 1;
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragDepth.current -= 1;
    if (dragDepth.current <= 0) {
      dragDepth.current = 0;
      setIsDragging(false);
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    // Without this preventDefault the browser treats the drop as "open this
    // file", navigates away from the app, and the drop handler never runs.
    e.preventDefault();
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      dragDepth.current = 0;
      setIsDragging(false);
      if (disabled) return;
      const file = e.dataTransfer.files?.[0];
      if (file) onFile(file);
    },
    [disabled, onFile],
  );

  return (
    <div
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      className="w-full"
    >
      <input
        id="galaxy-input"
        type="file"
        accept="image/*"
        disabled={disabled}
        className="peer sr-only"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFile(file);
          // Reset so selecting the same file twice in a row still fires change.
          e.target.value = "";
        }}
      />

      <label
        htmlFor="galaxy-input"
        aria-disabled={disabled}
        className={[
          "group relative flex w-full cursor-pointer flex-col items-center justify-center gap-4",
          "rounded-3xl border border-dashed px-8 py-16 text-center",
          "transition-colors duration-200 motion-reduce:transition-none",
          "peer-focus-visible:ring-2 peer-focus-visible:ring-cyan-400 peer-focus-visible:ring-offset-2 peer-focus-visible:ring-offset-[#05060a]",
          disabled
            ? "pointer-events-none border-white/10 opacity-50"
            : isDragging
              ? "border-cyan-400/70 bg-cyan-400/[0.06]"
              : "border-white/15 hover:border-cyan-400/40 hover:bg-white/[0.02]",
        ].join(" ")}
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          aria-hidden="true"
          className={[
            "h-10 w-10 transition-colors duration-200 motion-reduce:transition-none",
            isDragging
              ? "text-cyan-300"
              : "text-white/30 group-hover:text-cyan-400/70",
          ].join(" ")}
        >
          <path
            d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M4 15v2.5A2.5 2.5 0 0 0 6.5 20h11a2.5 2.5 0 0 0 2.5-2.5V15"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
          />
        </svg>

        <div className="space-y-1.5">
          <p className="text-[15px] font-medium text-white/90">
            {isDragging ? "Release to analyse" : "Drop a galaxy image"}
          </p>
          <p className="text-[13px] text-white/40">
            or <span className="text-cyan-400/90 underline decoration-cyan-400/30 underline-offset-4">browse your files</span>
          </p>
        </div>

        <p className="text-[11px] tracking-wide text-white/25">
          PNG or JPEG · up to 10 MB
        </p>
      </label>
    </div>
  );
}
