import type { PredictResponse } from "./types";

/**
 * Base URL of the FastAPI inference service.
 *
 * NEXT_PUBLIC_* variables are inlined by the bundler at BUILD time — they are
 * not read from the environment when the page runs. That differs from
 * appsettings.json, where changing a value and restarting the host is enough:
 * here, editing .env.local requires restarting `npm run dev` for the new value
 * to reach the browser.
 *
 * The `?? ` fallback is a convenience for a fresh clone with no .env.local, not
 * a substitute for setting it.
 */
export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** Largest upload we will attempt. Galaxy10 frames are ~150 KB; this is slack. */
export const MAX_FILE_BYTES = 10 * 1024 * 1024;

/**
 * A failure we can describe to the user.
 *
 * `kind` separates the two cases that need different wording:
 *   - "network"  the request never reached the server (backend down, or the
 *                browser blocked it as a CORS violation)
 *   - "http"     the server answered, but with a non-2xx status
 */
export class ApiError extends Error {
  readonly kind: "network" | "http";
  readonly status?: number;

  constructor(kind: "network" | "http", message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
  }
}

/** Reject obviously-bad files before spending a round trip on them. */
export function validateFile(file: File): string | null {
  if (!file.type.startsWith("image/")) {
    return `That looks like ${file.type || "an unknown file type"}. Please choose an image.`;
  }
  if (file.size > MAX_FILE_BYTES) {
    const mb = (file.size / 1024 / 1024).toFixed(1);
    return `That image is ${mb} MB. Please choose one under 10 MB.`;
  }
  return null;
}

/**
 * POST an image to /predict and return the classification plus Grad-CAM overlay.
 *
 * @param file   the image to classify
 * @param signal lets the caller abort an in-flight request (see page.tsx — a
 *               second upload aborts the first so a slow early response cannot
 *               land after a fast later one)
 */
export async function predictGalaxy(
  file: File,
  signal?: AbortSignal,
): Promise<PredictResponse> {
  const form = new FormData();

  // The field name MUST be "file". It matches the parameter name in the FastAPI
  // handler (`file: UploadFile = File(...)`). Any other name and FastAPI reports
  // 422 Unprocessable Entity for a missing required field.
  form.append("file", file);

  let response: Response;
  try {
    response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      body: form,
      signal,

      // Note what is NOT here: a `headers` object.
      //
      // Passing `Content-Type: multipart/form-data` by hand is the classic way
      // to break a file upload. The browser has to set that header itself,
      // because it needs to append the generated part boundary:
      //
      //     Content-Type: multipart/form-data; boundary=----WebKitFormBoundaryAbc123
      //
      // Set it manually and the boundary is missing, so the server cannot split
      // the body into parts. The resulting 422 looks like a schema error and
      // sends you hunting in completely the wrong place.
    });
  } catch (err) {
    // fetch() rejects only on network-level failure — DNS, connection refused,
    // or a CORS block. It does NOT reject on 4xx/5xx, unlike HttpClient with
    // EnsureSuccessStatusCode. A non-2xx status is a *resolved* promise.
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError(
      "network",
      `Cannot reach the inference API at ${API_URL}. Is the backend running?`,
    );
  }

  if (!response.ok) {
    // FastAPI reports errors as {"detail": "..."} — both for our explicit
    // HTTPException(400, ...) calls and for its own 422 validation failures,
    // though in the 422 case `detail` is an array of error objects.
    let detail: string | undefined;
    try {
      const body = await response.json();
      detail =
        typeof body?.detail === "string"
          ? body.detail
          : Array.isArray(body?.detail)
            ? body.detail[0]?.msg
            : undefined;
    } catch {
      // Body was not JSON. Fall through to the generic message.
    }
    throw new ApiError(
      "http",
      detail ?? `The API returned ${response.status} ${response.statusText}.`,
      response.status,
    );
  }

  return (await response.json()) as PredictResponse;
}

/** Build an <img> src from the raw base64 the API returns. */
export function heatmapSrc(base64: string): string {
  return `data:image/png;base64,${base64}`;
}
