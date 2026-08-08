/**
 * The wire contract with the FastAPI backend.
 *
 * This interface mirrors the `PredictResponse` Pydantic model in
 * `backend/main.py`. There is no code generation between the two — if you change
 * the model on the Python side, change this by hand. Keeping the field names
 * snake_case (rather than camelCasing them here) is deliberate: it means the
 * JSON maps onto this type with zero transformation, so there is nowhere for a
 * silent mismatch to hide.
 */
export interface PredictResponse {
  /** One of the ten Galaxy10 DECaLS morphology classes. */
  predicted_class: string;
  /** Softmax probability of the predicted class, 0-1. Render as a percentage. */
  confidence: number;
  /**
   * Grad-CAM overlay as a base64-encoded PNG.
   *
   * Raw base64 with NO data-URI prefix. To display it, build the src as
   * `data:image/png;base64,${heatmap}` — forgetting that prefix yields a broken
   * image with no console error, which is a genuinely annoying thing to debug.
   */
  heatmap: string;
}

/**
 * The ten classes the model can predict, in the index order it learned.
 *
 * The API returns the class *name*, so the frontend never needs to index into
 * this. It exists for display purposes — showing the user what the model is
 * choosing between is part of making the prediction interpretable.
 */
export const CLASS_NAMES = [
  "Disturbed",
  "Merging",
  "Round Smooth",
  "In-between Round Smooth",
  "Cigar Shaped Smooth",
  "Barred Spiral",
  "Unbarred Tight Spiral",
  "Unbarred Loose Spiral",
  "Edge-on without Bulge",
  "Edge-on with Bulge",
] as const;

/**
 * Below this softmax probability the UI explicitly flags the prediction as
 * low-confidence.
 *
 * The model is ~72% accurate across ten classes, and real validation images
 * produce predictions as low as 0.41. A bare percentage next to a confident-
 * looking class name overstates certainty, which is precisely the failure mode
 * an explainability project exists to avoid.
 */
export const LOW_CONFIDENCE_THRESHOLD = 0.5;
