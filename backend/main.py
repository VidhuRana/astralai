"""
AstralAI — FastAPI inference service.

One endpoint: POST /predict. Upload a galaxy image, get back the predicted
morphology class, a confidence score, and a base64-encoded Grad-CAM overlay PNG.

Run it:
    .venv\\Scripts\\python.exe -m uvicorn main:app --reload

Then open http://localhost:8000/docs — FastAPI generates that Swagger page from
the type hints below, no configuration required.
"""

from __future__ import annotations

import base64
import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from inference import CLASS_NAMES, MODEL_PATH, load_model, predict_with_heatmap

# Log through uvicorn's logger rather than print(). Python buffers stdout when it
# is not attached to a terminal, so bare print() calls at startup can vanish or
# arrive out of order in the server log.
logger = logging.getLogger("uvicorn.error")

# Origins allowed to call this API from a browser. Next.js dev server lives on
# 3000. Listing explicit origins rather than "*" because a wildcard is rejected
# by browsers on credentialed requests, which the frontend may need later.
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------


class PredictResponse(BaseModel):
    """The JSON contract for POST /predict.

    Pydantic models are roughly C# DTOs with validation attributes built in —
    and FastAPI additionally uses them to generate the OpenAPI schema, so this
    one class does the job of a DTO, FluentValidation and Swashbuckle together.
    """

    predicted_class: str = Field(
        ..., description="Galaxy morphology class name.", examples=["Merging"]
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Softmax probability of the predicted class, 0-1.",
        examples=[0.87],
    )
    heatmap: str = Field(
        ...,
        description=(
            "Grad-CAM overlay as a base64-encoded PNG. Raw base64, no data-URI "
            "prefix — a browser client should prepend 'data:image/png;base64,'."
        ),
        examples=["iVBORw0KGgoAAAANSUhEUg..."],
    )


# ---------------------------------------------------------------------------
# Application lifespan — load the model exactly once
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hook.

    Everything before `yield` runs once at startup, everything after runs once
    at shutdown. Loading the 45 MB checkpoint here and stashing it on app.state
    makes it a singleton — the same idea as registering a singleton in .NET's DI
    container and resolving it per request. Loading it inside the endpoint would
    re-read and re-initialise the network on every single upload.

    With `--reload` this runs again after each file save. That is expected.
    """
    logger.info("Loading galaxy classifier from %s ...", MODEL_PATH)
    app.state.model = load_model()
    logger.info("Model ready. Serving %d morphology classes.", len(CLASS_NAMES))
    yield
    logger.info("Releasing model.")
    app.state.model = None


app = FastAPI(
    title="AstralAI Inference API",
    description="Galaxy morphology classification with Grad-CAM explanations.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(
    request: Request,
    file: UploadFile = File(..., description="Galaxy image (PNG or JPEG)."),
) -> PredictResponse:
    """Classify an uploaded galaxy image and explain the decision.

    Declared with `def` rather than `async def` on purpose. FastAPI runs a plain
    `def` endpoint in a worker threadpool, so the blocking PyTorch forward and
    backward passes do not stall the event loop. An `async def` endpoint doing
    the same synchronous work would freeze the whole server for its duration.
    """
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image upload, got content type: {file.content_type!r}",
        )

    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not decode image: {exc}"
        ) from exc

    predicted_class, confidence, overlay = predict_with_heatmap(
        request.app.state.model, image
    )

    # PIL image -> in-memory PNG bytes -> base64 text safe to embed in JSON.
    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")
    heatmap_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    return PredictResponse(
        predicted_class=predicted_class,
        confidence=round(confidence, 4),
        heatmap=heatmap_b64,
    )
