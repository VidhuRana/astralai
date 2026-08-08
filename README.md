# AstralAI

An explainable AI platform for astronomical image analysis.
MCA final-year capstone project.

## Status
🚧 Week 3 complete — the project now has a face. Week 1 proved the
ML pipeline (ResNet18 + Grad-CAM on Galaxy10 DECaLS, ~72% validation
accuracy across 10 classes), Week 2 served it over HTTP, and Week 3
added a Next.js frontend that consumes that API across origins.

![Grad-CAM on a merging galaxy](first_gradcam.png)

## Stack
Python · PyTorch · FastAPI · Next.js · PostgreSQL

## Running the whole thing

Two apps, two ports, one JSON contract between them. Both must be running.

```bash
# terminal 1 — inference API on :8000
cd backend
.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000

# terminal 2 — web UI on :3000
cd frontend
npm install
npm run dev
```

Open <http://localhost:3000>, drop in a galaxy image, and you get the predicted
morphology, a confidence percentage, and the Grad-CAM heatmap beside the
original.

The weights are not in this repo — see the Inference API section below.

## Inference API (Week 2)

A local FastAPI service exposing a single endpoint.

### `POST /predict`

Accepts a `multipart/form-data` upload with a `file` field containing a galaxy
image (PNG or JPEG). Returns:

```json
{
  "predicted_class": "Merging",
  "confidence": 0.87,
  "heatmap": "<base64-encoded PNG of the Grad-CAM overlay>"
}
```

`heatmap` is raw base64 with no data-URI prefix — a browser client should
prepend `data:image/png;base64,`.

### Setup

The trained weights are **not** in this repo (a 45 MB binary does not belong in
git history). Download `galaxy10_resnet18.pth` from Google Drive
(`MyDrive/astralai/galaxy10_resnet18.pth`, produced by `AstralAI_Week1.ipynb`)
and place it at `backend/models/galaxy10_resnet18.pth`.

```bash
cd backend
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

CPU-only is fine — single-image inference takes well under a second.

### Run

```bash
cd backend
.venv\Scripts\python.exe -m uvicorn main:app --reload
```

Open <http://localhost:8000/docs> for the auto-generated Swagger UI and upload
an image there.

To test the model without the web layer at all:

```bash
.venv\Scripts\python.exe inference.py test_images\val_02385_c01_Merging.png
```

That prints the predicted class and confidence, and writes `gradcam_out.png`.

Test images are not committed (regenerate them, they are derived data). See
`backend/test_images/README.md` for the Colab cell that exports labelled
validation galaxies — the ground-truth class is encoded in each filename, so a
prediction can be checked for correctness rather than mere plausibility.

### Layout

```
backend/
├── models/galaxy10_resnet18.pth   # gitignored — download separately
├── inference.py                   # model loading, prediction, Grad-CAM
├── main.py                        # FastAPI app
└── requirements.txt
```

`inference.py` contains no web code and `main.py` contains no ML code, so the
model can be tested independently of the server.

## Web UI (Week 3)

A separate Next.js app (App Router, TypeScript, Tailwind) in `frontend/`, running
on its own dev server and talking to the API across origins. Deliberately a
separate app rather than templates served by FastAPI, because the project is
meant to grow into a multi-module platform.

One screen: drop a galaxy image, watch it infer, see the original and the
Grad-CAM overlay side by side with the class and confidence. Predictions below
50% confidence are explicitly flagged as uncertain — the model is ~72% accurate,
and a bare percentage next to a confident-looking class name overstates what it
actually knows.

Configure the API base with `NEXT_PUBLIC_API_URL` (see `frontend/.env.example`).
Details and the three upload gotchas are in `frontend/README.md`.

## Roadmap
- [x] Repo setup
- [x] Galaxy classifier + Grad-CAM (Week 1)
- [x] FastAPI inference endpoint (Week 2)
- [x] Web UI + upload flow (Week 3)
- [ ] Research dashboard
- [ ] AI astronomy assistant
- [ ] *Stretch:* exoplanet light-curve detector
