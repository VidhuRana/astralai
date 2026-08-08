# AstralAI — frontend

Next.js (App Router, TypeScript, Tailwind v4) client for the AstralAI galaxy
morphology classifier. One screen: upload a galaxy image, see the predicted
class, the confidence, and a Grad-CAM heatmap showing what the model looked at.

This app is a pure client of the FastAPI service in `../backend`. It holds no
model and no inference logic — it renders whatever `POST /predict` returns.

## Running it

The backend must be running first, or every upload fails with a "cannot reach
the inference API" message.

```bash
# terminal 1 — from ../backend
.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000

# terminal 2 — from here
npm install
npm run dev
```

Then open <http://localhost:3000>.

## Configuration

```bash
cp .env.example .env.local
```

| Variable | Default | Notes |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Base URL of the FastAPI service |

`NEXT_PUBLIC_*` variables are inlined into the client bundle at **build** time,
not read at runtime — restart `npm run dev` after changing one.

## Structure

```
src/
├── app/
│   ├── layout.tsx      fonts, metadata
│   ├── page.tsx        the screen; owns the upload state machine
│   └── globals.css     dark theme, CSS-only star field
├── components/
│   ├── UploadDropzone.tsx   drag-drop + file picker + validation
│   └── ResultPanel.tsx      side-by-side result, confidence, skeleton
└── lib/
    ├── api.ts          predictGalaxy(), ApiError, validateFile()
    └── types.ts        PredictResponse — mirrors the backend Pydantic model
```

## Three things that will bite you if you change them

1. **The FormData field name must be `file`.** It matches the parameter name in
   the FastAPI handler. Anything else returns 422.
2. **Never set `Content-Type` on the upload fetch.** The browser must set it
   itself so it can append the multipart boundary. Setting it by hand produces a
   422 that looks like a schema error and isn't.
3. **The heatmap needs the `data:image/png;base64,` prefix.** The API returns
   raw base64. Without the prefix you get a broken image and no console error.
