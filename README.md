# AstralAI

An explainable AI platform for astronomical image analysis.
MCA final-year capstone project.

## Status
🚧 Week 1 complete — ML pipeline proven: galaxy morphology
classification (ResNet18 + Grad-CAM) on Galaxy10 DECaLS.
Validation accuracy ~72% across 10 classes, with interpretable
heatmaps landing on real galaxy structure.

![Grad-CAM on a merging galaxy](notebooks/first_gradcam.png)

## Stack
Python · PyTorch · FastAPI · Next.js · PostgreSQL

## Roadmap
- [x] Repo setup
- [x] Galaxy classifier + Grad-CAM (Week 1)
- [ ] FastAPI inference endpoint (Week 2)
- [ ] Web UI + upload flow
- [ ] Research dashboard
- [ ] AI astronomy assistant
- [ ] *Stretch:* exoplanet light-curve detector
