"""
AstralAI — pure inference logic. No web framework code lives in this file.

This module is deliberately importable and runnable on its own, so the model can
be proven to work before any HTTP layer exists:

    .venv\\Scripts\\python.exe inference.py test_images\\some_galaxy.png

Everything here mirrors the Week 1 training notebook exactly. That is not
stylistic — the constants below (class order, image size, normalisation values)
form the contract with the trained weights. Change one of them and the model
returns confident nonsense with no error message. That failure mode is called
train/serve skew, and it is the main thing this file is written to prevent.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Constants — copied verbatim from the Week 1 notebook.
# ---------------------------------------------------------------------------

# The ORDER of this list is load-bearing. The model never learned the strings,
# only the integers 0-9; this list is the only thing that maps index -> name.
# Reordering it silently relabels every prediction.
CLASS_NAMES = [
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
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = (224, 224)

# The exact validation transform from training. Note Resize((224, 224)) is a
# non-aspect-preserving squash to exactly 224x224 — NOT a short-side resize plus
# centre crop. The model was trained on squashed images, so it must be served
# squashed images.
val_tf = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

# Resolved relative to THIS file, not the current working directory, so uvicorn
# can be launched from anywhere and still find the weights.
MODEL_PATH = Path(__file__).resolve().parent / "models" / "galaxy10_resnet18.pth"

# Grad-CAM attaches hooks to the shared model object and runs a backward pass.
# Two requests doing that at the same time would interfere with each other, so
# the explanation path is serialised. Single-image CPU inference takes well under
# a second, so this costs nothing in practice.
_cam_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def build_model() -> nn.Module:
    """Reconstruct the Week 1 architecture — no weights loaded yet.

    A .pth file saved with torch.save(model.state_dict()) contains only the
    learned tensors, keyed by parameter name. It does not contain the network
    structure. So the architecture has to be rebuilt in code first, and it has
    to match, or load_state_dict() will reject the keys.
    """
    # weights=None means "give me the architecture, skip the ImageNet download".
    # Downloading pretrained weights here would be pure waste — the .pth
    # overwrites every one of them a moment later.
    model = models.resnet18(weights=None)

    # Freeze the whole backbone, then unfreeze layer4. This mirrors training.
    #
    # At inference time freezing does not affect the prediction at all, but it
    # DOES affect Grad-CAM: the explanation is computed from gradients flowing
    # back into layer4's feature maps. If every parameter is frozen, there is no
    # gradient to read and the heatmap comes back flat — with no exception
    # raised. That silent failure is why these two loops stay in the serving code.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True

    # Stock ResNet18 ends in Linear(512, 1000) for ImageNet's 1000 classes.
    # Training replaced it with Linear(512, 10). The checkpoint has the 10-class
    # shape, so this swap must happen before loading or the shapes won't match.
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    return model


def load_model(path: Path | str = MODEL_PATH) -> nn.Module:
    """Build the architecture, load the trained weights, put it in eval mode."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained weights not found at: {path}\n"
            "Download galaxy10_resnet18.pth from Google Drive "
            "(/MyDrive/astralai/galaxy10_resnet18.pth) and place it there. "
            "The file is gitignored, so a fresh clone will not have it."
        )

    model = build_model()

    # map_location='cpu' loads GPU-trained weights onto a CPU-only machine.
    # Without it, torch tries to restore the tensors to cuda:0 and raises.
    # weights_only=True refuses to unpickle arbitrary Python objects — safe here
    # because a state_dict is just tensors, and it is the modern default.
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    # eval() is NOT optional. ResNet18 contains BatchNorm layers which behave
    # differently in training mode (using the current batch's statistics) versus
    # eval mode (using the running averages learned during training). Serving a
    # single image in train mode normalises it against itself and produces
    # garbage — again, with no error. This one line prevents that.
    model.eval()

    return model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def _preprocess(image: Image.Image) -> torch.Tensor:
    """PIL image -> normalised tensor of shape (1, 3, 224, 224)."""
    # Uploaded files may be RGBA (PNG with transparency), greyscale, or palette
    # indexed. Training data was always 3-channel RGB. Without this conversion a
    # 4-channel image produces a (4, 224, 224) tensor and the first conv layer
    # raises a shape error — or worse, a 1-channel greyscale silently broadcasts.
    image = image.convert("RGB")

    tensor = val_tf(image)  # (3, 224, 224)

    # PyTorch models always expect a batch dimension, even for one image.
    # unsqueeze(0) inserts a length-1 axis at position 0 -> (1, 3, 224, 224).
    # The convention is NCHW: batch, channels, height, width.
    return tensor.unsqueeze(0)


def _logits_to_result(logits: torch.Tensor) -> tuple[int, float]:
    """Turn raw model output into (class_index, confidence)."""
    # The model's final Linear layer emits *logits* — unbounded real numbers,
    # one per class. They are not probabilities and do not sum to 1. Softmax
    # exponentiates and normalises them into a proper distribution.
    probs = F.softmax(logits, dim=1)
    confidence, index = probs.max(dim=1)
    return int(index.item()), float(confidence.item())


def predict(model: nn.Module, image: Image.Image) -> tuple[str, float]:
    """Classify a galaxy image. Returns (class_name, confidence 0-1)."""
    tensor = _preprocess(image)

    # no_grad() tells PyTorch not to build the computation graph used for
    # backpropagation. Nothing here needs gradients, and skipping the graph is
    # faster and uses less memory. Grad-CAM below deliberately does NOT use it.
    with torch.no_grad():
        logits = model(tensor)

    index, confidence = _logits_to_result(logits)
    return CLASS_NAMES[index], confidence


def predict_with_heatmap(
    model: nn.Module, image: Image.Image
) -> tuple[str, float, Image.Image]:
    """Classify, and also produce a Grad-CAM overlay explaining the decision.

    Returns (class_name, confidence, overlay_image).

    Kept separate from predict() because the two have opposite requirements:
    plain prediction runs under torch.no_grad(); Grad-CAM needs the gradient
    graph in order to run a backward pass.
    """
    image = image.convert("RGB")
    tensor = _preprocess(image)

    with torch.no_grad():
        index, confidence = _logits_to_result(model(tensor))

    with _cam_lock:
        # Constructed per call rather than cached: GradCAM registers forward and
        # backward hooks on the model in its constructor and removes them on
        # exit, so scoping it with `with` keeps the shared model clean.
        # Target layer is the last block of layer4 — the deepest conv features,
        # where spatial detail is still present but semantics are strongest.
        with GradCAM(model=model, target_layers=[model.layer4[-1]]) as cam:
            # Targeting the PREDICTED class means the heatmap answers
            # "why did you say that?" rather than "where would you look for
            # some other class?".
            grayscale_cam = cam(
                input_tensor=tensor,
                targets=[ClassifierOutputTarget(index)],
            )[0]  # (224, 224) float32 in [0, 1]

        # Grad-CAM leaves gradients sitting on layer4's parameters. Nothing reads
        # them, but clearing keeps successive requests independent and tidy.
        model.zero_grad(set_to_none=True)

    # The overlay must be built from the UN-normalised image. show_cam_on_image
    # expects RGB floats in [0, 1]; feeding it the normalised tensor (which
    # ranges roughly -2.1 to 2.6) produces a lurid, meaningless picture.
    rgb = np.asarray(image.resize(IMAGE_SIZE), dtype=np.float32) / 255.0

    overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)  # uint8 HWC

    return CLASS_NAMES[index], confidence, Image.fromarray(overlay)


# ---------------------------------------------------------------------------
# Command-line harness — lets this file be tested with no web layer at all.
# ---------------------------------------------------------------------------


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: python {Path(__file__).name} <image-path>", file=sys.stderr)
        return 2

    image_path = Path(argv[1])
    if not image_path.is_file():
        print(f"error: no such file: {image_path}", file=sys.stderr)
        return 1

    print(f"Loading model from {MODEL_PATH} ...")
    model = load_model()

    image = Image.open(image_path)
    print(f"Loaded image {image_path.name}  size={image.size}  mode={image.mode}")

    name, confidence = predict(model, image)
    print(f"\n  predict()            -> {name}  ({confidence:.2%})")

    name, confidence, overlay = predict_with_heatmap(model, image)
    print(f"  predict_with_heatmap -> {name}  ({confidence:.2%})")

    out_path = Path(__file__).resolve().parent / "gradcam_out.png"
    overlay.save(out_path)
    print(f"\nGrad-CAM overlay written to {out_path}")
    print("Open it: the heat should sit on real galaxy structure, not a flat wash.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
