# Test images

Sample galaxy images used to verify the inference pipeline outside Colab.

These are held-out **validation** images from Galaxy10 DECaLS — images the model
never trained on. The filename carries the ground-truth label, so a prediction
can be checked for *correctness*, not just plausibility.

Format: `val_<dataset index>_c<class index>_<class name>.png`, e.g.
`val_15261_c08_Edge-on-without-Bulge.png`.

The **class index** is the authoritative label — it indexes `CLASS_NAMES` in
`inference.py` directly. The trailing name is for human readability only, with
spaces replaced by hyphens; do not try to reverse that substitution, because
"Edge-on without Bulge" already contains a hyphen and the mapping is not
one-to-one. Compare on the index.

## How to regenerate them

Run this cell at the bottom of `AstralAI_Week1.ipynb` in Colab (after the
existing cells have run, so `images`, `labels`, `val_idx` and `CLASS_NAMES` are
in memory), then download the zip and extract it here.

```python
import os, zipfile
from PIL import Image

OUT_DIR = '/content/astralai_test_images'
os.makedirs(OUT_DIR, exist_ok=True)

# Pick 5 validation images spread across different classes.
picked, seen = [], set()
for i in val_idx:
    label = int(labels[i])
    if label not in seen:
        seen.add(label)
        picked.append(int(i))
    if len(picked) == 5:
        break

for i in picked:
    # The class INDEX is the real label; the hyphenated name is decoration.
    # Encoding the index makes the ground truth unambiguous even though the
    # name-to-filename mapping is lossy.
    c = int(labels[i])
    name = CLASS_NAMES[c].replace(' ', '-')
    path = os.path.join(OUT_DIR, f'val_{i:05d}_c{c:02d}_{name}.png')
    Image.fromarray(images[i]).save(path)      # raw 256x256x3 uint8, no matplotlib
    print('saved', path)

with zipfile.ZipFile('/content/astralai_test_images.zip', 'w') as zf:
    for f in sorted(os.listdir(OUT_DIR)):
        zf.write(os.path.join(OUT_DIR, f), f)

from google.colab import files
files.download('/content/astralai_test_images.zip')
```

Two things that matter here:

* `Image.fromarray(images[i])` writes the **raw array** straight to PNG. Do not
  use `plt.imsave` or `plt.savefig` — those add axes, margins and resampling,
  which changes the pixels the model sees.
* Saving as PNG (lossless) rather than JPEG avoids compression artefacts that
  would make the local result differ slightly from the Colab result.
