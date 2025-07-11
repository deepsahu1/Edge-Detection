# 🚗 U-Net Edge Detection

PyTorch project that predicts razor-sharp object boundaries on street-scene images.  
A custom-trained **U-Net (ResNet-34 encoder)** does the heavy lifting; its logits are fused with Sobel + Canny maps for cleaner, unbroken edges.

<div align="center">

| Original → Prediction |
|-----------------------|
| ![](examples/RGB_070_comparison.png) |
| ![](examples/RGB_090_comparison.png) |
| ![](examples/RGB_138_comparison.png) |

<sub><sup>More results in <code>examples/</code> (10 curated side-by-side images)</sup></sub>

</div>

---

## ✨ Highlights

| What I built | Where to look |
|--------------|--------------|
| **End-to-end DL pipeline** | `train.py` – U-Net fine-tuning, BCE + Dice loss, on-the-fly Gaussian blur, per-epoch checkpoints |
| **Hybrid CV + DL logic** | `predict.py` – logits thresholded, resized to original res, OR-fused with Sobel/Canny for extra crispness |
| **Clean engineering** | Modular `src/` (typed, doc-stringed), minimal pinned `requirements.txt`, tidy repo tree |
| **Reproducibility** | Deterministic seeds, version-pinned deps, one-command demo |

---

## 🚀 Quick start (inference)

```bash
python -m venv .venv && source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# edit paths in predict.py __main__ block *or* call the function manually:
python - <<'PY'
from predict import predict
predict(
    input_dir="examples/originals",
    model_path="models/unet_biped_epoch_92.pth"
)
PY
```
## 🏋️‍♂️ Training from scratch
```bash
python train.py
```
train.py expects BIPED images in data/BIPED/imgs/train/rgbr/real
and matching edge masks in data/BIPED/edge_maps/train/rgbr/real.

Edit those two paths at the bottom of train.py if your dataset lives elsewhere.

Checkpoints drop into models/ after every epoch.

## 🗂 Repo layout 

```bash
src/            dataset.py, model.py (U-Net factory)
train.py        fine-tunes U-Net on BIPED masks
predict.py      generates edge maps + comparison collages
examples/       10 cherry-picked results for the README
models/         (git-ignored) trained .pth files go here
data/           (git-ignored) datasets live here
requirements.txt
README.md
```
