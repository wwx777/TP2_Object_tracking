

# Object Tracking Toolkit — Quick Overview

A small toolkit containing reference implementations of classical tracking algorithms, deep feature–enhanced variants, and simple tools for annotation and evaluation. Designed for coursework, teaching, and lightweight experimentation.

## What’s Included

### **Classical Trackers**

* **Mean Shift**

  * Hue histogram
  * Hue+Saturation histogram
  * Adaptive model update
* **Generalized Hough Transform**

  * R-table modeling from gradient orientations
  * Voting-based center localization
  * Optional Kalman-based motion prediction (predictive Hough)

### **Deep Feature Variants**

* **Deep Mean Shift**

  * Uses ResNet-50 (layer 3) feature maps extracted via PyTorch
  * Supports layer/channel selection
* (Optional) Deep Hough voting with CNN-based similarity maps

### **Modern Trackers**

* **SiameseFC** (via official GOT-10k toolkit)
* **OSTrack** (official transformer-based tracker with ViT-B MAE checkpoint)

### **Utilities**

* Manual ROI selector
* Visualization tools (Hue/backprojection, gradients, accumulator heatmaps)
* Basic annotation and evaluation scripts (IoU, CLE)

---

## Quick Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick Examples

### **Mean Shift (Hue)**

```python
from src.classical_tracker import ClassicalTracker

ClassicalTracker(
    video_path='Test-Videos/Antoine_Mug.mp4',
    method='meanshift',
    color_space='hue'
).track_video(visualize=False, save_result=True)
```

### **Hough Transform with Gradient Visualization**

```python
from src.classical_tracker import ClassicalTracker

ClassicalTracker(
    video_path='Test-Videos/VOT-ball.mp4',
    method='hough',
    gradient_threshold=30
).track_video(
    visualize=True,
    visualize_process=True
)
```

---

## Repository Structure

* **`src/classical_tracker.py`** — main controller & strategy pattern
* **`src/features.py`** — color & gradient feature extraction
* **`src/deep_tracker.py`** — deep feature extraction + Deep MeanShift
* **`src/siamese_tracker.py`** — wrapper for SiameseFC
* **`src/ostrack.py`** — wrapper for OSTrack
* **`test/basic_questions.ipynb`** — answers & demonstrations for project tasks

---

## Dependencies

* **OpenCV**
* **NumPy / SciPy**
* **Matplotlib**
* **PyTorch**
* OSTrack (official repo)
* SiamFC / GOT-10k toolkit

See `requirements.txt` for full list.

---

## Licensing Notice

This repository includes third-party components
(OSTrack, got10k-toolkit, siamfc-pytorch).
Their original **LICENSE** files are preserved and must remain intact when redistributing.

Project code is provided under the MIT License (see `LICENSE`).

---

