# Evaluation & Annotation Guide (English)

Overview (Quick Steps)
- Run a tracker and write results to `results/evaluation/<method>_<video>` (that folder should contain `predictions.csv` and `meta.json`).
- Annotate ground truth (GT) and save as `results/gt_<video>/gt.csv` (CSV columns: `frame,x,y,w,h`; we recommend 1-based frame indexing).
- Use `test/evaluation.ipynb` or `src/evaluation.py` to compute metrics. By default the evaluation only computes metrics on frames where both a prediction and GT exist; to treat missing predictions as failures, see the union-style evaluation notes later in this document.

Ground-truth Annotation (Interactive & Semi-automatic)

1) Annotation scripts (included in this repo):
   - `scripts/annotate_video.py`: interactive annotator for drawing/reviewing boxes (supports keyframe-only mode, review mode, and auto-save).
   - `scripts/select_keyframes.py`: helper to automatically or semi-automatically pick keyframes from a video (useful for keyframe-only annotation).
   - `scripts/interpolate_gt.py`: generate per-frame GT by linearly interpolating between annotated keyframes.

2) Output format
- Example output directory: `results/gt_mug/`
- Required file: `gt.csv` (must include header line: `frame,x,y,w,h`)
- Convention: `x,y` are the top-left pixel coordinates, `w,h` are width and height in pixels; ensure `frame` indexing matches the frame numbering used in `predictions.csv` (commonly 1-based).

3) Interactive usage example (zsh)
```bash
# Launch the interactive GUI to annotate frames (or keyframe mode)
python scripts/annotate_video.py --video Test-Videos/Antoine_Mug.mp4 --output results/gt_mug \
    --frames-file optional_keyframes.txt   # optional: only show selected keyframes
```

Common keys in the annotator (if using the repo annotator):
- `SPACE`: accept current box and advance to next frame (common action)
- `a`: accept current box but do not advance (save current only)
- `n` / `p`: next / previous frame (manual navigation)
- `c`: cancel current box and redraw
- `q` or `ESC`: quit (note: confirmed boxes are auto-saved)

4) Generate per-frame GT by interpolating keyframes (example)
```bash
python scripts/interpolate_gt.py --keyframes results/gt_mug/keyframes.csv --out results/gt_mug/gt.csv
```
`interpolate_gt.py` linearly interpolates `x,y,w,h` between keyframes and writes `gt.csv` with a header row.

Keyframe-based annotation (every 20 frames)
------------------------------------------
If you prefer sparse annotation plus interpolation to produce per-frame GT, a fixed-step keyframe strategy works well (for example: annotate every 20 frames). The following workflow shows how to generate a keyframe list, annotate those frames, and interpolate to produce per-frame GT.

1) Generate keyframe list (every 20 frames) — example (zsh):

```bash
python - <<'PY'
import cv2
from pathlib import Path

video = 'Test-Videos/Antoine_Mug.mp4'
out_dir = Path('results/gt_mug')
out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(video)
if not cap.isOpened():
    raise SystemExit('Cannot open video')

n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

step = 20
keyframes_path = out_dir / 'keyframes.txt'
with open(keyframes_path, 'w') as f:
    for i in range(1, n_frames+1, step):
        f.write(f"{i}\n")

print('Wrote keyframes to', keyframes_path)
PY
```

2) Launch annotator using the keyframes list (only these frames will be shown):

```bash
python scripts/annotate_video.py --video Test-Videos/Antoine_Mug.mp4 --out_dir results/gt_mug --frames-file results/gt_mug/keyframes.txt
```

3) Interpolate keyframe annotations into per-frame GT (linear interpolation):

```bash
python scripts/interpolate_gt.py --keyframes results/gt_mug/keyframes.txt --annotations results/gt_mug/annotations.csv --out_dir results/gt_mug/gt.csv
```

Notes:
- `keyframes.txt`: one frame index per line (the script above generates `1,21,41,...`).
- `scripts/annotate_video.py` in `--frames-file` mode will only display those frames for annotation and will save results as `annotations.csv` (or `gt_partial.csv`, depending on the script implementation).
- `scripts/interpolate_gt.py` reads keyframe annotations and linearly interpolates `x,y,w,h` for intermediate frames, then writes `gt.csv` (per-frame GT).

Tips & cautions:
- A step size of 20 is a trade-off: larger steps (e.g. 30–40) reduce annotation time but increase interpolation error; smaller steps (e.g. 10) increase accuracy at the cost of more annotation work. Choose `step` based on target motion speed.
- Interpolation is linear (see `scripts/interpolate_gt.py`); for targets with fast non-linear motion or rapid scale changes, reduce step size or add extra keyframes near motion/scale changes.
- After interpolation, manually review several key frames to verify correctness (use `scripts/demo_gt_check.py` or open a few saved frames such as `results/gt_mug/Frame_XXXX.png` and overlay boxes).

Running trackers and saving predictions
(Examples below assume you run them from the repository root.)

- Mean Shift (Hue):
```bash
python - <<'PY'
from src.classical_tracker import ClassicalTracker

VIDEO='Test-Videos/Antoine_Mug.mp4'
OUT='results/evaluation/meanshift_mug'
tr = ClassicalTracker(video_path=VIDEO, method='meanshift', color_space='hue')
tr.track_video(visualize=False, save_result=True, output_dir=OUT)
PY
```

- Mean Shift (HS):
```bash
python - <<'PY'
from src.classical_tracker import ClassicalTracker

VIDEO='Test-Videos/Antoine_Mug.mp4'
OUT='results/evaluation/hs_mug'
tr = ClassicalTracker(video_path=VIDEO, method='meanshift', color_space='hsv')
tr.track_video(visualize=False, save_result=True, output_dir=OUT)
PY
```

- Adaptive (online histogram update):
```bash
python - <<'PY'
from src.classical_tracker import ClassicalTracker

VIDEO='Test-Videos/Antoine_Mug.mp4'
OUT='results/evaluation/adaptive_mug'
tr = ClassicalTracker(video_path=VIDEO, method='meanshift', color_space='hue', update_model=True, update_rate=0.1)
tr.track_video(visualize=False, save_result=True, output_dir=OUT)
PY
```

- Hough (base) / Predictive Hough (Kalman + adaptive update):
```bash
# Base Hough
python - <<'PY'
from src.classical_tracker import ClassicalTracker

VIDEO='Test-Videos/VOT-Basket.mp4'
OUT='results/evaluation/hough_basket'
tr = ClassicalTracker(video_path=VIDEO, method='hough')
tr.track_video(visualize=False, save_result=True, output_dir=OUT)
PY

# Predictive Hough
python - <<'PY'
from src.classical_tracker import ClassicalTracker

VIDEO='Test-Videos/VOT-Basket.mp4'
OUT='results/evaluation/hough_kalman_basket'
tr = ClassicalTracker(video_path=VIDEO, method='predictive_hough')
tr.track_video(visualize=False, save_result=True, output_dir=OUT)
PY
```

- Deep feature enhanced methods (Deep MeanShift / Deep Hough) example:
```bash
python - <<'PY'
from src.deep_tracker import DeepTracker

VIDEO='Test-Videos/Antoine_Mug.mp4'
OUT='results/evaluation/deep_meanshift_mug'
tr = DeepTracker(video_path=VIDEO, model_name='resnet50', layer_name='layer3', top_k_channels=64, device='cpu')
tr.track_video(visualize=False, save_result=True, output_dir=OUT)
PY
```

Evaluation (single results folder)
- Use `src/evaluation.evaluate()`:
```bash
python - <<'PY'
from src import evaluation

pred_csv = 'results/evaluation/meanshift_mug/predictions.csv'
gt_csv   = 'results/gt_mug/gt.csv'
res = evaluation.evaluate(pred_csv, gt_csv, cle_threshold=20.0)
print(res)
PY
```

Evaluation notes (reminder)
- By default `evaluate()` computes metrics only on frames where both a prediction and GT are present (i.e., the intersection frames). This is useful to measure prediction accuracy but does not reflect sparsity/missing predictions. To treat missing predictions as failures (more reflective of deployed performance), run a union-style evaluation—see the diagnostic scripts or ask me to generate one.
- CSV files must use the `frame,x,y,w,h` header and ensure `frame` indexing matches your GT.
- `save_prediction` currently writes coordinates as integers; if you need higher precision, switch to floats and update `_read_boxes_from_csv` to parse floats (I can help modify this if needed).

Please record the primary metric (e.g., `success_rate`) for each method/video in the table below and submit the results:

| Method / Video | Mug | Soccer | Basket | Car | Sun | Occl |
|---|---:|---:|---:|---:|---:|---:|
| Mean Shift (Hue) | xx.x | xx.x | xx.x | xx.x | xx.x | xx.x |
| Mean Shift (HS)  | xx.x | xx.x | xx.x | xx.x | xx.x | xx.x |
| Adaptive         | xx.x | xx.x | xx.x | xx.x | xx.x | xx.x |
| HS + Adaptive    | xx.x | xx.x | xx.x | xx.x | xx.x | xx.x |
| Hough (base)     | xx.x | xx.x | xx.x | xx.x | xx.x | xx.x |
| Hough (Kalman+update) | xx.x | xx.x | xx.x | xx.x | xx.x | xx.x |
| Deep MeanShift   | xx.x | xx.x | xx.x | xx.x | xx.x | xx.x |
| Deep Hough       | xx.x | xx.x | xx.x | xx.x | xx.x | xx.x |

Additional diagnostics & visualizations (optional)
- I can generate diagnostics for any `results/evaluation/<method>_<video>` folder: per-frame IoU/CLE curves, binary hit maps, and screenshots of outlier frames saved under `results/evaluation/<method>_<video>/diagnostics/`. If you'd like me to run this and generate files, tell me which methods and videos to process.

FAQ (short)
- Q: Why is `n_frames` small (e.g., 36) even though the video is long?
  A: Because `evaluate()` by default only counts frames in the intersection between predictions and GT. To penalize missing predictions as failures, use the union-style evaluation. See the diagnostics or ask me to produce union-style evaluation code.
- Q: I forgot to save annotations during labeling — what now?
  A: The annotator auto-saves confirmed boxes. If you force-quit, confirmed frames should still be written to `gt.csv` (or a temporary file). You can also use `scripts/demo_gt_check.py` to review and fix missing frames.


