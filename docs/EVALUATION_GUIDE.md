# Evaluation Guide

Purpose
- Provide clear instructions so teammates can run the tracker methods on the provided videos, save predictions, and produce evaluateable results (Success Rate / Precision / Mean IoU / Mean CLE / FPS).

Quick overview
- Put results under `results/evaluation/<method>_<video>` (each run should create `predictions.csv` and `meta.json`).
- Use the evaluation notebook `test/evaluation.ipynb` or call the evaluation functions in `src/evaluation.py` to compute metrics.

Videos (placeholders)
- Mug: `Test-Videos/Antoine_Mug.mp4`
- Soccer: `Test-Videos/<Soccer-video>.mp4`  # replace with the actual filename if different
- Basket: `Test-Videos/VOT-Basket.mp4`
- Car: `Test-Videos/VOT-Car.mp4`
- Sun: `Test-Videos/<Sun-video>.mp4`      # replace with actual filename
- Occl: `Test-Videos/<Occl-video>.mp4`    # replace with actual filename (occlusion case)

How to run (examples you can copy/paste into zsh)

- Run a classical tracker (Mean-shift, Hough, etc.) and save results non-interactively. Replace `VIDEO_PATH` and `output_dir` accordingly.

```bash
# Example: Mean-shift (Hue only)
python - <<'PY'
from src.classical_tracker import ClassicalTracker
VIDEO = 'Test-Videos/Antoine_Mug.mp4'        # change to the target video
OUT = 'results/evaluation/meanshift_mug'
tr = ClassicalTracker(video_path=VIDEO, method='meanshift', color_space='hue')
tr.track_video(visualize=False, save_result=True, output_dir=OUT)
PY
```

- Dual-channel (H+S):

```bash
python - <<'PY'
from src.classical_tracker import ClassicalTracker
VIDEO = 'Test-Videos/Antoine_Mug.mp4'
OUT = 'results/evaluation/hs_mug'
tr = ClassicalTracker(video_path=VIDEO, method='meanshift', color_space='hsv')
tr.track_video(visualize=False, save_result=True, output_dir=OUT)
PY
```

- Adaptive mean-shift (enable model update):

```bash
python - <<'PY'
from src.classical_tracker import ClassicalTracker
VIDEO = 'Test-Videos/Antoine_Mug.mp4'
OUT = 'results/evaluation/adaptive_mug'
tr = ClassicalTracker(video_path=VIDEO, method='meanshift', color_space='hue', update_model=True, update_rate=0.1)
tr.track_video(visualize=False, save_result=True, output_dir=OUT)
PY
```

- Hough basic / predictive Hough (Kalman + update):

```bash
python - <<'PY'
from src.classical_tracker import ClassicalTracker
VIDEO = 'Test-Videos/VOT-Basket.mp4'
OUT = 'results/evaluation/hough_basket'
tr = ClassicalTracker(video_path=VIDEO, method='hough')
tr.track_video(visualize=False, save_result=True, output_dir=OUT)
PY
```

Predictive (Kalman + update):

```bash
python - <<'PY'
from src.classical_tracker import ClassicalTracker
VIDEO = 'Test-Videos/VOT-Basket.mp4'
OUT = 'results/evaluation/hough_kalman_basket'
tr = ClassicalTracker(video_path=VIDEO, method='predictive_hough')
tr.track_video(visualize=False, save_result=True, output_dir=OUT)
PY
```

- Deep trackers (run from `src/deep_tracker.py`):

```bash
python - <<'PY'
from src.deep_tracker import DeepTracker
VIDEO = 'Test-Videos/Antoine_Mug.mp4'
OUT = 'results/evaluation/deep_meanshift_mug'
tr = DeepTracker(video_path=VIDEO, model_name='resnet50', layer_name='layer3', top_k_channels=64, device='cpu')
tr.track_video(visualize=False, save_result=True, output_dir=OUT)
PY
```

Evaluation (single folder)
- You can evaluate one result folder using the evaluation function directly from Python. Adjust the `pred_csv` and `gt_csv` paths.

```bash
python - <<'PY'
from src import evaluation
pred_csv = 'results/evaluation/meanshift_mug/predictions.csv'
gt_csv   = 'results/gt_mug/gt.csv'   # ensure a GT file exists for the video
res = evaluation.evaluate(pred_csv, gt_csv, cle_threshold=20.0)
print(res)
PY
```

Batch evaluation (notebook)
- Open `test/evaluation.ipynb` and set `GT_CSV` and `RESULTS_ROOT` in cell 2, or edit the candidates list in cell 3 to point to specific result folders. Run cells 1–4.

Metrics and definitions
- Success Rate: fraction of frames with IoU > 0.5 (evaluated on overlapping frames by default).  
- Precision: fraction of frames with CLE < 20 px (threshold configurable).  
- mean_iou / mean_cle: averages across evaluated frames.  
- fps: computed from `meta.json` if present (frames / total_time).

Important notes / conventions
- `predictions.csv` and `gt.csv` must use columns: `frame,x,y,w,h`. Frame indexing should match (both 1-based is recommended).  
- By default `src/evaluation.evaluate()` only computes metrics on overlapping frames (frames present in both prediction and GT). If you want missing predictions to count as failures, use the union-style diagnostic script in `docs/` or ask the repo owner to enable the union option.
- `save_prediction` currently writes integer box coordinates; if you expect fractional boxes, please note that reading converts to integers—this can slightly change CLE/IoU.

Table template for teammates
- Copy this table into your reply after you run the experiment (replace `xx.x` with percentages/values). Use the folder naming convention `results/evaluation/<method>_<video>`.

| Method / Video | Mug | Soccer | Basket | Car | Sun | Occl |
|---|---:|---:|---:|---:|---:|---:|
| Mean Shift (Hue) | xx.x | xx.x | xx.x | xx.x | xx.x | xx.x |
| Mean Shift (HS)  | xx.x | xx.x | xx.x | xx.x | xx.x | xx.x |
| Adaptive         | xx.x | xx.x | xx.x | xx.x | xx.x | xx.x |
| HS + Adaptive    | xx.x | xx.x | xx.x | xx.x | xx.x | xx.x |
| Hough Basic      | xx.x | xx.x | xx.x | xx.x | xx.x | xx.x |
| Hough (Kalman+update) | xx.x | xx.x | xx.x | xx.x | xx.x | xx.x |
| Deep MeanShift   | xx.x | xx.x | xx.x | xx.x | xx.x | xx.x |
| Deep Hough       | xx.x | xx.x | xx.x | xx.x | xx.x | xx.x |

What to paste back to the team
- After running each experiment, paste one table row per method with the success rate (%) or the primary metric your team agreed on. If possible, also attach the `predictions.csv` file for quick debugging.

Where I saved diagnostics
- If you want, I can run union-style evaluation and save per-frame IoU/CLE plots under `results/evaluation/<method>_<video>/diagnostics/` for easier inspection.

Questions / support
- If the dataset filenames differ from those above, replace the video path accordingly. If you want me to (A) change `evaluate()` to treat missing predictions as failures automatically, or (B) change CSV read/write to float coordinates, tell me and I'll apply the patch and run a small test on one video.

Small reminder: run the commands from the project root (where `src/` and `test/` live) so imports work correctly.
