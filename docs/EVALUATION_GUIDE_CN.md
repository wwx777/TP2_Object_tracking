# 评估与标注指南（中文）

概要（快速步骤）
- 运行某个方法把结果写入：`results/evaluation/<method>_<video>`（该文件夹应包含 `predictions.csv` 与 `meta.json`）。
- 标注 GT 并保存为 `results/gt_<video>/gt.csv`（CSV 列为 `frame,x,y,w,h`，frame 建议用 1-based）。
- 使用 `test/evaluation.ipynb` 或 `src/evaluation.py` 计算指标。默认评估只在预测与 GT 有重叠的帧上计算；若要把“缺失预测”也算作失败，请参考文档后半的 union 评估说明。

GT 标注（交互式 & 半自动化）
1) 标注脚本（仓库内）：
   - `scripts/annotate_video.py`：交互式标注与复查（支持 keyframe-only、复查模式、自动保存）。
   - `scripts/select_keyframes.py`：从视频自动/半自动选择关键帧列表（用于只标注 keyframes）。
   - `scripts/interpolate_gt.py`：依据关键帧插值生成逐帧 GT（线性插值），用于稀疏 GT 扩充到每帧。

2) 标注输出格式
- 输出目录示例：`results/gt_mug/`
- 必要文件：`gt.csv`（必须有标题行：`frame,x,y,w,h`）
- 约定：`x,y,w,h` 中 `x,y` 为左上角坐标（像素），`w,h` 为宽度和高度（像素）；`frame` 建议保持与 `predictions.csv` 同样的帧编号基准（通常是 1-based）。

3) 交互式使用示例（zsh）
```bash
# 标注：打开交互 GUI 逐帧标注（或 keyframe 模式）
python scripts/annotate_video.py --video Test-Videos/Antoine_Mug.mp4 --output results/gt_mug \
    --frames-file optional_keyframes.txt   # 可选，若只标注关键帧
```

脚本常见按键（若你使用的是仓库内 annotator）：
- `SPACE`：接受当前框并前进到下一帧（常用）
- `a`：接受当前框但不前进（仅保存当前）
- `n` / `p`：前 / 后 一帧（手动翻页）
- `c`：取消当前框并重选
- `q` 或 `ESC`：退出（注意会自动保存已确认的框）

4) 从关键帧插值生成逐帧 GT（示例）
```bash
python scripts/interpolate_gt.py --keyframes results/gt_mug/keyframes.csv --out results/gt_mug/gt.csv
```
（`interpolate_gt.py` 会按帧号线性插值 x,y,w,h，并输出含 header 的 `gt.csv`。）

关键帧标注（每 20 帧）
---------------------------------
如果希望用“稀疏标注 + 插值”的方式快速生成逐帧 GT，可以采用固定步长的关键帧策略，例如“每 20 帧标注一个关键帧”。下面给出生成关键帧列表、使用标注脚本标注关键帧、以及插值生成逐帧 GT 的完整流程：

1) 生成关键帧列表（每 20 帧）——示例（zsh）：

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

2) 用关键帧列表启动标注器（只在这些帧显示并让你标注）

```bash
python scripts/annotate_video.py --video Test-Videos/Antoine_Mug.mp4 --out_dir results/gt_mug --frames-file results/gt_mug/keyframes.txt
```

3) 插值生成逐帧 GT（线性插值）

```bash
python scripts/interpolate_gt.py --keyframes results/gt_mug/keyframes.txt --annotations results/gt_mug/annotations.csv --out_dir results/gt_mug/gt.csv
```

说明：
- `keyframes.txt`：每行一个帧号（上面脚本生成的是 1,21,41,...）。
- `scripts/annotate_video.py` 在 `--frames-file` 模式下会只展示这些帧让你标注，并将标注保存为 `annotations.csv`（或 `gt_partial.csv`，取决于脚本实现）。
- `scripts/interpolate_gt.py` 会读取关键帧标注并按帧号线性插值出中间帧的 `x,y,w,h`，输出 `gt.csv`（逐帧）。

提示与注意事项：
- 步长 20 是一个折衷：更大步长（例如 30、40）能更省工作量但插值误差更大；更小步长（例如 10）能提高准确性但工作量增加。你可以根据视频运动速度调整 `step` 值。 
- 插值方法是线性的（在 `scripts/interpolate_gt.py` 中实现），若目标有快速非线性运动或经常变化的尺度，建议缩短步长或在关键转折处额外添加关键帧。
- 插值完成后，务必在若干关键帧处人工复核（使用 `scripts/demo_gt_check.py` 或直接打开部分帧查看 `results/gt_mug/Frame_XXXX.png` 并叠加框）。

运行 Tracker 并保存预测
（下面给出常用方法的示例命令，均在仓库根目录运行）

- Mean Shift（Hue）：
```bash
python - <<'PY'
from src.classical_tracker import ClassicalTracker
VIDEO='Test-Videos/Antoine_Mug.mp4'
OUT='results/evaluation/meanshift_mug'
tr = ClassicalTracker(video_path=VIDEO, method='meanshift', color_space='hue')
tr.track_video(visualize=False, save_result=True, output_dir=OUT)
PY
```

- Mean Shift（HS）：
```bash
python - <<'PY'
from src.classical_tracker import ClassicalTracker
VIDEO='Test-Videos/Antoine_Mug.mp4'
OUT='results/evaluation/hs_mug'
tr = ClassicalTracker(video_path=VIDEO, method='meanshift', color_space='hsv')
tr.track_video(visualize=False, save_result=True, output_dir=OUT)
PY
```

- Adaptive（在线更新直方图）：
```bash
python - <<'PY'
from src.classical_tracker import ClassicalTracker
VIDEO='Test-Videos/Antoine_Mug.mp4'
OUT='results/evaluation/adaptive_mug'
tr = ClassicalTracker(video_path=VIDEO, method='meanshift', color_space='hue', update_model=True, update_rate=0.1)
tr.track_video(visualize=False, save_result=True, output_dir=OUT)
PY
```

- Hough（基础） / Predictive Hough（Kalman + 更新）：
```bash
# 基础 Hough
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

- Deep 特征增强方法（Deep MeanShift / Deep Hough）示例：
```bash
python - <<'PY'
from src.deep_tracker import DeepTracker
VIDEO='Test-Videos/Antoine_Mug.mp4'
OUT='results/evaluation/deep_meanshift_mug'
tr = DeepTracker(video_path=VIDEO, model_name='resnet50', layer_name='layer3', top_k_channels=64, device='cpu')
tr.track_video(visualize=False, save_result=True, output_dir=OUT)
PY
```

评估（单一结果文件夹）
- 使用 `src/evaluation.evaluate()`：
```bash
python - <<'PY'
from src import evaluation
pred_csv = 'results/evaluation/meanshift_mug/predictions.csv'
gt_csv   = 'results/gt_mug/gt.csv'
res = evaluation.evaluate(pred_csv, gt_csv, cle_threshold=20.0)
print(res)
PY
```

评估注意点（重申）
- `evaluate()` 默认只在预测与 GT 的交集帧上计算指标（即如果多数帧没有预测，`n_frames` 会很小）。这有利于检查“预测准确度”，但不会反映“稀疏性/缺失预测”的问题。若你想把缺失也计为失败（更接近实际可用性），请运行 union-style 的诊断脚本（仓库内或我可生成）。
- CSV 格式必须为 `frame,x,y,w,h`，并确保 `frame` 编号与视频 GT 使用的基准一致。
- `save_prediction` 当前把坐标写入为整数；若你需要更高精度可以改为写浮点并在 `_read_boxes_from_csv` 中解析为 float（我可以帮你修改）。


- 请把每个方法对每个视频的 `success_rate`（或其它你们约定的主指标）填到下表并提交：

| 方法 / 视频 | Mug | Soccer | Basket | Car | Sun | Occl |
|---|---:|---:|---:|---:|---:|---:|
| Mean Shift（Hue） | xx.x | xx.x | xx.x | xx.x | xx.x | xx.x |
| Mean Shift（HS）  | xx.x | xx.x | xx.x | xx.x | xx.x | xx.x |
| Adaptive         | xx.x | xx.x | xx.x | xx.x | xx.x | xx.x |
| HS + Adaptive    | xx.x | xx.x | xx.x | xx.x | xx.x | xx.x |
| Hough（基础）    | xx.x | xx.x | xx.x | xx.x | xx.x | xx.x |
| Hough（Kalman+更新） | xx.x | xx.x | xx.x | xx.x | xx.x | xx.x |
| Deep MeanShift   | xx.x | xx.x | xx.x | xx.x | xx.x | xx.x |
| Deep Hough       | xx.x | xx.x | xx.x | xx.x | xx.x | xx.x |

额外诊断与可视化（可选）
- 我可以为指定 `results/evaluation/<method>_<video>` 生成 diagnostics：每帧的 IoU/CLE 曲线、二值命中图、并把离群帧截图保存到 `results/evaluation/<method>_<video>/diagnostics/`。如果你需要我代跑并生成这些文件，请告诉我要哪几个方法和视频。

FAQ（简短）
- Q: `n_frames` 为什么很小（比如 36）但视频很长？
  A: 这是因为 `evaluate()` 默认只在预测与 GT 的交集帧上统计；如需把缺失预测也计入失败，请使用 union-style 评估。详细区别见文中说明。
- Q: 标注时忘记保存怎么办？
  A: annotator 支持自动保存已确认框；如果强制退出，已确认帧会被写入 `gt.csv`（或临时文件），也可以用 `scripts/demo_gt_check.py` 来复查并补标。

