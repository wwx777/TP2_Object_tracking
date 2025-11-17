"""Evaluation utilities: IoU, CLE, FPS and a small evaluator that reads CSVs.

Predictions: CSV with columns `frame,x,y,w,h` saved by trackers into the output directory
(default file name `predictions.csv`). Ground-truth: same CSV format.

Usage: import functions here or run the CLI `evaluate.py` at project root.
"""
import csv
import json
import os
import math
from typing import Tuple, Dict


def _read_boxes_from_csv(path: str) -> Dict[int, Tuple[int, int, int, int]]:
    boxes = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame = int(row['frame'])
                x = int(row['x']); y = int(row['y']); w = int(row['w']); h = int(row['h'])
            except Exception:
                continue
            boxes[frame] = (x, y, w, h)
    return boxes


def iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Compute IoU between two boxes in (x,y,w,h) format."""
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB
    x1A, y1A = xA, yA
    x2A, y2A = xA + wA, yA + hA
    x1B, y1B = xB, yB
    x2B, y2B = xB + wB, yB + hB

    inter_x1 = max(x1A, x1B)
    inter_y1 = max(y1A, y1B)
    inter_x2 = min(x2A, x2B)
    inter_y2 = min(y2A, y2B)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    areaA = max(0, wA) * max(0, hA)
    areaB = max(0, wB) * max(0, hB)
    union = areaA + areaB - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area) / float(union)


def center_error(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB
    cxA = xA + wA / 2.0
    cyA = yA + hA / 2.0
    cxB = xB + wB / 2.0
    cyB = yB + hB / 2.0
    return math.hypot(cxA - cxB, cyA - cyB)


def evaluate(pred_csv: str, gt_csv: str, cle_threshold: float = 20.0) -> Dict:
    """Evaluate predictions vs ground-truth CSVs.

    Returns a dict with fields: success_rate, precision, mean_iou, mean_cle, fps (if meta found),
    n_frames, n_evaluated
    """
    preds = _read_boxes_from_csv(pred_csv)
    gts = _read_boxes_from_csv(gt_csv)

    frames = sorted(set(preds.keys()) & set(gts.keys()))
    if not frames:
        raise RuntimeError('No overlapping frames between predictions and ground-truth')

    ious = []
    cles = []
    for f in frames:
        p = preds[f]
        g = gts[f]
        i = iou(p, g)
        c = center_error(p, g)
        ious.append(i)
        cles.append(c)

    successes = sum(1 for v in ious if v > 0.5)
    precision = sum(1 for v in cles if v < cle_threshold)
    n = len(frames)

    # try to read FPS from meta.json in same dir as pred_csv
    fps = None
    pred_dir = os.path.dirname(pred_csv)
    meta_path = os.path.join(pred_dir, 'meta.json')
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
                if 'frames' in meta and 'total_time' in meta and meta['total_time'] > 0:
                    fps = float(meta['frames']) / float(meta['total_time'])
        except Exception:
            fps = None

    results = {
        'n_frames': n,
        'n_evaluated_frames': n,
        'success_rate': successes / n,
        'precision': precision / n,
        'mean_iou': sum(ious) / n,
        'mean_cle': sum(cles) / n,
        'fps': fps
    }
    return results


if __name__ == '__main__':
    # Minimal CLI for quick testing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True, help='Predictions CSV path')
    parser.add_argument('--gt', required=True, help='Ground-truth CSV path')
    parser.add_argument('--cle', type=float, default=20.0, help='CLE threshold in pixels')
    args = parser.parse_args()
    res = evaluate(args.pred, args.gt, cle_threshold=args.cle)
    print('Evaluation results:')
    for k, v in res.items():
        print(f'  {k}: {v}')
