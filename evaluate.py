#!/usr/bin/env python3
"""CLI wrapper to evaluate predictions against ground-truth.

Example:
  python evaluate.py --pred_dir results/q1_basic --gt path/to/gt.csv

This script expects `predictions.csv` and optional `meta.json` in the prediction directory.
"""
import argparse
import os
from src import evaluation


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pred_dir', required=True, help='Directory with predictions.csv (and optional meta.json)')
    p.add_argument('--gt_csv', required=True, help='Ground-truth CSV file with columns frame,x,y,w,h')
    p.add_argument('--cle', type=float, default=20.0, help='CLE threshold in pixels')
    args = p.parse_args()

    pred_csv = os.path.join(args.pred_dir, 'predictions.csv')
    if not os.path.exists(pred_csv):
        raise FileNotFoundError(f'Predictions CSV not found: {pred_csv}')

    res = evaluation.evaluate(pred_csv, args.gt_csv, cle_threshold=args.cle)

    print('\n=== Evaluation Summary ===')
    print(f"Frames evaluated: {res['n_evaluated_frames']}")
    print(f"Success rate (IoU>0.5): {res['success_rate']*100:.2f}%")
    print(f"Precision (CLE<{args.cle}px): {res['precision']*100:.2f}%")
    print(f"Mean IoU: {res['mean_iou']:.4f}")
    print(f"Mean CLE: {res['mean_cle']:.2f} px")
    if res.get('fps') is not None:
        print(f"FPS (from meta.json): {res['fps']:.2f}")
    else:
        print("FPS: meta.json not found; run tracker with save_result=True to generate timing info.")


if __name__ == '__main__':
    main()
