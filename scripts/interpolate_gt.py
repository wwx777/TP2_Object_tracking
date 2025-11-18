#!/usr/bin/env python3
"""Interpolate sparse ground-truth keyframes to per-frame GT using linear interpolation.

Input: a CSV `gt_sparse.csv` with rows `frame,x,y,w,h` (may be non-consecutive frames)
Output: `gt_full.csv` with every frame in the provided range filled by linear interpolation

Usage:
  python scripts/interpolate_gt.py --in gt_sparse.csv --out gt_full.csv --start 1 --end 300

If `--end` omitted and `--video` provided, uses video frame count.
"""
import argparse
import csv
from collections import OrderedDict
import math
import os


def read_gt(path):
    data = OrderedDict()
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fr = int(row['frame'])
                x = float(row['x']); y = float(row['y']); w = float(row['w']); h = float(row['h'])
            except Exception:
                continue
            data[fr] = (x, y, w, h)
    return data


def write_gt(path, data):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'x', 'y', 'w', 'h'])
        for k in sorted(data.keys()):
            x, y, w, h = data[k]
            writer.writerow([k, int(round(x)), int(round(y)), int(round(w)), int(round(h))])


def interp(a, b, t):
    return a + (b - a) * t


def interpolate(sparse, start, end):
    # sparse: dict frame->(x,y,w,h)
    frames = sorted(sparse.keys())
    if not frames:
        return {}

    full = {}
    # ensure start/end coverage
    if start < frames[0]:
        # extend first
        for f in range(start, frames[0]):
            full[f] = sparse[frames[0]]
    # iterate segments
    for i in range(len(frames) - 1):
        f0, f1 = frames[i], frames[i+1]
        v0 = sparse[f0]; v1 = sparse[f1]
        for f in range(f0, f1 + 1):
            if f1 == f0:
                t = 0.0
            else:
                t = (f - f0) / (f1 - f0)
            x = interp(v0[0], v1[0], t)
            y = interp(v0[1], v1[1], t)
            w = interp(v0[2], v1[2], t)
            h = interp(v0[3], v1[3], t)
            full[f] = (x, y, w, h)
    # extend to end
    last = frames[-1]
    for f in range(last + 1, end + 1):
        full[f] = sparse[last]

    return full


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='infile', required=True, help='Input sparse gt CSV')
    p.add_argument('--out', dest='outfile', required=True, help='Output full gt CSV')
    p.add_argument('--start', type=int, default=1, help='Start frame (inclusive)')
    p.add_argument('--end', type=int, default=None, help='End frame (inclusive)')
    p.add_argument('--video', default=None, help='If provided, use video length as end if --end omitted')
    args = p.parse_args()

    sparse = read_gt(args.infile)
    if args.end is None:
        if args.video:
            import cv2
            cap = cv2.VideoCapture(args.video)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            args.end = max(args.start, total)
        else:
            # fallback: use last annotated frame
            if sparse:
                args.end = max(sparse.keys())
            else:
                args.end = args.start

    full = interpolate(sparse, args.start, args.end)
    write_gt(args.outfile, full)
    print(f'Wrote interpolated GT: {args.outfile} ({len(full)} frames)')


if __name__ == '__main__':
    main()
