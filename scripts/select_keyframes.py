#!/usr/bin/env python3
"""Select keyframes from a video using simple strategies.

Modes:
 - uniform: pick every `step` frames
 - frame-diff: pick frames with largest frame-to-frame difference (top-k or threshold)
 - flow: pick frames with largest average optical-flow magnitude (top-k)

Output: a text file with one 1-based frame index per line.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np


def uniform_select(total_frames, step):
    return list(range(1, total_frames + 1, step))


def frame_diff_select(video_path, k=None, threshold=None):
    cap = cv2.VideoCapture(str(video_path))
    prev = None
    diffs = []  # (frame_idx, diff_value)
    idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            d = np.mean(np.abs(gray.astype(np.float32) - prev.astype(np.float32)))
            diffs.append((idx, float(d)))
        prev = gray
        idx += 1
    cap.release()

    if threshold is not None:
        return [i for i, v in diffs if v >= threshold]
    if k is not None:
        diffs.sort(key=lambda x: x[1], reverse=True)
        sel = [i for i, _ in diffs[:k]]
        return sorted(sel)
    # fallback: return every 30th
    return list(range(1, idx, 30))


def flow_select(video_path, k=None):
    cap = cv2.VideoCapture(str(video_path))
    ret, prev = cap.read()
    if not ret:
        return []
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    mags = []
    idx = 2
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mags.append((idx, float(np.mean(mag))))
        prev_gray = gray
        idx += 1
    cap.release()

    if k is not None:
        mags.sort(key=lambda x: x[1], reverse=True)
        sel = [i for i, _ in mags[:k]]
        return sorted(sel)
    return [i for i, _ in mags if _ > 0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True, help='Path to video')
    p.add_argument('--mode', choices=['uniform', 'frame-diff', 'flow'], default='uniform')
    p.add_argument('--step', type=int, default=20, help='Step for uniform sampling')
    p.add_argument('--k', type=int, default=None, help='Top-k frames to select (for diff/flow)')
    p.add_argument('--threshold', type=float, default=None, help='Threshold for frame-diff')
    p.add_argument('--out', default='keyframes.txt', help='Output frames file (one frame index per line)')
    args = p.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise FileNotFoundError(args.video)

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if args.mode == 'uniform':
        frames = uniform_select(total, args.step)
    elif args.mode == 'frame-diff':
        frames = frame_diff_select(video, k=args.k, threshold=args.threshold)
    else:
        frames = flow_select(video, k=args.k)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'w') as f:
        for fr in sorted(set(frames)):
            f.write(f"{int(fr)}\n")

    print(f"Wrote {len(frames)} frames to {outp}")


if __name__ == '__main__':
    main()
