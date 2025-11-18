#!/usr/bin/env python3
"""Interactive video annotator that exports ground-truth CSV `frame,x,y,w,h`.

Features:
- Play/step through video frames
- Draw a rectangle with the mouse (click-drag)
- Keys: n (next), p (prev), s (save CSV), r (reset current), q (quit + save), space (accept/save current and next)
- Optionally load existing `predictions.csv` or `gt.csv` to prefill annotations

Usage:
  python scripts/annotate_video.py --video path/to/video.mp4 --out_dir results/q_annot

Output: writes `gt.csv` into `out_dir` with header `frame,x,y,w,h`.
"""
import argparse
import os
import csv
import cv2
from pathlib import Path


class Annotator:
    def __init__(self, video_path, out_dir, start_frame=1):
        self.video_path = video_path
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.cap = cv2.VideoCapture(str(video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.frame_idx = max(1, start_frame)
        self.drawing = False
        self.ix = self.iy = 0
        self.curr_box = None  # (x,y,w,h)
        self.annotations = {}  # frame -> (x,y,w,h)

        # Load existing GT if present
        gt_path = self.out_dir / 'gt.csv'
        if gt_path.exists():
            self._load_gt(gt_path)
        # optional frames list for keyframe-only annotation
        self.frames_list = None
        self.list_pos = 0
        # optional predictions for review mode
        self.predictions = None

    def _load_gt(self, path):
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    fnum = int(row['frame'])
                    x = int(row['x']); y = int(row['y']); w = int(row['w']); h = int(row['h'])
                except Exception:
                    continue
                self.annotations[fnum] = (x, y, w, h)

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.curr_box = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # show rubber-band rectangle
            self.curr_box = (min(self.ix, x), min(self.iy, y), abs(x - self.ix), abs(y - self.iy))
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.curr_box = (min(self.ix, x), min(self.iy, y), abs(x - self.ix), abs(y - self.iy))

    def _goto(self, idx):
        # goto frame index (1-based)
        if idx < 1:
            idx = 1
        if self.total_frames and idx > self.total_frames:
            idx = self.total_frames
        self.frame_idx = idx
        # set video position (0-based index)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx - 1)

    def set_frames_list(self, frames):
        """Provide a list of 1-based frames to annotate; the annotator will visit only these in order."""
        frames = [int(f) for f in sorted(set(frames)) if f >= 1]
        if not frames:
            return
        self.frames_list = frames
        self.list_pos = 0
        self._goto(self.frames_list[self.list_pos])

    def _next(self):
        """Advance to the next frame in the (optional) frames list or the next video frame.
        Returns True if the annotator moved to a next frame, False if already at the end.
        """
        if self.frames_list is not None:
            if self.list_pos + 1 < len(self.frames_list):
                self.list_pos += 1
                self._goto(self.frames_list[self.list_pos])
                return True
            return False
        else:
            # if total_frames is known and we're at or beyond the last frame, don't advance
            if self.total_frames and self.frame_idx >= self.total_frames:
                return False
            self._goto(self.frame_idx + 1)
            return True

    def _prev(self):
        """Move to previous frame. Returns True if moved, False if already at beginning."""
        if self.frames_list is not None:
            if self.list_pos - 1 >= 0:
                self.list_pos -= 1
                self._goto(self.frames_list[self.list_pos])
                return True
            return False
        else:
            if self.frame_idx <= 1:
                return False
            self._goto(self.frame_idx - 1)
            return True

    def load_predictions(self, preds_csv):
        """Load predictions CSV with columns frame,x,y,w,h for review mode."""
        preds = {}
        try:
            with open(preds_csv, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        fr = int(row['frame'])
                        x = int(row['x']); y = int(row['y']); w = int(row['w']); h = int(row['h'])
                    except Exception:
                        continue
                    preds[fr] = (x, y, w, h)
        except Exception:
            raise
        self.predictions = preds

    def _read_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx - 1)
        ret, frame = self.cap.read()
        return ret, frame

    def _save_csv(self):
        path = self.out_dir / 'gt.csv'
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'x', 'y', 'w', 'h'])
            for k in sorted(self.annotations.keys()):
                x, y, w, h = self.annotations[k]
                writer.writerow([int(k), int(x), int(y), int(w), int(h)])
        print(f"Saved GT to: {path}")

    def run(self):
        win = 'Annotator'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, self._mouse_cb)

        while True:
            ret, frame = self._read_frame()
            if not ret:
                print('End of video or cannot read frame')
                break

            # Use a small polling loop so that mouse callbacks update `self.curr_box` live
            key = None
            while True:
                display = frame.copy()

                # draw existing annotation for this frame
                if self.frame_idx in self.annotations:
                    x, y, w, h = self.annotations[self.frame_idx]
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # show prediction (review mode) if present and not yet accepted
                if self.predictions is not None and self.frame_idx in self.predictions and self.frame_idx not in self.annotations:
                    px, py, pw, ph = self.predictions[self.frame_idx]
                    cv2.rectangle(display, (px, py), (px + pw, py + ph), (255, 0, 255), 2)

                # draw current rubber-box
                if self.curr_box is not None:
                    x, y, w, h = self.curr_box
                    cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)

                cv2.putText(display, f'Frame: {self.frame_idx}/{self.total_frames}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(display, "Keys: n-next p-prev SPACE-accept s-save r-reset q-quit", (10, display.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                cv2.imshow(win, display)

                k = cv2.waitKey(30)
                if k != -1 and k != 255:
                    key = k & 0xFF
                    break

                # continue looping to update display while dragging
            if key == ord('n'):
                # advance (keep current box but do not auto-save unless SPACE)
                self._next()
                self.curr_box = None
            elif key == ord('p'):
                self._prev()
                self.curr_box = None
            elif key == 32:  # SPACE: accept current box and advance
                if self.curr_box is not None:
                    self.annotations[self.frame_idx] = self.curr_box
                elif self.predictions is not None and self.frame_idx in self.predictions:
                    # accept predicted box and advance
                    self.annotations[self.frame_idx] = self.predictions[self.frame_idx]
                # autosave after accepting
                try:
                    self._save_csv()
                except Exception:
                    pass
                moved = self._next()
                self.curr_box = None
                if not moved:
                    # We're at the last frame and couldn't advance — exit after saving
                    break
            elif key == ord('s'):
                # save CSV
                # if there is a current box, save it for current frame
                if self.curr_box is not None:
                    self.annotations[self.frame_idx] = self.curr_box
                self._save_csv()
            elif key == ord('r'):
                # reset current annotation
                if self.frame_idx in self.annotations:
                    del self.annotations[self.frame_idx]
                self.curr_box = None
            elif key == ord('q'):
                # quit and save
                if self.curr_box is not None:
                    self.annotations[self.frame_idx] = self.curr_box
                if self.annotations:
                    self._save_csv()
                break
            elif key == ord('a'):
                # accept current without advancing OR accept prediction
                if self.curr_box is not None:
                    self.annotations[self.frame_idx] = self.curr_box
                elif self.predictions is not None and self.frame_idx in self.predictions:
                    self.annotations[self.frame_idx] = self.predictions[self.frame_idx]
                # autosave after accepting
                try:
                    self._save_csv()
                except Exception:
                    pass
                self.curr_box = None
            else:
                # any other key: ignore
                pass

        cv2.destroyWindow(win)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True, help='Path to video file')
    p.add_argument('--out_dir', default='results/gt', help='Directory to save gt.csv')
    p.add_argument('--frames-file', default=None, help='Optional file with one frame index per line (1-based). If provided, annotator visits only these frames in order')
    p.add_argument('--review-preds', default=None, help='Optional predictions.csv to review (accept or correct).')
    p.add_argument('--start', type=int, default=1, help='Start frame index (1-based)')
    args = p.parse_args()
    annot = Annotator(args.video, args.out_dir, start_frame=args.start)

    # load frames-file if provided
    if args.frames_file:
        try:
            with open(args.frames_file) as f:
                frames = [int(l.strip()) for l in f if l.strip()]
            annot.set_frames_list(frames)
            print(f"Annotated will visit {len(frames)} frames from {args.frames_file}")
        except Exception as e:
            print(f"Failed to load frames file: {e}")

    # load predictions for review mode
    if args.review_preds:
        try:
            annot.load_predictions(args.review_preds)
            print(f"Loaded predictions from {args.review_preds} for review mode")
        except Exception as e:
            print(f"Failed to load predictions: {e}")

    annot.run()


if __name__ == '__main__':
    main()
