# src/ostrack.py
"""
Thin wrapper around the official OSTrack demo code.

Directory layout assumed:

    object_tracking/
        src/
            ostrack.py        # this file
        OSTrack/              # git clone https://github.com/botaoye/OSTrack

Usage in notebook:

    import sys
    sys.path.append('..')
    from src.ostrack import OSTrack

    VIDEO = globals().get('VIDEO_PATH_MUG', '../Test-Videos/Antoine_Mug.mp4')

    tracker = OSTrack(
        video_path=VIDEO,
        config_name='vitb_256_mae_ce_32x4_ep300',  # corresponds to a yaml under experiments/ostrack/
        device='cpu',                               # or 'cuda:0'
        debug=True
    )

    tracker.track_video(
        use_optional_box=True,     # True: select ROI first and use it as the initial bbox
        visualize=True,
        save_result=False,
        output_dir='../results/qX_ostrack'
    )
"""

import os
import sys
from typing import Tuple

import cv2
import numpy as np
import types as _types

# ---------------------------------------------------------------------
# 0. Compatibility shim for OSTrack's old dependency on torch._six
# ---------------------------------------------------------------------
try:
    from torch._six import string_classes  # type: ignore
except ModuleNotFoundError:
    six_mod = _types.ModuleType("torch._six")
    six_mod.string_classes = (str,)
    six_mod.int_classes = (int,)
    six_mod.text_type = str
    sys.modules["torch._six"] = six_mod

# ---------------------------------------------------------------------
# 1. Add the official OSTrack repository to sys.path
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
OSTRACK_ROOT = os.path.join(PROJECT_ROOT, "OSTrack")  # assume it's placed alongside `src`

if OSTRACK_ROOT not in sys.path:
    sys.path.insert(0, OSTRACK_ROOT)

try:
    from lib.test.evaluation.tracker import Tracker as EvalTracker
    from lib.test.evaluation.environment import env_settings
except ImportError as e:
    raise ImportError(
        "Cannot import OSTrack's lib.test.evaluation.\n"
        "Please ensure:\n"
        "1) The official OSTrack repository is cloned to project_root/OSTrack\n"
        "2) Your conda environment has dependencies installed per OSTrack/install.sh\n"
        "3) You have run `python tracking/create_default_local_file.py`"
    ) from e


class OSTrack:
    """
    High-level interface: call the official EvalTracker.run_video for demos.

    Keep the API consistent with the placeholder used in your notebook:
        tracker = OSTrack(...)
        tracker.track_video(...)
    """

    def __init__(
        self,
        video_path: str,
        config_name: str = "vitb_256_mae_ce_32x4_ep300",
        device: str = "cuda",
        debug: bool = False,
    ):
        self.video_path = video_path
        self.config_name = config_name
        self.device = device
        self.debug = debug

        # env_settings() mainly reads local.py; call it here to ensure it's configured
        _ = env_settings()

        # EvalTracker: official evaluation/demo high-level Tracker
        # Note: run_id must be None or an int, otherwise an AssertionError may occur
        self.eval_tracker = EvalTracker(
            name="ostrack",
            parameter_name=self.config_name,
            dataset_name="video_demo",  # just a label; not important for run_video
            run_id=None,
        )

        if self.debug:
            print(
                f"[OSTrack wrapper] Created EvalTracker("
                f"name='ostrack', param='{self.config_name}', dataset='video_demo')"
            )

    # ---------------- High-level API ----------------

    def _select_roi_on_first_frame(self) -> Tuple[float, float, float, float] | None:
        """Interactively select an ROI on the first frame using OpenCV."""
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            print("Cannot read first frame for ROI selection.")
            return None

        roi = cv2.selectROI(
            "Select Target - OSTrack (official)",
            frame,
            fromCenter=False,
            showCrosshair=True,
        )
        cv2.destroyWindow("Select Target - OSTrack (official)")

        x, y, w, h = roi
        if w <= 0 or h <= 0:
            print("Invalid ROI selected.")
            return None

        return float(x), float(y), float(w), float(h)

    def track_video(
        self,
        use_optional_box: bool = True,
        visualize: bool = True,
        save_result: bool = True,
        output_dir: str = "../results/ostrack",
    ):
        """
        Call the official EvalTracker.run_video.

        Parameters
        ----------
        use_optional_box : bool
            True  -> Select an ROI on the first frame and pass it to OSTrack as `optional_box`.
            False -> Do not pass an optional_box; let the official code handle initialization.
        visualize : bool
            True  -> Display tracking results via OSTrack's `cv2.imshow` windows.
            False -> Do not display windows, but frames can still be saved as PNGs.
        save_result : bool
            Whether to let OSTrack save results to its `tracking_results` directory
            and whether to save per-frame PNGs.
        output_dir : str
            Only meaningful when `save_result=True`. This function will create
            a `frames_png` subdirectory under `output_dir` to store per-frame PNGs.
        """
        import os
        import cv2

        # 0) Prepare directory for saving PNGs (tied to save_result)
        if save_result:
            frames_dir = output_dir
            os.makedirs(frames_dir, exist_ok=True)
            if self.debug:
                print(f"[OSTrack wrapper] Saving frame PNGs to: {frames_dir}")
        else:
            frames_dir = None
            if self.debug:
                print("[OSTrack wrapper] save_result=False, will NOT save frame PNGs.")

        # 1) Select ROI (if requested)
        if use_optional_box:
            optional_box = self._select_roi_on_first_frame()
            if optional_box is None:
                print("No valid ROI, aborting.")
                return
            if self.debug:
                print(f"[OSTrack wrapper] Using optional_box = {optional_box}")
        else:
            optional_box = None
            if self.debug:
                print("[OSTrack wrapper] Running without optional_box")

        # 2) Monkey-patch `cv2.imshow` to capture each frame and save PNGs
        #    - If visualize=False, do not open GUI windows; only save PNGs
        #    - Always restore the original imshow afterwards
        orig_imshow = getattr(cv2, "imshow", None)
        frame_idx = {"i": 0}  # use dict so it's mutable inside the closure

        def imshow_hook(winname, img):
            # Save PNG
            if frames_dir is not None:
                png_path = os.path.join(frames_dir, f"Frame_{frame_idx['i']:04d}.png")
                cv2.imwrite(png_path, img)
                frame_idx["i"] += 1
                if self.debug and frame_idx["i"] % 50 == 0:
                    print(f"[OSTrack wrapper] Saved frame {frame_idx['i']} to {png_path}")

            # Control whether to actually display the window
            if visualize and orig_imshow is not None:
                return orig_imshow(winname, img)
            else:
                # If not showing a window, just return
                return None

        # If imshow does not exist in the environment, do not patch it
        if orig_imshow is not None:
            cv2.imshow = imshow_hook
            if self.debug:
                print("[OSTrack wrapper] cv2.imshow has been patched for frame saving.")
        else:
            if self.debug:
                print("[OSTrack wrapper] cv2.imshow not found, cannot patch for frame saving.")

        # 3) Call the official run_video
        try:
            try:
                self.eval_tracker.run_video(
                    self.video_path,
                    optional_box=list(optional_box) if optional_box is not None else None,
                    save_results=bool(save_result),
                )
            except TypeError as e:
                # In case the current run_video signature differs slightly, fall back to a safer call
                if self.debug:
                    print(f"[OSTrack wrapper] run_video signature mismatch: {e}")
                    print("[OSTrack wrapper] Retrying with positional arguments only.")
                if optional_box is not None:
                    self.eval_tracker.run_video(self.video_path, list(optional_box))
                else:
                    self.eval_tracker.run_video(self.video_path)
        finally:
            # 4) Restore imshow to avoid affecting other code
            if orig_imshow is not None:
                cv2.imshow = orig_imshow
                if self.debug:
                    print("[OSTrack wrapper] cv2.imshow has been restored.")

        print("\n[OSTrack wrapper] Tracking finished (official run_video).")
        if frames_dir is not None:
            print(f"[OSTrack wrapper] Saved {frame_idx['i']} frame PNGs to: {frames_dir}")

   