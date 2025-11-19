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
        config_name='vitb_256_mae_ce_32x4_ep300',  # 对应 experiments/ostrack/ 下的 yaml
        device='cpu',                               # 或 'cuda:0'
        debug=True
    )

    tracker.track_video(
        use_optional_box=True,     # True: 先选 ROI，然后以该框作为初始 bbox
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
# 0. 兼容 OSTrack 代码里对 torch._six 的老依赖
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
# 1. 把官方 OSTrack 仓库加入 sys.path
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
OSTRACK_ROOT = os.path.join(PROJECT_ROOT, "OSTrack")  # 你已经放在和 src 并列了

if OSTRACK_ROOT not in sys.path:
    sys.path.insert(0, OSTRACK_ROOT)

try:
    from lib.test.evaluation.tracker import Tracker as EvalTracker
    from lib.test.evaluation.environment import env_settings
except ImportError as e:
    raise ImportError(
        "Cannot import OSTrack's lib.test.evaluation.\n"
        "请确认：\n"
        "1) 官方 OSTrack 仓库已克隆到 project_root/OSTrack\n"
        "2) 当前 conda 环境已经按 OSTrack/install.sh 安装好依赖\n"
        "3) 已经执行过 python tracking/create_default_local_file.py"
    ) from e


class OSTrack:
    """
    高层接口：直接调用官方 EvalTracker.run_video 做 demo。

    和你 notebook 里之前占位版的 API 尽量保持一致：
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

        # env_settings() 主要用来读 local.py；这里调用一下确保其正确
        _ = env_settings()

        # EvalTracker: 官方评测 / demo 用的高层 Tracker
        # 注意：run_id 必须是 None 或 int，否则你遇到的那个 AssertionError 就会出现
        self.eval_tracker = EvalTracker(
            name="ostrack",
            parameter_name=self.config_name,
            dataset_name="video_demo",  # 只是一个标签，对 run_video 不重要
            run_id=None,
        )

        if self.debug:
            print(
                f"[OSTrack wrapper] Created EvalTracker("
                f"name='ostrack', param='{self.config_name}', dataset='video_demo')"
            )

    # ---------------- 高层接口 ----------------

    def _select_roi_on_first_frame(self) -> Tuple[float, float, float, float] | None:
        """在第一帧上用 OpenCV 交互式选择 ROI."""
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
        调用官方 EvalTracker.run_video.

        Parameters
        ----------
        use_optional_box : bool
            True  -> 先在第一帧选 ROI，然后把这个 bbox 作为 `optional_box` 传给 OSTrack。
            False -> 不传 optional_box，完全交给官方代码（有的版本会要求在命令行指定 bbox）。
        visualize : bool
            True  -> 通过 OSTrack 的 cv2.imshow 显示跟踪结果窗口。
            False -> 不显示窗口，但仍然可以保存每一帧 PNG。
        save_result : bool
            是否让 OSTrack 把结果存到它自己的 tracking_results 目录下，
            同时也控制是否额外保存每一帧 PNG。
        output_dir : str
            仅当 save_result=True 时有意义。
            本函数会在 output_dir 下创建子目录 `frames_png` 保存每一帧 PNG。
        """
        import os
        import cv2

        # 0) 准备保存 PNG 的目录（与 save_result 绑定）
        if save_result:
            frames_dir = output_dir
            os.makedirs(frames_dir, exist_ok=True)
            if self.debug:
                print(f"[OSTrack wrapper] Saving frame PNGs to: {frames_dir}")
        else:
            frames_dir = None
            if self.debug:
                print("[OSTrack wrapper] save_result=False, will NOT save frame PNGs.")

        # 1) 选 ROI（如果需要）
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

        # 2) Monkey Patch cv2.imshow，用来截获每一帧并保存 PNG
        #    - 如果 visualize=False，则不再真正弹窗，只保存 PNG
        #    - 最后一定恢复原来的 imshow
        orig_imshow = getattr(cv2, "imshow", None)
        frame_idx = {"i": 0}  # 用 dict 是为了在闭包里可变

        def imshow_hook(winname, img):
            # 保存 PNG
            if frames_dir is not None:
                png_path = os.path.join(frames_dir, f"Frame_{frame_idx['i']:04d}.png")
                cv2.imwrite(png_path, img)
                frame_idx["i"] += 1
                if self.debug and frame_idx["i"] % 50 == 0:
                    print(f"[OSTrack wrapper] Saved frame {frame_idx['i']} to {png_path}")

            # 控制是否真正显示窗口
            if visualize and orig_imshow is not None:
                return orig_imshow(winname, img)
            else:
                # 不显示窗口时，直接返回即可
                return None

        # 如果环境里本来就没有 imshow，就不打补丁
        if orig_imshow is not None:
            cv2.imshow = imshow_hook
            if self.debug:
                print("[OSTrack wrapper] cv2.imshow has been patched for frame saving.")
        else:
            if self.debug:
                print("[OSTrack wrapper] cv2.imshow not found, cannot patch for frame saving.")

        # 3) 调用官方的 run_video
        try:
            try:
                self.eval_tracker.run_video(
                    self.video_path,
                    optional_box=list(optional_box) if optional_box is not None else None,
                    save_results=bool(save_result),
                )
            except TypeError as e:
                # 万一当前版本的 run_video 参数名稍微不一样，就走一个保底调用
                if self.debug:
                    print(f"[OSTrack wrapper] run_video signature mismatch: {e}")
                    print("[OSTrack wrapper] Retrying with positional arguments only.")
                if optional_box is not None:
                    self.eval_tracker.run_video(self.video_path, list(optional_box))
                else:
                    self.eval_tracker.run_video(self.video_path)
        finally:
            # 4) 一定要恢复 imshow，避免影响别的代码
            if orig_imshow is not None:
                cv2.imshow = orig_imshow
                if self.debug:
                    print("[OSTrack wrapper] cv2.imshow has been restored.")

        print("\n[OSTrack wrapper] Tracking finished (official run_video).")
        if frames_dir is not None:
            print(f"[OSTrack wrapper] Saved {frame_idx['i']} frame PNGs to: {frames_dir}")

   