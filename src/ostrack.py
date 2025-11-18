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
        save_result: bool = False,
        output_dir: str = "results/ostrack",
    ):
        """
        调用官方 EvalTracker.run_video.

        Parameters
        ----------
        use_optional_box : bool
            True  -> 先在第一帧选 ROI，然后把这个 bbox 作为 `optional_box` 传给 OSTrack。
            False -> 不传 optional_box，完全交给官方代码（有的版本会要求在命令行指定 bbox）。
        visualize : bool
            这里只能通过 OSTrack 自己的可视化控制。大部分版本 run_video()
            总是开一个窗口显示跟踪结果，没有额外开关。
        save_result : bool
            是否让 OSTrack 把结果存到它自己的 tracking_results 目录下。
            注意：官方 video_demo.py 一般是通过 argparse 的参数控制保存路径，
            这里我们只能尽量模拟。
        output_dir : str
            仅当 save_result=True 时有意义。具体路径由 OSTrack 的环境设置决定，
            这里只是给一个提示，真正的结果目录还是看 lib/test/evaluation/local.py
        """

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

        # 2) 调用官方的 run_video
        # 官方 Tracker.run_video 的常见签名是：
        #     run_video(video: str, optional_box: list = None, save_results: bool = False)
        # 有的版本还会有 debug/vis 等参数，但我们只传最基本的几个，保证兼容性。
        try:
            # 注意：这里不做任何“加速视频”的处理，只是单纯调用 OSTrack 的原始逻辑
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

        print("\n[OSTrack wrapper] Tracking finished (official run_video).")


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) < 2:
        print("Usage: python -m src.ostrack <video_path>")
        _sys.exit(1)

    _video_path = _sys.argv[1]
    _tracker = OSTrack(
        video_path=_video_path,
        config_name="vitb_256_mae_ce_32x4_ep300",
        device="cuda",
        debug=True,
    )
    _tracker.track_video(use_optional_box=True, visualize=True, save_result=False)
