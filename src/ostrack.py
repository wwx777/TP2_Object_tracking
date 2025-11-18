# src/ostrack.py
"""
Real OSTrack wrapper for your classical-tracking project.

Directory assumption:
    project_root/
        src/
            ostrack.py        # this file
        OSTrack/              # official botaoye/OSTrack repo (git clone)

Usage in notebook:
    import sys
    sys.path.append('..')
    from src.ostrack import OSTrack

    VIDEO = globals().get('VIDEO_PATH_MUG', '../Test-Videos/Antoine_Mug.mp4')

    tracker = OSTrack(
        video_path=VIDEO,
        config_name='vitb_256_mae_ce_32x4_ep300',  # or vitb_384_...
        device='cuda:0',                           # or 'cpu'
        debug=False
    )
    tracker.track_video(visualize=True, save_result=False)
"""

import os
import sys
from typing import Tuple

import cv2
import numpy as np
import torch
import types as _types

# ---------------------------------------------------------------------
# 0. 兼容旧版 OSTrack 对 torch._six 的依赖（你的 PyTorch 比较新，已经删掉这个模块了）
# ---------------------------------------------------------------------
try:
    # 老版本是 from torch._six import string_classes
    from torch._six import string_classes  # type: ignore
except ModuleNotFoundError:
    # 手动注入一个假的 torch._six 模块，满足它需要的属性即可
    six_mod = _types.ModuleType("torch._six")
    six_mod.string_classes = (str,)
    six_mod.int_classes = (int,)
    six_mod.text_type = str
    sys.modules["torch._six"] = six_mod

# ---------------------------------------------------------------------
# 1. 把官方 OSTrack 仓库加入 sys.path
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))     # object_tracking/
OSTRACK_ROOT = os.path.join(PROJECT_ROOT, "OSTrack")             # object_tracking/OSTrack

if OSTRACK_ROOT not in sys.path:
    sys.path.insert(0, OSTRACK_ROOT)

# 现在 import 官方代码（按 tracking/test.py 的方式）
try:
    from lib.test.evaluation.tracker import Tracker as EvalTracker
    from lib.test.evaluation.environment import env_settings
except ImportError as e:
    raise ImportError(
        "Cannot import OSTrack's lib.test.evaluation. "
        "Please ensure the official OSTrack repo is cloned to project_root/OSTrack "
        "and that you are running inside the correct environment (依赖都装好，包括 torch 等)."
    ) from e


class OSTrackTracker:
    """
    实际调用预训练 OSTrack 模型的底层 tracker（真正做推理的那层）。

    公共接口：
        - init(frame, roi)
        - update(frame) -> (x, y, w, h)

    和你之前占位版本的接口保持一致，只是内部不再是 template matching，
    而是调用官方的 ViT 模型。
    """

    def __init__(
        self,
        config_name: str = "vitb_256_mae_ce_32x4_ep300",
        device: str = "cuda",
        debug: bool = False,
    ):
        self.config_name = config_name
        self.device = device
        self.debug = debug

        self.current_box: Tuple[int, int, int, int] | None = None
        self.initialized: bool = False

        # 创建 OSTrack 的环境配置（tracking/create_default_local_file.py 会写这个）
        env = env_settings()

        # EvalTracker: 根据 name + parameter_name 找对应配置和 checkpoint
        # name='ostrack' 对应 lib/test/tracker/ostrack.py
        self.eval_tracker = EvalTracker(
            name="ostrack",
            parameter_name=self.config_name,
            dataset_name="video_demo",   # 占位名称，不用于真实 benchmark
            run_id="online_demo",
            env=env,
        )

        # 真实的单目标 tracker 对象，有 initialize / track 方法
        self.tracker = self.eval_tracker.tracker

        # 有些实现提供 .to() 方法，可以显式指定 device
        if hasattr(self.tracker, "to"):
            try:
                self.tracker.to(device)
            except TypeError:
                # 如果不接受 device 参数，那就用它自己的默认逻辑
                pass

        if self.debug:
            print(f"[OSTrack] Loaded tracker with config={self.config_name}, device={self.device}")

    def init(self, frame: np.ndarray, roi: Tuple[int, int, int, int]):
        """
        用首帧和初始 bbox 初始化 OSTrack.

        Args:
            frame: BGR 图像, shape (H, W, 3)
            roi:   (x, y, w, h) 左上角 + 宽高
        """
        if frame is None:
            raise ValueError("Empty frame passed to OSTrackTracker.init()")

        x, y, w, h = roi
        H, W = frame.shape[:2]

        # clamp，防越界
        x = max(0, min(int(x), W - 1))
        y = max(0, min(int(y), H - 1))
        w = max(1, min(int(w), W - x))
        h = max(1, min(int(h), H - y))

        # OSTrack 内部默认用 RGB 图像
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # info 里传入 init_bbox，和官方 tracker 的 initialize 接口一致
        init_info = {"init_bbox": [float(x), float(y), float(w), float(h)]}

        self.tracker.initialize(rgb, init_info)

        self.current_box = (x, y, w, h)
        self.initialized = True

        if self.debug:
            print(f"[OSTrack] Initialized with bbox={self.current_box}")

    def update(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """
        在新帧上更新目标位置，返回新的 bbox (x, y, w, h).
        """
        if not self.initialized:
            raise RuntimeError("OSTrackTracker not initialized. Call init() first.")

        if frame is None:
            if self.debug:
                print("[OSTrack] Empty frame, returning previous bbox")
            return self.current_box

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        info = {}
        out = self.tracker.track(rgb, info)

        # 常见返回键：'target_bbox' 或 'pred_bbox'
        if "target_bbox" in out:
            x, y, w, h = out["target_bbox"]
        elif "pred_bbox" in out:
            x, y, w, h = out["pred_bbox"]
        else:
            if self.debug:
                print(f"[OSTrack] Unexpected output keys: {list(out.keys())}, fallback to previous bbox")
            return self.current_box

        self.current_box = (
            int(round(x)),
            int(round(y)),
            int(round(w)),
            int(round(h)),
        )

        if self.debug:
            print(f"[OSTrack] Updated bbox={self.current_box}")

        return self.current_box


class OSTrack:
    """
    高层接口：负责
      - 打开视频
      - 交互式选 ROI
      - 循环调用 OSTrackTracker.update()
      - 可选保存结果帧

    用法（notebook）：
        tracker = OSTrack(video_path=..., config_name=..., device=..., debug=...)
        tracker.track_video(visualize=True, save_result=False)
    """

    def __init__(
        self,
        video_path: str,
        config_name: str = "vitb_256_mae_ce_32x4_ep300",
        device: str = "cuda",
        debug: bool = False,
    ):
        self.video_path = video_path
        self.device = device
        self.debug = debug

        self.tracker = OSTrackTracker(
            config_name=config_name,
            device=device,
            debug=debug,
        )
        self.initialized: bool = False

    def select_roi(self, frame: np.ndarray):
        roi = cv2.selectROI(
            "Select Target - OSTrack",
            frame,
            fromCenter=False,
            showCrosshair=True,
        )
        cv2.destroyWindow("Select Target - OSTrack")
        return roi  # (x, y, w, h)

    def initialize(self, frame: np.ndarray, roi: Tuple[int, int, int, int]):
        self.tracker.init(frame, roi)
        self.initialized = True

    def update(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        if not self.initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() first.")
        return self.tracker.update(frame)

    def track_video(
        self,
        visualize: bool = True,
        save_result: bool = False,
        output_dir: str = "results/ostrack",
    ):
        """
        视频 tracking 主循环。

        说明：
        - 不会主动“加速视频”：不跳帧、不倍速，只是按模型推理速度 + waitKey(1) 自然跑。
        - 如果模型本身很快，你看到的画面可能会比真实帧率快，这是因为算得快，但代码不会故意跳帧。
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        try:
            ret, frame = cap.read()
            if not ret:
                print("Cannot read first frame")
                return

            # 选 ROI 并初始化
            roi = self.select_roi(frame)
            if roi[2] == 0 or roi[3] == 0:
                print("Invalid ROI selected")
                return

            print("\nInitializing OSTrack...")
            self.initialize(frame, roi)

            if save_result:
                os.makedirs(output_dir, exist_ok=True)

            frame_idx = 1
            fps_list = []

            print("Press ESC to quit | Press 'p' to pause")

            while True:
                start = cv2.getTickCount()
                ret, frame = cap.read()
                if not ret:
                    break

                bbox = self.update(frame)
                x, y, w, h = bbox

                out = frame.copy()
                cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 2)

                end = cv2.getTickCount()
                # 单纯计算本次循环的 FPS，不做任何帧选择操作
                delta = max(end - start, 1)
                fps = cv2.getTickFrequency() / delta
                fps_list.append(fps)
                avg_fps = float(np.mean(fps_list[-30:])) if fps_list else 0.0

                cv2.putText(
                    out,
                    "OSTrack",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    out,
                    f"Frame: {frame_idx}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    out,
                    f"FPS: {avg_fps:.1f}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                if visualize:
                    cv2.imshow("OSTrack", out)

                if save_result:
                    filename = f"frame_{frame_idx:04d}.jpg"
                    cv2.imwrite(os.path.join(output_dir, filename), out)

                # 不刻意加速；这一步只负责响应键盘
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("\nTracking interrupted by user")
                    break
                elif key == ord("p"):
                    print("\nPaused. Press any key to continue...")
                    cv2.waitKey(0)

                frame_idx += 1

            print("\nTracking completed")
            if fps_list:
                print(f"Average FPS: {np.mean(fps_list):.1f}")
        finally:
            try:
                cap.release()
            except Exception:
                pass
            cv2.destroyAllWindows()
            try:
                cv2.waitKey(1)
            except Exception:
                pass


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
    _tracker.track_video(visualize=True, save_result=False)
