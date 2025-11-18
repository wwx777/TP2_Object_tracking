# src/deep_tracker.py

"""
Q6: Deep learning-based tracking with CNN features.

Idea:
- Use a pre-trained CNN (ResNet-50) as feature extractor.
- Choose a middle layer (e.g. layer2 or layer3) as feature channel bank.
- Within that layer, select the K most discriminative channels for the current target
  (via Fisher score between target ROI and background, or simple variance).
- Build a similarity map (cosine similarity) between target template and every location
  in the feature map, then plug this map into Mean-Shift (exactly like color backprojection).
"""

import time
from dataclasses import dataclass, field
from typing import Tuple, Optional, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image


@dataclass
class DeepTrackState:
    track_window: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) in image coords
    feature_window: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) in feature coords
    selected_channels: Optional[np.ndarray] = None  # [K] channel indices (int)
    template: Optional[torch.Tensor] = None          # [1, K, h_t, w_t], on device
    template_norm: float = 1.0
    last_similarity_map: Optional[np.ndarray] = None  # HxW (image size), float32


class DeepTracker:
    """
    Q6: Deep Mean-shift Tracking with Cosine Similarity and CNN features.
    """

    def __init__(
        self,
        video_path: str,
        model_name: str = "resnet50",
        layer_name: str = "layer3",
        top_k_channels: int = 128,
        channel_selection: str = "variance",  # or 'fisher'
        device: str = "cpu",
        term_crit=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1),
    ):
        self.video_path = video_path
        self.model_name = model_name
        self.layer_name = layer_name
        self.top_k_channels = top_k_channels
        self.channel_selection = channel_selection.lower()
        self.device = torch.device(device)
        self.term_crit = term_crit

        # 状态
        self.state = DeepTrackState()

        # 准备 CNN 特征提取器
        self.model, self._feature_blob = self._build_feature_extractor(model_name, layer_name)
        self.model.to(self.device)
        self.model.eval()

        # 标准 ImageNet 预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # [0,1], CHW
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # utils 模块：沿用 classic tracker 的 ROISelector / 可视化 / 保存函数
        from .utils import ROISelector, visualize_tracking, save_frame, save_prediction, save_meta
        self.ROISelector = ROISelector
        self.visualize_tracking = visualize_tracking
        self.save_frame = save_frame
        self.save_prediction = save_prediction
        self.save_meta = save_meta

    # ----------------------------------------------------------------------
    # 构建 CNN 特征抽取器（带 hook 抓取中间层 feature map）
    # ----------------------------------------------------------------------
    def _build_feature_extractor(self, model_name, layer_name):
        if model_name.lower() == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif model_name.lower() == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        feature_blob = {}

        def hook_fn(module, input, output):
            feature_blob["feat"] = output.detach()

        # 注册 hook 到指定层
        try:
            layer = getattr(backbone, layer_name)
        except AttributeError:
            raise ValueError(f"Layer {layer_name} not found in {model_name}")
        layer.register_forward_hook(hook_fn)

        return backbone, feature_blob

    # ----------------------------------------------------------------------
    # 图像 -> tensor，跑一遍 CNN，取出 feature map
    # ----------------------------------------------------------------------
    def _extract_features(self, frame_bgr: np.ndarray) -> torch.Tensor:
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        x = self.transform(pil_img).unsqueeze(0).to(self.device)  # [1,3,H,W]
        with torch.no_grad():
            _ = self.model(x)
        feat = self._feature_blob["feat"]  # [1,C,Hf,Wf]
        return feat

    # ----------------------------------------------------------------------
    # 把 image coords 的 ROI 映射到 feature map coords
    # ----------------------------------------------------------------------
    @staticmethod
    def _map_window_to_feature(
        img_window: Tuple[int, int, int, int],
        img_size: Tuple[int, int],
        feat_size: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
        x, y, w, h = img_window
        H, W = img_size
        Hf, Wf = feat_size

        sx = W / float(Wf)
        sy = H / float(Hf)

        fx = int(round(x / sx))
        fy = int(round(y / sy))
        fw = max(1, int(round(w / sx)))
        fh = max(1, int(round(h / sy)))

        # clamp
        fx = max(0, min(Wf - 1, fx))
        fy = max(0, min(Hf - 1, fy))
        if fx + fw > Wf:
            fw = Wf - fx
        if fy + fh > Hf:
            fh = Hf - fy

        return fx, fy, fw, fh

    # ----------------------------------------------------------------------
    # 通道选择：variance 或 Fisher score
    # ----------------------------------------------------------------------
    def _select_channels(
        self,
        feat: torch.Tensor,       # [1,C,Hf,Wf]
        fwin: Tuple[int, int, int, int],  # feature window
    ) -> np.ndarray:
        _, C, Hf, Wf = feat.shape
        fx, fy, fw, fh = fwin

        # target ROI 特征: [C, Nh]
        target = feat[0, :, fy:fy+fh, fx:fx+fw].reshape(C, -1)

        if self.channel_selection == "variance":
            # 只看目标 ROI 上的方差
            var = target.var(dim=1, unbiased=False)  # [C]
            scores = var
        else:
            # Fisher score: (mu_f - mu_b)^2 / (var_f + var_b)
            target_mean = target.mean(dim=1)     # [C]
            target_var = target.var(dim=1, unbiased=False)  # [C]

            # background 是整个 feature map
            full = feat[0].reshape(C, -1)
            full_mean = full.mean(dim=1)
            full_var = full.var(dim=1, unbiased=False)

            # 简单 Fisher score
            eps = 1e-6
            scores = (target_mean - full_mean) ** 2 / (target_var + full_var + eps)

        # 选出 top-K 通道
        k = min(self.top_k_channels, C)
        topk = torch.topk(scores, k=k, largest=True).indices
        return topk.cpu().numpy()  # [K]

    # ----------------------------------------------------------------------
    # 构建 template：选通道后，把目标 ROI 特征保存下来
    # ----------------------------------------------------------------------
    def _build_template(
        self,
        feat: torch.Tensor,
        fwin: Tuple[int, int, int, int],
        channels: np.ndarray,
    ):
        fx, fy, fw, fh = fwin
        # [1, C, Hf, Wf] -> 选 K 个通道 -> 裁剪 ROI
        feat_sel = feat[:, channels, :, :]           # [1,K,Hf,Wf]
        template = feat_sel[:, :, fy:fy+fh, fx:fx+fw]  # [1,K,fh,fw]
        # 计算 template 范数
        with torch.no_grad():
            t_norm = torch.sqrt((template ** 2).sum()).item() + 1e-6
        return template, t_norm

    # ----------------------------------------------------------------------
    # 对一帧计算 similarity map（cosine similarity sliding window）
    # ----------------------------------------------------------------------
    def _compute_similarity_map(
        self,
        feat: torch.Tensor,          # [1,C,Hf,Wf]
        template: torch.Tensor,      # [1,K,fh,fw]
        template_norm: float,
        channels: np.ndarray,
    ) -> torch.Tensor:
        # feat_sel: [1,K,Hf,Wf]
        feat_sel = feat[:, channels, :, :]
        _, K, Hf, Wf = feat_sel.shape
        _, _, fh, fw = template.shape
        eps = 1e-6

        # numerator: conv2d(feat, template)
        # template 作为 kernel，需要 shape [out_channels, in_channels, kh, kw]，这里输出 1 个通道
        kernel = template.to(self.device)  # [1,K,fh,fw]
        with torch.no_grad():
            num = F.conv2d(feat_sel, kernel, stride=1)  # [1,1,Hs,Ws]
            # patch norm: sqrt( conv2d(feat^2, ones) )
            ones_kernel = torch.ones_like(kernel)
            patch_norm_sq = F.conv2d(feat_sel ** 2, ones_kernel, stride=1)
            patch_norm = torch.sqrt(patch_norm_sq + eps)  # [1,1,Hs,Ws]

            denom = template_norm * patch_norm + eps
            sim = num / denom  # [1,1,Hs,Ws]

            # 插值回 feature map 大小，方便和 image 对齐
            sim_up = F.interpolate(sim, size=(Hf, Wf), mode="bilinear", align_corners=False)
        return sim_up.squeeze(0).squeeze(0)  # [Hf,Wf]

    # ----------------------------------------------------------------------
    # 把 similarity map 从 feature coords 插值到 image coords
    # ----------------------------------------------------------------------
    @staticmethod
    def _resize_sim_to_image(sim_f: torch.Tensor, img_size: Tuple[int, int]) -> np.ndarray:
        H, W = img_size
        with torch.no_grad():
            sim_img = F.interpolate(
                sim_f.unsqueeze(0).unsqueeze(0), size=(H, W),
                mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0)
        sim_np = sim_img.cpu().numpy().astype(np.float32)  # HxW
        return sim_np

    # ----------------------------------------------------------------------
    # 公开接口：选择 ROI
    # ----------------------------------------------------------------------
    def select_roi(self, frame: np.ndarray):
        roi = self.ROISelector().select_roi(frame)
        return roi  # (x,y,w,h)

    # ----------------------------------------------------------------------
    # 初始化：选择 ROI -> 计算特征 -> 选通道 -> 构建 template
    # ----------------------------------------------------------------------
    def initialize(self, frame: np.ndarray, roi: Tuple[int, int, int, int]):
        H, W = frame.shape[:2]
        feat = self._extract_features(frame)    # [1,C,Hf,Wf]
        _, C, Hf, Wf = feat.shape

        fwin = self._map_window_to_feature(roi, img_size=(H, W), feat_size=(Hf, Wf))

        channels = self._select_channels(feat, fwin)
        template, t_norm = self._build_template(feat, fwin, channels)

        self.state.track_window = roi
        self.state.feature_window = fwin
        self.state.selected_channels = channels
        self.state.template = template
        self.state.template_norm = t_norm

    # ----------------------------------------------------------------------
    # 基于 similarity map + mean-shift 更新 track_window
    # ----------------------------------------------------------------------
    def update(self, frame: np.ndarray, visualize_backproj: bool = False):
        if self.state.template is None or self.state.selected_channels is None:
            return self.state.track_window

        H, W = frame.shape[:2]
        feat = self._extract_features(frame)
        _, _, Hf, Wf = feat.shape

        # 计算 similarity map（在 feature 上）
        sim_f = self._compute_similarity_map(
            feat,
            self.state.template,
            self.state.template_norm,
            self.state.selected_channels,
        )  # [Hf,Wf]

        # 插值到 image 大小，作为 backprojection
        sim_img = self._resize_sim_to_image(sim_f, img_size=(H, W))  # [H,W], float32

        # 归一化为 [0,255] uint8，交给 meanShift
        sim_norm = cv2.normalize(sim_img, None, 0, 255, cv2.NORM_MINMAX)
        backproj = sim_norm.astype(np.uint8)

        # mean-shift 跟踪
        x, y, w, h = self.state.track_window
        _, new_window = cv2.meanShift(backproj, (x, y, w, h), self.term_crit)
        self.state.track_window = new_window
        self.state.last_similarity_map = sim_img

        if visualize_backproj:
            # 可视化 similarity map
            disp = cv2.applyColorMap(sim_norm.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imshow("Q6: Similarity Backprojection", disp)

        return new_window

    # ----------------------------------------------------------------------
    # 公开方法：计算整张图像的 similarity map（归一化到 [0,1]，float32）
    # ----------------------------------------------------------------------
    def compute_similarity_map(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute the similarity backprojection (same shape as input image HxW),
        normalized to [0,1] float32. Requires `initialize` has been called
        to build `self.state.template` and `self.state.selected_channels`.
        """
        if self.state.template is None or self.state.selected_channels is None:
            raise RuntimeError("DeepTracker: template not initialized. Call initialize() first.")

        H, W = frame.shape[:2]
        feat = self._extract_features(frame)

        sim_f = self._compute_similarity_map(
            feat,
            self.state.template,
            self.state.template_norm,
            self.state.selected_channels,
        )  # [Hf,Wf], torch.Tensor on device

        sim_img = self._resize_sim_to_image(sim_f, img_size=(H, W))  # numpy HxW float32

        # normalize to [0,1]
        mn, mx = sim_img.min(), sim_img.max()
        if mx - mn < 1e-6:
            return np.zeros_like(sim_img, dtype=np.float32)
        sim_norm = (sim_img - mn) / (mx - mn)
        return sim_norm.astype(np.float32)

    # ----------------------------------------------------------------------
    # 主入口：整段视频跟踪
    # ----------------------------------------------------------------------
    def track_video(
        self,
        visualize: bool = True,
        save_result: bool = False,
        output_dir: str = "results/q6_deep_meanshift",
        visualize_backproj: bool = False,
    ):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read video")
            cap.release()
            return

        print("Q6 Step 1: Select ROI on first frame")
        roi = self.select_roi(frame)
        if roi is None or roi[2] <= 0 or roi[3] <= 0:
            print("Error: Invalid ROI")
            cap.release()
            return

        print("Q6 Step 2: Initialize DeepTracker (CNN features + template)")
        self.initialize(frame, roi)

        print("Q6 Step 3: Start deep mean-shift tracking")
        print("Press 's' to save frame, 'ESC' to exit")

        frame_count = 1
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 先复制用于可视化/保存
            frame_with_box = frame.copy()

            new_window = self.update(frame, visualize_backproj=visualize_backproj)

            # 画 bbox
            frame_with_box = self.visualize_tracking(
                frame, new_window,
                window_name="Q6: Deep Tracking" if visualize else None,
                color=(0, 255, 0), thickness=2
            )

            if save_result:
                self.save_frame(frame_with_box, frame_count, output_dir)
                try:
                    self.save_prediction(output_dir, frame_count, new_window)
                except Exception as e:
                    print(f"Failed to save prediction for frame {frame_count}: {e}")

            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                print("\nTracking stopped by user")
                break
            elif key == ord('s'):
                self.save_frame(frame_with_box, frame_count, output_dir)
                print(f"Saved frame {frame_count}")

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        total_time = time.time() - start_time
        frames_processed = max(0, frame_count - 1)
        if save_result:
            try:
                self.save_meta(output_dir, frames_processed, total_time)
            except Exception:
                pass

        print(f"\nQ6: Deep tracking completed. Total frames: {frames_processed}")
