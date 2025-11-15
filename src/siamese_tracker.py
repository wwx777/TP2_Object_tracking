"""
Simple SiamFC-like tracker implementation.

This module provides a lightweight SiamFC-style tracker suitable as a prototype
for one-shot tracking in the existing project. It is intentionally self-contained
and does not require external pretrained Siamese weights (you may extend it to
load trained weights later).

API:
  - SiamFCTracker.init(frame, roi)  -> initialize with first frame and ROI
  - SiamFCTracker.update(frame)     -> returns (x,y,w,h) updated bbox
  - SiameseTracker.track_video(...) -> helper to run on a video (interactive ROI)

Notes:
  - This is a simple prototype that uses a small conv backbone and cross-correlation
    to compute the response map. It is intended for experiments and comparisons
    with your existing trackers (mean-shift / Hough / deep-mean-shift).
"""

import os
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _preprocess_frame(frame: np.ndarray, out_size: int = 255, device='cpu') -> torch.Tensor:
    """Convert BGR frame to normalized torch tensor resized to out_size (square).

    Returns tensor shape (1,3,H,W)
    """
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (out_size, out_size))
    img_resized = img_resized.astype(np.float32) / 255.0
    # Normalize with ImageNet stats (same as other modules)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_resized = (img_resized - mean) / std
    # HWC -> CHW
    img_chw = img_resized.transpose(2, 0, 1)
    tensor = torch.from_numpy(img_chw).unsqueeze(0).to(device)
    return tensor


class _SimpleBackbone(nn.Module):
    """Small conv backbone inspired by early Siamese trackers (AlexNet-like).

    Output: feature map (N, C, h, w)
    """

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.feature(x)


class SiamFCTracker:
    """Prototype SiamFC tracker.

    This implementation uses a small backbone and direct cross-correlation
    (implemented via conv2d) between the template feature map and the search
    feature map. It is a prototype for experiments and comparisons.
    """

    def __init__(self, device='cpu', exemplar_size=127, instance_size=255):
        self.device = device
        self.exemplar_size = exemplar_size
        self.instance_size = instance_size

        self.backbone = _SimpleBackbone().to(device)
        self.backbone.eval()

        # template feature (torch.Tensor) - set on init
        self.z_feat = None  # shape (1, C, kz, kz)
        self.template_box = None  # (x,y,w,h) in original frame coords
        self.current_box = None

    def init(self, frame: np.ndarray, roi: Tuple[int, int, int, int]):
        """Initialize tracker with first frame and ROI (x,y,w,h).

        We build a template by cropping the ROI, resizing to exemplar_size, and
        extracting features.
        """
        x, y, w, h = roi
        self.template_box = roi
        self.current_box = roi

        # Crop ROI and resize to exemplar size
        crop = frame[y:y + h, x:x + w]
        if crop.size == 0:
            raise ValueError('Invalid ROI crop')
        crop_resized = cv2.resize(crop, (self.exemplar_size, self.exemplar_size))
        z = _preprocess_frame(crop_resized, out_size=self.exemplar_size, device=self.device)
        with torch.no_grad():
            zf = self.backbone(z)  # (1, C, kz, kz)

        # use zf as template kernel (without gradients)
        # normalize template along channel dimension
        zf = zf.squeeze(0)  # (C, kz, kz)
        zf_norm = zf.view(zf.size(0), -1).norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
        zf = zf / zf_norm.view(-1, 1, 1)
        self.z_feat = zf.unsqueeze(0)  # (1, C, kz, kz)

    def _compute_response(self, frame: np.ndarray) -> np.ndarray:
        """Compute response map on full frame by treating template as conv kernel.

        Returns response map as numpy array (H_out, W_out) in float32.
        """
        x = _preprocess_frame(frame, out_size=self.instance_size, device=self.device)
        with torch.no_grad():
            xf = self.backbone(x)  # (1, C, hx, wx)

        # perform cross-correlation: conv2d(input=xf, weight=template)
        # weight shape: (out_channels=1, in_channels=C, kz, kz)
        response = F.conv2d(xf, self.z_feat.to(xf.device))  # (1,1,H_out,W_out)
        response = response.squeeze(0).squeeze(0).cpu().numpy()

        # Normalize response to [0,1]
        response = (response - response.min()) / (response.max() - response.min() + 1e-6)
        return response

    def update(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """Update and return new bbox (x,y,w,h) in original frame coordinates.

        This simple mapping rescales the argmax in the response map back to the
        original frame coordinates. The box size is kept equal to the initial ROI size.
        """
        if self.z_feat is None:
            raise RuntimeError('Tracker not initialized')

        response = self._compute_response(frame)  # H_out x W_out

        # find peak
        r_h, r_w = response.shape
        peak_idx = np.unravel_index(response.argmax(), response.shape)
        peak_y, peak_x = peak_idx

        # Map peak from response coordinates to frame coordinates
        frame_h, frame_w = frame.shape[:2]
        # response corresponds to features of instance_size -> scale factor
        scale_x = frame_w / r_w
        scale_y = frame_h / r_h

        center_x = int((peak_x + 0.5) * scale_x)
        center_y = int((peak_y + 0.5) * scale_y)

        # keep same box size as template_box
        tx, ty, tw, th = self.template_box
        new_x = max(0, min(center_x - tw // 2, frame_w - tw))
        new_y = max(0, min(center_y - th // 2, frame_h - th))
        self.current_box = (int(new_x), int(new_y), int(tw), int(th))
        return self.current_box


class SiameseTracker:
    """High-level wrapper to match the project's tracker interface.

    Methods:
      - initialize(frame, roi)
      - update(frame) -> (x,y,w,h)
      - track_video(...)
    """

    def __init__(self, video_path: str, device: str = 'cpu'):
        self.video_path = video_path
        self.device = device
        self.tracker = SiamFCTracker(device=device)
        self.initialized = False

    def select_roi(self, frame: np.ndarray):
        from .utils import ROISelector
        return ROISelector().select_roi(frame)

    def initialize(self, frame: np.ndarray, roi: Tuple[int, int, int, int]):
        self.tracker.init(frame, roi)
        self.initialized = True

    def update(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        if not self.initialized:
            raise RuntimeError('Tracker not initialized')
        return self.tracker.update(frame)

    def track_video(self, visualize=True, save_result=False, output_dir='results/siamfc'):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        if not ret:
            print('Cannot read video')
            return

        roi = self.select_roi(frame)
        self.initialize(frame, roi)

        if save_result:
            os.makedirs(output_dir, exist_ok=True)

        frame_idx = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            bbox = self.update(frame)
            # draw
            x, y, w, h = bbox
            out = frame.copy()
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if visualize:
                cv2.imshow('SiamFC Tracker', out)
            if save_result:
                from .utils import save_frame
                save_frame(out, frame_idx, output_dir)

            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                break
            frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()
