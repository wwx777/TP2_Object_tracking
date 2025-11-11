"""
Q6: Deep Learning-based Tracking using CNN Features

This module implements tracking using features from pre-trained deep networks.
We replace traditional features (color histograms/gradients) with CNN feature channels.

Key Questions:
1. How to choose the best layer?
2. How to choose the best channels within the chosen layer?
3. How to integrate deep features into Mean-shift or Hough Transform?
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class FeatureLayerInfo:
    """Information about a CNN layer for feature extraction."""
    layer_name: str
    layer_index: int
    feature_shape: Tuple[int, int, int]  # (C, H, W)
    receptive_field: int
    semantic_level: str  # 'low', 'mid', 'high'


class CNNFeatureExtractor:
    """
    Extract features from pre-trained CNN for tracking.
    
    Supports:
    - ResNet-50 (recommended for tracking)
    - VGG-16 (alternative)
    """
    
    def __init__(self, model_name='resnet50', device='cpu'):
        """
        Args:
            model_name: 'resnet50' or 'vgg16'
            device: 'cpu' or 'cuda'
        
        Note:
            使用 ImageNet 预训练模型（Transfer Learning）
            - pretrained=True: 自动下载 ImageNet 训练好的权重
            - 不需要重新训练：直接提取特征用于跟踪
            - eval() 模式：冻结参数，只做前向传播
        """
        self.device = device
        self.model_name = model_name
        
        # Load pre-trained model (使用 ImageNet 预训练权重)
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)  # ← ImageNet 预训练
            self.layers_info = self._get_resnet_layers()
        elif model_name == 'vgg16':
            self.model = models.vgg16(pretrained=True)     # ← ImageNet 预训练
            self.layers_info = self._get_vgg_layers()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model.to(device)
        self.model.eval()  # 评估模式：不训练，不更新权重
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Feature hooks
        self.features = {}
        self.hooks = []
    
    def _get_resnet_layers(self) -> List[FeatureLayerInfo]:
        """
        Define ResNet-50 layers for feature extraction.
        
        Layer selection criteria:
        - conv1: Low-level (edges, textures), RF=7x7
        - layer1: Low-level, RF=35x35
        - layer2: Mid-level (parts), RF=91x91
        - layer3: Mid-level (object parts), RF=267x267
        - layer4: High-level (whole object), RF=427x427
        """
        return [
            FeatureLayerInfo('conv1', 0, (64, 56, 56), 7, 'low'),
            FeatureLayerInfo('layer1', 1, (256, 56, 56), 35, 'low'),
            FeatureLayerInfo('layer2', 2, (512, 28, 28), 91, 'mid'),
            FeatureLayerInfo('layer3', 3, (1024, 14, 14), 267, 'mid'),
            FeatureLayerInfo('layer4', 4, (2048, 7, 7), 427, 'high'),
        ]
    
    def _get_vgg_layers(self) -> List[FeatureLayerInfo]:
        """Define VGG-16 layers for feature extraction."""
        return [
            FeatureLayerInfo('conv1_2', 3, (64, 224, 224), 4, 'low'),
            FeatureLayerInfo('conv2_2', 8, (128, 112, 112), 10, 'low'),
            FeatureLayerInfo('conv3_3', 15, (256, 56, 56), 24, 'mid'),
            FeatureLayerInfo('conv4_3', 22, (512, 28, 28), 52, 'mid'),
            FeatureLayerInfo('conv5_3', 29, (512, 14, 14), 116, 'high'),
        ]
    
    def register_hooks(self, layer_name: str):
        """Register forward hook to extract features from specific layer."""
        self.remove_hooks()
        
        if self.model_name == 'resnet50':
            target_layer = getattr(self.model, layer_name)
        elif self.model_name == 'vgg16':
            # VGG uses sequential features
            layer_idx = next(l.layer_index for l in self.layers_info if l.layer_name == layer_name)
            target_layer = self.model.features[layer_idx]
        
        def hook_fn(module, input, output):
            self.features[layer_name] = output.detach()
        
        hook = target_layer.register_forward_hook(hook_fn)
        self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.features = {}
    
    def extract_features(self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> torch.Tensor:
        """
        Extract CNN features from frame.
        
        Args:
            frame: BGR image (H, W, 3)
            roi: Not used - always extracts from whole frame for tracking
        
        Returns:
            features: (C, H, W) feature map from entire frame
        """
        # Always process the entire frame for tracking
        # (ROI selection happens in feature space, not image space)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        # Get features from hook
        layer_name = list(self.features.keys())[0]
        features = self.features[layer_name].squeeze(0)  # (C, H, W)
        
        return features
    
    def select_best_layer(self, object_size: Tuple[int, int]) -> str:
        """
        Select best layer based on object size.
        
        Rule of thumb:
        - Small objects (< 64x64): Use mid-level layers (layer2/conv3_3)
        - Medium objects (64-128): Use mid-level layers (layer2/layer3)
        - Large objects (> 128): Use high-level layers (layer3/layer4)
        
        Args:
            object_size: (width, height) of tracking object
        
        Returns:
            layer_name: Best layer name
        """
        w, h = object_size
        size = max(w, h)
        
        if self.model_name == 'resnet50':
            if size < 64:
                return 'layer2'  # RF=91, good for small objects
            elif size < 128:
                return 'layer3'  # RF=267, good for medium objects
            else:
                return 'layer3'  # Still layer3, layer4 too abstract
        else:  # vgg16
            if size < 64:
                return 'conv3_3'
            elif size < 128:
                return 'conv4_3'
            else:
                return 'conv4_3'
    
    def select_best_channels(self, features: torch.Tensor, roi: Tuple[int, int, int, int], 
                           method='variance', top_k=64, frame_shape=None) -> List[int]:
        """
        Select best channels from feature map.
        
        Args:
            features: (C, H, W) feature map
            roi: (x, y, w, h) region of interest in image space
            method: Channel selection method
            top_k: Number of channels to select
            frame_shape: (height, width) of original frame
        
        Returns:
            channel_indices: List of selected channel indices
        """
        C, H, W = features.shape
        
        # Default frame shape
        if frame_shape is None:
            frame_h, frame_w = 224, 224
        else:
            frame_h, frame_w = frame_shape
        
        # Map ROI from image space (x,y,w,h) to feature space
        scale_h = H / frame_h
        scale_w = W / frame_w
        x, y, w, h = roi
        # x->col, y->row
        x_feat = int(x * scale_w)
        y_feat = int(y * scale_h)
        w_feat = max(1, int(w * scale_w))
        h_feat = max(1, int(h * scale_h))
        
        # Clamp to feature map bounds
        x_feat = max(0, min(W - w_feat, x_feat))
        y_feat = max(0, min(H - h_feat, y_feat))
        
        # Extract ROI features [C, H, W] -> [C, h, w]
        roi_features = features[:, y_feat:y_feat+h_feat, x_feat:x_feat+w_feat]
        
        if method == 'variance':
            # Improved: Consider both variance within ROI and contrast with background
            roi_flat = roi_features.reshape(C, -1)
            roi_var = roi_flat.var(dim=1)  # Variance within ROI
            roi_mean = roi_flat.mean(dim=1)  # Mean of ROI
            
            # Sample background (expand ROI by 50% on each side)
            bg_x1 = max(0, x_feat - w_feat // 4)
            bg_y1 = max(0, y_feat - h_feat // 4)
            bg_x2 = min(W, x_feat + w_feat + w_feat // 4)
            bg_y2 = min(H, y_feat + h_feat + h_feat // 4)
            
            # Create mask for background (exclude ROI)
            mask = torch.ones((H, W), dtype=torch.bool, device=features.device)
            mask[y_feat:y_feat+h_feat, x_feat:x_feat+w_feat] = False
            
            # Get background features
            bg_features = features[:, bg_y1:bg_y2, bg_x1:bg_x2]
            bg_mask = mask[bg_y1:bg_y2, bg_x1:bg_x2]
            bg_flat = bg_features.reshape(C, -1)[:, bg_mask.reshape(-1)]
            bg_mean = bg_flat.mean(dim=1) if bg_flat.numel() > 0 else roi_mean
            
            # Contrast: absolute difference between ROI and background
            contrast = torch.abs(roi_mean - bg_mean)
            
            # Combined score: variance + contrast (both normalized)
            roi_var_norm = roi_var / (roi_var.max() + 1e-6)
            contrast_norm = contrast / (contrast.max() + 1e-6)
            channel_scores = (0.5 * roi_var_norm + 0.5 * contrast_norm).cpu().numpy()
        
        elif method == 'max_response':
            # High activation = strong response to object
            channel_scores = roi_features.reshape(C, -1).max(dim=1)[0].cpu().numpy()
        
        elif method == 'gradients':
            # Strong gradients = edges and structure
            roi_np = roi_features.cpu().numpy()
            channel_scores = np.zeros(C)
            for i in range(C):
                gx = cv2.Sobel(roi_np[i], cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(roi_np[i], cv2.CV_32F, 0, 1, ksize=3)
                grad_mag = np.sqrt(gx**2 + gy**2)
                channel_scores[i] = grad_mag.mean()
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Select top-k channels
        top_indices = np.argsort(channel_scores)[-top_k:][::-1]
        return top_indices.tolist()


class DeepMeanShiftTracker:
    """
    Mean-shift tracking using CNN features instead of color histograms.
    
    Replace: Color histogram → CNN feature "histogram" (feature distribution)
    """
    
    def __init__(self, extractor: CNNFeatureExtractor, 
                 layer_name: str, top_k_channels: int = 64,
                 channel_selection: str = 'variance',
                 term_crit=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)):
        """
        Args:
            extractor: CNN feature extractor
            layer_name: Layer to extract features from
            top_k_channels: Number of channels to use
            channel_selection: Method for channel selection
            term_crit: Mean-shift termination criteria
        """
        self.extractor = extractor
        self.layer_name = layer_name
        self.top_k_channels = top_k_channels
        self.channel_selection = channel_selection
        self.term_crit = term_crit
        
        self.selected_channels = None
        self.model_features = None  # ROI feature template (K-dim vector) - our "kernel"
        self.track_window = None
    
    def init(self, frame: np.ndarray, roi: Tuple[int, int, int, int]):
        """
        Initialize tracker with first frame.
        
        Args:
            frame: BGR image (H, W, 3)
            roi: (x, y, w, h) region of interest in image coordinates
        """
        # Keep ROI in (x, y, w, h) format for cv2.meanShift
        self.track_window = roi
        
        # Register hook for target layer
        self.extractor.register_hooks(self.layer_name)
        
        # Extract features from entire frame (not ROI)
        features = self.extractor.extract_features(frame)  # (C, H, W)
        
        # Select best channels using the ROI
        self.selected_channels = self.extractor.select_best_channels(
            features, roi, method=self.channel_selection, top_k=self.top_k_channels,
            frame_shape=(frame.shape[0], frame.shape[1])
        )
        
        # Store ROI feature template (direct features, not histogram!)
        self.model_features = self._extract_roi_features(features, roi, 
                                                         frame_shape=(frame.shape[0], frame.shape[1]))
    
    def _extract_roi_features(self, features: torch.Tensor, 
                              roi: Tuple[int, int, int, int], frame_shape=None) -> np.ndarray:
        """
        Extract and store ROI feature template directly (no histogram quantization).
        
        Returns:
            roi_template: Average feature vector in ROI (K,) - this is our "kernel"
        """
        # Extract selected channels
        features_np = features[self.selected_channels].cpu().numpy()  # (K, H, W)
        
        # Default frame shape
        if frame_shape is None:
            frame_h, frame_w = 224, 224
        else:
            frame_h, frame_w = frame_shape
        
        # Map ROI from image space (x,y,w,h) to feature space
        C, H, W = features.shape
        scale_h = H / frame_h
        scale_w = W / frame_w
        x, y, w, h = roi
        # x->col, y->row in feature map
        x_feat = int(x * scale_w)
        y_feat = int(y * scale_h)
        w_feat = max(1, int(w * scale_w))
        h_feat = max(1, int(h * scale_h))
        # Clamp to feature map bounds
        x_feat = max(0, min(W - w_feat, x_feat))
        y_feat = max(0, min(H - h_feat, y_feat))
        
        # Extract ROI features (features_np is [K, H, W], indexing is [K, row, col])
        roi_features = features_np[:, y_feat:y_feat+h_feat, x_feat:x_feat+w_feat]  # (K, h, w)
        
        # Compute average feature vector in ROI - this is our template/kernel
        # Shape: (K,) where K is number of selected channels
        roi_template = roi_features.mean(axis=(1, 2))  # Average over spatial dimensions
        
        # Normalize to unit vector for cosine similarity
        norm = np.linalg.norm(roi_template)
        if norm > 0:
            roi_template = roi_template / norm
        
        return roi_template
    
    def update(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Update tracking window using mean-shift.
        
        Returns:
            new_roi: (x, y, w, h) region of interest in image coordinates
        """
        # Extract features
        features = self.extractor.extract_features(frame)
        features_np = features[self.selected_channels].cpu().numpy()  # (K, H, W)
        
        # Compute back-projection using cosine similarity (not histogram lookup!)
        backproj = self._compute_similarity_map(features_np)
        
        # Resize backprojection to match frame size
        frame_h, frame_w = frame.shape[:2]
        backproj_resized = cv2.resize(backproj, (frame_w, frame_h))
        
        # Apply mean-shift with (x, y, w, h) format
        x, y, w, h = self.track_window
        ret, new_window = cv2.meanShift(backproj_resized, (x, y, w, h), self.term_crit)
        
        # Validate and fix the new window to ensure positive dimensions
        x_new, y_new, w_new, h_new = new_window
        
        # Ensure positive dimensions (meanShift can sometimes return weird values)
        if w_new <= 0 or h_new <= 0:
            # Keep the original window if meanShift failed
            print(f"⚠️  Warning: meanShift returned invalid window {new_window}, keeping previous window")
            new_window = self.track_window
        else:
            # Clip to frame boundaries
            x_new = max(0, min(x_new, frame_w - 1))
            y_new = max(0, min(y_new, frame_h - 1))
            w_new = min(w_new, frame_w - x_new)
            h_new = min(h_new, frame_h - y_new)
            new_window = (x_new, y_new, w_new, h_new)
        
        self.track_window = new_window
        return new_window  # (x, y, w, h)
    
    def _compute_similarity_map(self, features_np: np.ndarray) -> np.ndarray:
        """
        Compute similarity map using cosine similarity between each position and ROI template.
        This directly uses the CNN features as "kernels" - no histogram quantization!
        
        Think of it as: each spatial position has a K-dimensional feature vector,
        we compute how similar it is to the ROI's average feature vector.
        
        Returns:
            similarity_map: Cosine similarity map [0, 255]
        """
        K, H, W = features_np.shape
        
        # Reshape features: (K, H, W) -> (K, H*W) -> (H*W, K)
        features_flat = features_np.reshape(K, -1).T  # (H*W, K)
        
        # Normalize each feature vector to unit length for cosine similarity
        norms = np.linalg.norm(features_flat, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        features_normalized = features_flat / norms  # (H*W, K)
        
        # Compute cosine similarity with ROI template (already normalized in init)
        # similarity = features_normalized @ model_features
        # Shape: (H*W, K) @ (K,) -> (H*W,)
        similarity = features_normalized @ self.model_features  # (H*W,)
        
        # Reshape back to spatial dimensions
        similarity_map = similarity.reshape(H, W)
        
        # Convert from [-1, 1] to [0, 1] then to [0, 255]
        # High similarity (close to 1) -> bright (255)
        # Low similarity (close to -1) -> dark (0)
        similarity_map = (similarity_map + 1) / 2  # [-1,1] -> [0,1]
        similarity_map = np.clip(similarity_map * 255, 0, 255)
        
        return similarity_map.astype(np.uint8)


# ====================================================================================
# Explanation: How to choose the best layer and channels
# ====================================================================================


class DeepTracker:
    """
    Wrapper class for deep learning-based tracking with video processing capabilities.
    Provides the same interface as ClassicalTracker.
    """
    
    def __init__(self, video_path: str, model_name: str = 'resnet50', 
                 layer_name: str = 'layer3', top_k_channels: int = 64,
                 channel_selection: str = 'variance', device: str = 'cpu'):
        """
        Args:
            video_path: Path to input video
            model_name: CNN model ('resnet50' or 'vgg16')
            layer_name: Layer to extract features from
            top_k_channels: Number of channels to use
            channel_selection: Method for channel selection
            device: 'cpu' or 'cuda'
        """
        self.video_path = video_path
        
        # Initialize CNN feature extractor
        self.feature_extractor = CNNFeatureExtractor(
            model_name=model_name,
            device=device
        )
        
        # Initialize deep mean-shift tracker
        self.tracker = DeepMeanShiftTracker(
            extractor=self.feature_extractor,
            layer_name=layer_name,
            top_k_channels=top_k_channels,
            channel_selection=channel_selection
        )
        
        self.initialized = False
    
    def select_roi(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Let user select ROI interactively.
        
        Returns:
            roi: (x, y, w, h) in image coordinates
        """
        from .utils import ROISelector
        return ROISelector().select_roi(frame)
    
    def initialize(self, frame: np.ndarray, roi: Tuple[int, int, int, int]):
        """Initialize tracker with first frame."""
        self.tracker.init(frame, roi)
        self.initialized = True
    
    def update(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """Update tracking window."""
        if not self.initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() first.")
        return self.tracker.update(frame)
    
    def track_video(self, visualize=True, save_result=False, output_dir='results/deep_tracking', 
                   visualize_backproj=False):
        """Track object in video with interactive ROI selection."""
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read video")
            return

        print("Step 1: Select ROI")
        roi = self.select_roi(frame)

        print("Step 2: Initialize tracker")
        self.initialize(frame, roi)

        print("Step 3: Start tracking")
        print("Press 's' to save frame, 'b' to toggle backproj, 'ESC' to exit")

        frame_count = 1
        show_backproj = visualize_backproj
        
        if save_result:
            import os
            os.makedirs(output_dir, exist_ok=True)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            new_window = self.update(frame)
            
            # Draw tracking box (needed for both visualization and saving)
            from .utils import visualize_tracking
            frame_with_box = visualize_tracking(
                frame, new_window, window_name='Deep Tracking Result' if visualize else None,
                color=(0, 255, 0), thickness=2
            )
            
            if visualize:
                # Show backprojection for debugging
                if show_backproj:
                    features = self.tracker.extractor.extract_features(frame)
                    features_np = features[self.tracker.selected_channels].cpu().numpy()
                    backproj = self.tracker._compute_similarity_map(features_np)
                    backproj_resized = cv2.resize(backproj, (frame.shape[1], frame.shape[0]))
                    
                    # Convert to color heatmap (like classical meanshift visualization)
                    backproj_norm = cv2.normalize(backproj_resized, None, 0, 255, cv2.NORM_MINMAX)
                    backproj_colored = cv2.applyColorMap(backproj_norm.astype(np.uint8), cv2.COLORMAP_JET)
                    
                    # Show side-by-side: original frame + backprojection
                    combined = np.hstack([frame, backproj_colored])
                    cv2.imshow('Backprojection (press b to toggle)', combined)

            if save_result:
                from .utils import save_frame
                save_frame(frame_with_box, frame_count, output_dir)

            key = cv2.waitKey(60) & 0xFF
            if key == 27:
                print("\nTracking stopped by user")
                break
            elif key == ord('s'):
                from .utils import save_frame
                save_frame(frame_with_box, frame_count, output_dir)
                print(f"Saved frame {frame_count}")
            elif key == ord('b'):
                show_backproj = not show_backproj
                print(f"Backprojection visualization: {'ON' if show_backproj else 'OFF'}")

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print(f"\n✅ Tracking completed. Total frames: {frame_count}")
        if save_result:
            print(f"   Output directory: {output_dir}")
        print("=" * 60)
