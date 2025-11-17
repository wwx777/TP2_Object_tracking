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
        """
        self.device = device
        self.model_name = model_name
        
        # Load pre-trained model 
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)  # ← ImageNet pre-trained
            self.layers_info = self._get_resnet_layers()
        elif model_name == 'vgg16':
            self.model = models.vgg16(pretrained=True)     # ← ImageNet pre-trained
            self.layers_info = self._get_vgg_layers()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model.to(device)
        self.model.eval() 
        
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
        return [
            # 保留原有的
            FeatureLayerInfo('conv1', 0, (64, 56, 56), 7, 'low'),
            FeatureLayerInfo('layer1', 1, (256, 56, 56), 35, 'low'),
            
            # 细分layer2 - 测试不同深度
            FeatureLayerInfo('layer2.0', 2, (512, 28, 28), 51, 'mid_early'),
            FeatureLayerInfo('layer2.2', 2, (512, 28, 28), 71, 'mid'),
            FeatureLayerInfo('layer2', 2, (512, 28, 28), 91, 'mid_late'),
            
            # 细分layer3
            FeatureLayerInfo('layer3.0', 3, (1024, 14, 14), 147, 'mid_high'),
            FeatureLayerInfo('layer3.3', 3, (1024, 14, 14), 207, 'high_early'),
            FeatureLayerInfo('layer3', 3, (1024, 14, 14), 267, 'high'),
            
            FeatureLayerInfo('layer4', 4, (2048, 7, 7), 427, 'high'),
        ]
    
    def _get_vgg_layers(self) -> List[FeatureLayerInfo]:
        """Define VGG-16 layers for feature extraction with more granularity."""
        return [
            # Block 1 - 低级特征（边缘、颜色）
            FeatureLayerInfo('conv1_1', 0, (64, 224, 224), 3, 'low'),
            FeatureLayerInfo('conv1_2', 2, (64, 224, 224), 4, 'low'),
            
            # Block 2 - 纹理特征
            FeatureLayerInfo('conv2_1', 5, (128, 112, 112), 8, 'low'),
            FeatureLayerInfo('conv2_2', 7, (128, 112, 112), 10, 'low_mid'),
            
            # Block 3 - 局部形状特征
            FeatureLayerInfo('conv3_1', 10, (256, 56, 56), 16, 'mid_early'),
            FeatureLayerInfo('conv3_2', 12, (256, 56, 56), 20, 'mid'),
            FeatureLayerInfo('conv3_3', 14, (256, 56, 56), 24, 'mid'),
            
            # Block 4 - 复杂形状和部件
            FeatureLayerInfo('conv4_1', 17, (512, 28, 28), 36, 'mid_high'),
            FeatureLayerInfo('conv4_2', 19, (512, 28, 28), 44, 'mid_high'),
            FeatureLayerInfo('conv4_3', 21, (512, 28, 28), 52, 'high'),
            
            # Block 5 - 高级语义特征
            FeatureLayerInfo('conv5_1', 24, (512, 14, 14), 84, 'high'),
            FeatureLayerInfo('conv5_2', 26, (512, 14, 14), 100, 'high'),
            FeatureLayerInfo('conv5_3', 28, (512, 14, 14), 116, 'high'),
        ]
        
    def register_hooks(self, layer_name: str):
        """Register forward hook to extract features from specific layer."""
        self.remove_hooks()
        
        if self.model_name == 'resnet50':
            # Support nested layer names like 'layer3.0'
            parts = layer_name.split('.')
            target_layer = self.model
            for part in parts:
                if part.isdigit():
                    target_layer = target_layer[int(part)]
                else:
                    target_layer = getattr(target_layer, part)
        elif self.model_name == 'vgg16':
            # VGG uses sequential features
            layer_idx = next(l.layer_index for l in self.layers_info if l.layer_name == layer_name)
            target_layer = self.model.features[layer_idx]
        
        def hook_fn(module, input, output):
            self.features[layer_name] = output.detach()
        
        hook = target_layer.register_forward_hook(hook_fn)
        self.hooks.append(hook)
    
    def select_best_channels(self, features: torch.Tensor, roi: Tuple[int, int, int, int], 
                        method: str = 'discrimination', top_k: int = 64,
                        frame_shape: Tuple[int, int] = None) -> List[int]:
        """Select channels that are discriminative for target vs background."""
        C, H, W = features.shape
        x, y, w, h = roi
        
        # Map coordinates
        if frame_shape is not None:
            frame_h, frame_w = frame_shape
            x = int(x * W / frame_w)
            y = int(y * H / frame_h)
            w = int(w * W / frame_w)
            h = int(h * H / frame_h)
        
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))
        
        # Extract target region
        target_region = features[:, y:y+h, x:x+w]
        target_mean = target_region.mean(dim=(1, 2))
        target_std = target_region.std(dim=(1, 2))
        
        # Extract multiple background regions (more robust!)
        bg_regions = []
        
        # Top region
        if y > h:
            bg_regions.append(features[:, max(0, y-h):y, x:x+w])
        
        # Bottom region  
        if y + h + h < H:
            bg_regions.append(features[:, y+h:min(H, y+2*h), x:x+w])
        
        # Left region
        if x > w:
            bg_regions.append(features[:, y:y+h, max(0, x-w):x])
        
        # Right region
        if x + w + w < W:
            bg_regions.append(features[:, y:y+h, x+w:min(W, x+2*w)])
        
        # Combine background statistics
        if bg_regions:
            bg_mean = torch.stack([r.mean(dim=(1, 2)) for r in bg_regions]).mean(dim=0)
            bg_std = torch.stack([r.std(dim=(1, 2)) for r in bg_regions]).mean(dim=0)
        else:
            # Fallback: use entire image
            mask = torch.ones_like(features[0], dtype=torch.bool)
            mask[y:y+h, x:x+w] = False
            bg_mean = features[:, mask].mean(dim=1)
            bg_std = features[:, mask].std(dim=1)
        
        if method == 'discrimination':
            # Signal-to-Noise Ratio with variance consideration
            # Strong response in target, weak response in background
            signal = target_mean
            noise = bg_mean + bg_std  # Background mean + variation
            
            snr = signal / (noise + 1e-6)
            top_channels = torch.topk(snr, k=min(top_k, C)).indices
            
        elif method == 'contrast':
            # Maximum contrast between target and background
            contrast = (target_mean - bg_mean).abs() / (target_std + bg_std + 1e-6)
            top_channels = torch.topk(contrast, k=min(top_k, C)).indices
            
        elif method == 'fisher':
            # Fisher discriminant ratio: between-class variance / within-class variance
            between_class = (target_mean - bg_mean).pow(2)
            within_class = target_std.pow(2) + bg_std.pow(2)
            fisher_score = between_class / (within_class + 1e-6)
            top_channels = torch.topk(fisher_score, k=min(top_k, C)).indices
            
        else:
            # Fallback: simple discrimination
            discrimination = (target_mean - bg_mean).abs()
            top_channels = torch.topk(discrimination, k=min(top_k, C)).indices
        
        return top_channels.cpu().tolist()

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
        import time
        from .utils import save_frame, save_prediction, save_meta
        
        start_time = time.time()
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
                save_frame(frame_with_box, frame_count, output_dir)
                try:
                    save_prediction(output_dir, frame_count, new_window)
                except Exception:
                    pass

            key = cv2.waitKey(60) & 0xFF
            if key == 27:
                print("\nTracking stopped by user")
                break
            elif key == ord('s'):
                save_frame(frame_with_box, frame_count, output_dir)
                try:
                    save_prediction(output_dir, frame_count, new_window)
                except Exception:
                    pass
                print(f"Saved frame {frame_count}")
            elif key == ord('b'):
                show_backproj = not show_backproj
                print(f"Backprojection visualization: {'ON' if show_backproj else 'OFF'}")

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        total_time = time.time() - start_time
        frames_processed = max(0, frame_count - 1)
        if save_result:
            try:
                save_meta(output_dir, frames_processed, total_time)
            except Exception:
                pass
            print(f"   Output directory: {output_dir}")
        print(f"\n✅ Tracking completed. Total frames: {frames_processed}")
        print("=" * 60)
        # Clean up any registered hooks in the feature extractor to avoid
        # accumulating forward hooks between runs (which can cause memory
        # leaks or duplicated outputs). Also clear CUDA cache if available.
        try:
            self.tracker.extractor.remove_hooks()
        except Exception:
            pass

        try:
            import torch
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
