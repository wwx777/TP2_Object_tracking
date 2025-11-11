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
        # 这些模型已经在 140 万张图片上训练好了！
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
            roi: (r, c, w, h) region of interest, None for whole frame
        
        Returns:
            features: (C, H, W) feature map
        """
        # Crop ROI if specified
        if roi is not None:
            r, c, w, h = roi
            frame = frame[c:c+h, r:r+w]
        
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
                           method='variance', top_k=64) -> List[int]:
        """
        Select best channels from feature map.
        
        Methods:
        1. 'variance': High variance channels (more discriminative)
        2. 'max_response': Channels with highest activation in ROI
        3. 'gradients': Channels with strong gradients (edges)
        
        Args:
            features: (C, H, W) feature map
            roi: (r, c, w, h) region of interest in feature space
            method: Channel selection method
            top_k: Number of channels to select
        
        Returns:
            channel_indices: List of selected channel indices
        """
        C, H, W = features.shape
        
        # Map ROI from image space to feature space
        # Assume features are downsampled uniformly
        scale_h = H / 224.0
        scale_w = W / 224.0
        r, c, w, h = roi
        r_feat = int(r * scale_w)
        c_feat = int(c * scale_h)
        w_feat = max(1, int(w * scale_w))
        h_feat = max(1, int(h * scale_h))
        
        # Clamp to feature map bounds
        r_feat = max(0, min(W - w_feat, r_feat))
        c_feat = max(0, min(H - h_feat, c_feat))
        
        # Extract ROI features
        roi_features = features[:, c_feat:c_feat+h_feat, r_feat:r_feat+w_feat]  # (C, h, w)
        
        if method == 'variance':
            # High variance = more discriminative
            channel_scores = roi_features.reshape(C, -1).var(dim=1).cpu().numpy()
        
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
        self.model_hist = None
        self.track_window = None
    
    def init(self, frame: np.ndarray, roi: Tuple[int, int, int, int]):
        """Initialize tracker with first frame."""
        # Register hook for target layer
        self.extractor.register_hooks(self.layer_name)
        
        # Extract features
        features = self.extractor.extract_features(frame)  # (C, H, W)
        
        # Select best channels
        self.selected_channels = self.extractor.select_best_channels(
            features, roi, method=self.channel_selection, top_k=self.top_k_channels
        )
        
        # Build feature histogram model
        self.model_hist = self._build_feature_histogram(features, roi)
        self.track_window = roi
    
    def _build_feature_histogram(self, features: torch.Tensor, 
                                 roi: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Build histogram from selected CNN features.
        
        Each channel is quantized to bins, creating a multi-dimensional histogram.
        """
        # Extract selected channels
        features_np = features[self.selected_channels].cpu().numpy()  # (K, H, W)
        
        # Map ROI to feature space
        C, H, W = features.shape
        scale_h = H / 224.0
        scale_w = W / 224.0
        r, c, w, h = roi
        r_feat = int(r * scale_w)
        c_feat = int(c * scale_h)
        w_feat = max(1, int(w * scale_w))
        h_feat = max(1, int(h * scale_h))
        r_feat = max(0, min(W - w_feat, r_feat))
        c_feat = max(0, min(H - h_feat, c_feat))
        
        # Extract ROI features
        roi_features = features_np[:, c_feat:c_feat+h_feat, r_feat:r_feat+w_feat]  # (K, h, w)
        
        # Quantize each channel to 16 bins
        bins = 16
        hist = np.zeros((self.top_k_channels, bins))
        
        for i, channel_features in enumerate(roi_features):
            # Normalize to [0, 1]
            feat_min = channel_features.min()
            feat_max = channel_features.max()
            if feat_max > feat_min:
                normalized = (channel_features - feat_min) / (feat_max - feat_min)
            else:
                normalized = np.zeros_like(channel_features)
            
            # Compute histogram
            hist[i], _ = np.histogram(normalized, bins=bins, range=(0, 1))
        
        # Normalize histogram
        hist = hist / (hist.sum() + 1e-6)
        return hist
    
    def update(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """Update tracking window using mean-shift."""
        # Extract features
        features = self.extractor.extract_features(frame)
        features_np = features[self.selected_channels].cpu().numpy()  # (K, H, W)
        
        # Compute back-projection (similarity to model histogram)
        backproj = self._compute_backprojection(features_np)
        
        # Apply mean-shift
        r, c, w, h = self.track_window
        ret, new_window = cv2.meanShift(backproj, (r, c, w, h), self.term_crit)
        
        self.track_window = new_window
        return new_window
    
    def _compute_backprojection(self, features_np: np.ndarray) -> np.ndarray:
        """Compute back-projection from CNN features."""
        K, H, W = features_np.shape
        backproj = np.zeros((H, W), dtype=np.float32)
        
        bins = 16
        
        for i in range(K):
            channel_features = features_np[i]
            
            # Normalize
            feat_min = channel_features.min()
            feat_max = channel_features.max()
            if feat_max > feat_min:
                normalized = (channel_features - feat_min) / (feat_max - feat_min)
            else:
                normalized = np.zeros_like(channel_features)
            
            # Quantize to bins
            quantized = np.clip((normalized * bins).astype(int), 0, bins - 1)
            
            # Look up histogram values
            channel_backproj = self.model_hist[i][quantized]
            backproj += channel_backproj
        
        # Normalize to [0, 255]
        backproj = (backproj / K * 255).astype(np.uint8)
        return backproj


# ====================================================================================
# Explanation: How to choose the best layer and channels
# ====================================================================================

"""
Q6: Layer and Channel Selection Strategy

1. **Best Layer Selection:**

   Criteria:
   a) Receptive Field (RF): Should match object size
      - Small objects (< 64px): Use early/mid layers (RF 50-100px)
      - Medium objects (64-128px): Use mid layers (RF 100-300px)
      - Large objects (> 128px): Use mid/late layers (RF 200-400px)
   
   b) Semantic Level:
      - Low-level (conv1, layer1): Edges, textures → Good for texture-rich objects
      - Mid-level (layer2, layer3): Object parts → Best for general tracking
      - High-level (layer4, fc): Whole object → Too abstract for tracking
   
   c) Spatial Resolution:
      - Higher resolution → Better localization precision
      - Lower resolution → More robust to deformation
   
   **Recommendation**: layer3 (ResNet) / conv4_3 (VGG)
   - Good balance of semantics and spatial resolution
   - RF = 267px (ResNet) / 52px (VGG)
   - Works for most tracking scenarios

2. **Best Channels Selection:**

   Methods:
   
   a) **Variance-based**: Select channels with high variance in ROI
      - High variance = more discriminative features
      - Works well for textured objects
      - Fast to compute
   
   b) **Max Response**: Select channels with highest activation
      - Strong response = relevant to object
      - Good for salient objects
      - May include background-activated channels
   
   c) **Gradient-based**: Select channels with strong edges
      - Strong gradients = structural information
      - Robust to illumination changes
      - Best for structured objects
   
   **Recommendation**: Variance-based or Gradient-based
   - Use top 32-64 channels (balance between discrimination and robustness)
   - Can combine multiple methods (ensemble selection)

3. **Integration into Mean-shift:**

   Traditional: Color histogram → Back-projection → Mean-shift
   Deep:       CNN features → Feature "histogram" → Mean-shift
   
   Advantages:
   - CNN features are more robust to illumination changes
   - Better semantic understanding of object vs background
   - Can track objects with similar colors
   
   Trade-offs:
   - Slower (CNN forward pass ~10-50ms)
   - Requires GPU for real-time tracking
   - Higher memory usage

4. **Integration into Hough Transform:**

   Traditional: Gradient orientation → R-Table → Voting
   Deep:       CNN feature gradients → R-Table → Voting
   
   Advantages:
   - More robust edge/structure detection
   - Better handling of cluttered backgrounds
   - Can capture semantic boundaries
   
   Implementation:
   - Use feature gradients instead of image gradients
   - Build R-Table from feature map edges
   - Vote using feature-based orientations
"""
