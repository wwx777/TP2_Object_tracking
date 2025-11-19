"""
Feature extraction for tracking
Includes color histograms, gradients and other feature extraction utilities
"""
import cv2
import numpy as np
from pathlib import Path


def extract_color_histogram(roi, feature_type='hue', mask=None):
    if feature_type == 'hue':
        # Q1: single-channel Hue histogram
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Create mask: filter out pixels with saturation < 30 or value <20 or >235
        if mask is None:
            mask = cv2.inRange(hsv, 
                             np.array((0., 30., 20.)), 
                             np.array((180., 255., 235.)))
        
        # Compute Hue channel histogram
        hist = cv2.calcHist([hsv], [0], mask, [180], [0, 180])
        
    elif feature_type == 'hsv':
        # Q2 improvement: H+S two-channel histogram
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        if mask is None:
            mask = cv2.inRange(hsv, 
                             np.array((0., 30., 20.)), 
                             np.array((180., 255., 235.)))
        
        # Compute 2D histogram for H and S
        hist = cv2.calcHist([hsv], [0, 1], mask, 
                  [180, 256], [0, 180, 0, 256])
        
    elif feature_type == 'rgb':
        # RGB color histogram
        hist_b = cv2.calcHist([roi], [0], mask, [256], [0, 256])
        hist_g = cv2.calcHist([roi], [1], mask, [256], [0, 256])
        hist_r = cv2.calcHist([roi], [2], mask, [256], [0, 256])
        hist = np.concatenate([hist_b, hist_g, hist_r])
    
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")
    
    # Normalize to [0, 255]
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    
    return hist


def compute_backprojection(frame, hist, feature_type='hue'):
   
    if feature_type == 'hue':
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
        
    elif feature_type == 'hsv':
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0, 1], hist, 
                                 [0, 180, 0, 256], 1)
        
    elif feature_type == 'rgb':
        # RGB backprojection computed per-channel
        hist_b = hist[:256]
        hist_g = hist[256:512]
        hist_r = hist[512:]
        
        dst_b = cv2.calcBackProject([frame], [0], hist_b, [0, 256], 1)
        dst_g = cv2.calcBackProject([frame], [1], hist_g, [0, 256], 1)
        dst_r = cv2.calcBackProject([frame], [2], hist_r, [0, 256], 1)
        
        # Merge three channels
        dst = cv2.addWeighted(dst_b, 0.33, dst_g, 0.33, 0)
        dst = cv2.addWeighted(dst, 1.0, dst_r, 0.33, 0)
    
    return dst


def visualize_hue_and_backprojection(frame, hist, track_window=None, save_dir=None, frame_num=None):
    """
    Q2 visualization: show Hue channel and backprojection

    Args:
        save_dir: if provided, save visualization results
        frame_num: frame index
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue_channel = hsv[:, :, 0]
    
    # compute backprojection
    backproj = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
    
    # convert to drawable format
    hue_img = cv2.cvtColor(hue_channel, cv2.COLOR_GRAY2BGR)
    backproj_img = cv2.cvtColor(backproj, cv2.COLOR_GRAY2BGR)
    
    # draw tracking rectangle
    if track_window is not None:
        r, c, w, h = track_window
        cv2.rectangle(hue_img, (r, c), (r + w, c + h), (0, 255, 0), 2)
        cv2.rectangle(backproj_img, (r, c), (r + w, c + h), (0, 255, 0), 2)
    
    # show
    cv2.imshow('Hue Channel', hue_img)
    cv2.imshow('Back Projection', backproj_img)
    
    # Save visualization results if requested
    if save_dir is not None and frame_num is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(f"{save_dir}/Hue_{frame_num:04d}.png", hue_img)
        cv2.imwrite(f"{save_dir}/BackProj_{frame_num:04d}.png", backproj_img)
    
    return hue_img, backproj_img

def compute_gradients(frame, threshold=30):
    """
    Q3: compute gradient orientations and magnitudes

    Args:
        frame: input frame (BGR color image)
        threshold: gradient magnitude threshold; pixels below are masked out

    Returns:
        orientations: gradient direction (radians), shape (H, W)
        magnitudes: gradient magnitude, shape (H, W)
        mask: boolean mask of salient gradients, shape (H, W)
    """
    # Convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Compute gradients in x and y using Sobel operator
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude
    magnitudes = np.sqrt(grad_x**2 + grad_y**2)
    
    # Compute gradient orientation (radians)
    orientations = np.arctan2(grad_y, grad_x)
    
    # Create mask by thresholding gradient magnitude
    mask = magnitudes > threshold
    
    return orientations, magnitudes, mask


def visualize_gradients(frame, orientations, magnitudes, mask, window_name='Gradient Orientation'):
    """
    Q3: visualize gradient orientations
    Pixels masked out (non-salient gradients) are shown in red

    Args:
        frame: original frame
        orientations: gradient orientations
        magnitudes: gradient magnitudes
        mask: salient gradient mask
        window_name: window title
    """
    # create visualization image
    h, w = frame.shape[:2]
    
    # Method 1: map orientation to color (HSV)
    # H (hue) = orientation, S = 1, V = normalized magnitude
    orientation_normalized = (orientations + np.pi) / (2 * np.pi)  # normalize to [0, 1]
    orientation_hue = (orientation_normalized * 180).astype(np.uint8)  # convert to [0, 180] for OpenCV

    # Create HSV image
    hsv_img = np.zeros((h, w, 3), dtype=np.uint8)
    hsv_img[:, :, 0] = orientation_hue  # H: orientation
    hsv_img[:, :, 1] = 255  # S: max saturation
    
    # V: set brightness according to gradient magnitude
    magnitude_normalized = cv2.normalize(magnitudes, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hsv_img[:, :, 2] = magnitude_normalized
    
    # Convert to BGR
    orientation_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    
    # Masked-out pixels (non-salient gradients) are shown in red
    orientation_img[~mask] = [0, 0, 255]  # BGR: red
    
    cv2.imshow(window_name, orientation_img)
    
    return orientation_img


def visualize_gradient_magnitude(magnitudes, mask, window_name='Gradient Magnitude'):
    """
    Q3: visualize gradient magnitude
    """
    # Normalize to [0, 255]
    mag_normalized = cv2.normalize(magnitudes, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # create 3-channel image for display
    mag_img = cv2.cvtColor(mag_normalized, cv2.COLOR_GRAY2BGR)
    
    # Masked-out pixels displayed as red
    mag_img[~mask] = [0, 0, 255]
    
    cv2.imshow(window_name, mag_img)
    
    return mag_img

def render_gradient_quadrants(frame, orientations, magnitudes, mask, save_path=None):
    # 原图
    A = frame.copy()

    # Gradient orientation（灰度）
    ori_gray = ((np.mod(orientations + np.pi, 2*np.pi) / (2*np.pi)) * 255).astype(np.uint8)
    B = cv2.cvtColor(ori_gray, cv2.COLOR_GRAY2BGR)

    # Gradient norm（黑白/骨骼色）
    mag = magnitudes / (magnitudes.max() + 1e-6)
    mag_u8 = (mag * 255).astype(np.uint8)
    C = cv2.applyColorMap(mag_u8, cv2.COLORMAP_BONE)

    # Selected orientations（mask 外为红）
    # Reuse existing visualization but do not create a popup window
    orientation_normalized = (orientations + np.pi) / (2 * np.pi)
    H = (orientation_normalized * 180).astype(np.uint8)
    S = np.full_like(H, 255, np.uint8)
    V = (mag * 255).astype(np.uint8)
    hsv = cv2.merge([H, S, V])
    D = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    D[~mask] = (0, 0, 255)

    # Titles
    def title(img, text):
        im = img.copy()
        cv2.putText(im, text, (10, im.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        return im

    A = title(A, 'Original')
    B = title(B, 'Gradient orientation')
    C = title(C, 'Gradient norm')
    D = title(D, 'Selected orientations')

    # stack into 2x2 panel
    top = np.hstack([A, B])
    bottom = np.hstack([C, D])
    panel = np.vstack([top, bottom])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), panel)
    return panel


def visualize_hough_transform(frame, accumulator, search_region, detected_window, save_path=None):
    """
    Q4: visualize Hough Transform accumulator and detection results

    Args:
        frame: original frame
        accumulator: Hough Transform accumulator H(x) (search-region local coords)
        search_region: search region (r1, c1, r2, c2) in image coords
        detected_window: detected window (r, c, w, h) in image coords
        save_path: path to save visualization

    Returns:
        visualization: visualization image
    """
    if accumulator is None:
        return frame.copy()
    
    r1, c1, r2, c2 = search_region
    
    # 1. normalize accumulator to [0, 255]
    acc_normalized = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 2. 应用热力图颜色映射 (JET colormap)
    acc_heatmap = cv2.applyColorMap(acc_normalized, cv2.COLORMAP_JET)
    
    # 3. resize accumulator heatmap to search-region size
    search_h, search_w = c2 - c1, r2 - r1
    if acc_heatmap.shape[0] != search_h or acc_heatmap.shape[1] != search_w:
        acc_heatmap = cv2.resize(acc_heatmap, (search_w, search_h))
    
    # 4. 创建原图副本
    frame_with_search = frame.copy()
    
    # 5. overlay heatmap on search region (50% alpha)
    alpha = 0.5
    frame_with_search[c1:c2, r1:r2] = cv2.addWeighted(
        frame[c1:c2, r1:r2], 1 - alpha,
        acc_heatmap, alpha, 0
    )
    
    # 6. draw search region boundary (green)
    cv2.rectangle(frame_with_search, (r1, c1), (r2, c2), (0, 255, 0), 2, cv2.LINE_4)
    
    # 7. 画出检测到的目标窗口 (红色实线)
    if detected_window is not None:
        r, c, w, h = detected_window
        cv2.rectangle(frame_with_search, (r, c), (r + w, c + h), (0, 0, 255), 3)
    
    # 8. find accumulator max location and mark it
    max_val = accumulator.max()
    max_loc_local = np.unravel_index(accumulator.argmax(), accumulator.shape)
    max_y_local, max_x_local = max_loc_local
    max_x_abs = r1 + max_x_local
    max_y_abs = c1 + max_y_local
    
    # 画十字标记最大值位置
    cv2.drawMarker(frame_with_search, (max_x_abs, max_y_abs), 
                   (255, 255, 255), cv2.MARKER_CROSS, 20, 3)
    
    # 添加文字说明
    cv2.putText(frame_with_search, f'Max vote: {max_val:.0f}', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame_with_search, 'Hough Transform H(x)', 
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 255), 2)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), frame_with_search)
    
    return frame_with_search