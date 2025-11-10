"""
Feature extraction for tracking
包括颜色直方图、梯度等特征提取方法
"""
import cv2
import numpy as np


def extract_color_histogram(roi, feature_type='hue', mask=None):
    """
    提取颜色直方图
    
    Args:
        roi: ROI区域 (BGR图像)
        feature_type: 'hue', 'hsv', 'rgb'
        mask: 可选的掩码
        
    Returns:
        hist: 归一化的直方图
    """
    if feature_type == 'hue':
        # Q1: 单通道Hue直方图
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 创建掩码：过滤掉饱和度<30, 亮度<20或>235的像素
        if mask is None:
            mask = cv2.inRange(hsv, 
                             np.array((0., 30., 20.)), 
                             np.array((180., 255., 235.)))
        
        # 计算Hue通道直方图
        hist = cv2.calcHist([hsv], [0], mask, [180], [0, 180])
        
    elif feature_type == 'hsv':
        # Q2改进: H+S双通道直方图
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        if mask is None:
            mask = cv2.inRange(hsv, 
                             np.array((0., 30., 20.)), 
                             np.array((180., 255., 235.)))
        
        # 计算H和S的二维直方图
        hist = cv2.calcHist([hsv], [0, 1], mask, 
                          [180, 256], [0, 180, 0, 256])
        
    elif feature_type == 'rgb':
        # RGB颜色直方图
        hist_b = cv2.calcHist([roi], [0], mask, [256], [0, 256])
        hist_g = cv2.calcHist([roi], [1], mask, [256], [0, 256])
        hist_r = cv2.calcHist([roi], [2], mask, [256], [0, 256])
        hist = np.concatenate([hist_b, hist_g, hist_r])
    
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")
    
    # 归一化到[0, 255]
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    
    return hist


def compute_backprojection(frame, hist, feature_type='hue'):
    """
    计算反向投影
    
    Args:
        frame: 当前帧 (BGR图像)
        hist: 目标直方图
        feature_type: 'hue', 'hsv', 'rgb'
        
    Returns:
        dst: 反向投影图像 (权重图)
    """
    if feature_type == 'hue':
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
        
    elif feature_type == 'hsv':
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0, 1], hist, 
                                 [0, 180, 0, 256], 1)
        
    elif feature_type == 'rgb':
        # RGB反向投影需要分别计算
        hist_b = hist[:256]
        hist_g = hist[256:512]
        hist_r = hist[512:]
        
        dst_b = cv2.calcBackProject([frame], [0], hist_b, [0, 256], 1)
        dst_g = cv2.calcBackProject([frame], [1], hist_g, [0, 256], 1)
        dst_r = cv2.calcBackProject([frame], [2], hist_r, [0, 256], 1)
        
        # 合并三个通道
        dst = cv2.addWeighted(dst_b, 0.33, dst_g, 0.33, 0)
        dst = cv2.addWeighted(dst, 1.0, dst_r, 0.33, 0)
    
    return dst


def visualize_hue_and_backprojection(frame, hist, track_window=None, save_dir=None, frame_num=None):
    """
    Q2可视化：显示Hue通道和反向投影
    
    Args:
        save_dir: 如果提供，会保存可视化结果
        frame_num: 帧编号
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue_channel = hsv[:, :, 0]
    
    # 计算反向投影
    backproj = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
    
    # 转换成可以画矩形的格式
    hue_img = cv2.cvtColor(hue_channel, cv2.COLOR_GRAY2BGR)
    backproj_img = cv2.cvtColor(backproj, cv2.COLOR_GRAY2BGR)
    
    # 画跟踪框
    if track_window is not None:
        r, c, w, h = track_window
        cv2.rectangle(hue_img, (r, c), (r + w, c + h), (0, 255, 0), 2)
        cv2.rectangle(backproj_img, (r, c), (r + w, c + h), (0, 255, 0), 2)
    
    # 显示
    cv2.imshow('Hue Channel', hue_img)
    cv2.imshow('Back Projection', backproj_img)
    
    # ✅ 保存可视化结果
    if save_dir is not None and frame_num is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(f"{save_dir}/Hue_{frame_num:04d}.png", hue_img)
        cv2.imwrite(f"{save_dir}/BackProj_{frame_num:04d}.png", backproj_img)
    
    return hue_img, backproj_img

def compute_gradients(frame, threshold=30):
    """
    Q3: 计算梯度方向和幅值
    
    Args:
        frame: 输入帧（BGR彩色图像）
        threshold: 梯度幅值阈值，低于此值的像素被mask
        
    Returns:
        orientations: 梯度方向 (弧度)，shape (H, W)
        magnitudes: 梯度幅值，shape (H, W)
        mask: 显著梯度的掩码，shape (H, W)，True表示梯度显著
    """
    # 转换为灰度图
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # 计算 x 和 y 方向的梯度 (使用 Sobel 算子)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度幅值
    magnitudes = np.sqrt(grad_x**2 + grad_y**2)
    
    # 计算梯度方向 (弧度)
    orientations = np.arctan2(grad_y, grad_x)
    
    # 使用阈值创建掩码：梯度幅值 > threshold 的像素
    mask = magnitudes > threshold
    
    return orientations, magnitudes, mask


def visualize_gradients(frame, orientations, magnitudes, mask, window_name='Gradient Orientation'):
    """
    Q3: 可视化梯度方向
    被mask的像素（梯度不显著）显示为红色
    
    Args:
        frame: 原始帧
        orientations: 梯度方向
        magnitudes: 梯度幅值
        mask: 显著梯度的掩码
        window_name: 窗口名称
    """
    # 创建可视化图像
    h, w = frame.shape[:2]
    
    # 方法1：将方向映射到颜色（HSV色彩空间）
    # H (色调) = 方向, S (饱和度) = 1, V (亮度) = 归一化的幅值
    orientation_normalized = (orientations + np.pi) / (2 * np.pi)  # 归一化到 [0, 1]
    orientation_hue = (orientation_normalized * 180).astype(np.uint8)  # 转换到 [0, 180] for OpenCV
    
    # 创建 HSV 图像
    hsv_img = np.zeros((h, w, 3), dtype=np.uint8)
    hsv_img[:, :, 0] = orientation_hue  # H: 方向
    hsv_img[:, :, 1] = 255  # S: 饱和度最大
    
    # V: 根据梯度幅值设置亮度
    magnitude_normalized = cv2.normalize(magnitudes, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hsv_img[:, :, 2] = magnitude_normalized
    
    # 转换为 BGR
    orientation_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    
    # ✅ 关键：被mask的像素（梯度不显著）显示为红色
    orientation_img[~mask] = [0, 0, 255]  # BGR: 红色
    
    cv2.imshow(window_name, orientation_img)
    
    return orientation_img


def visualize_gradient_magnitude(magnitudes, mask, window_name='Gradient Magnitude'):
    """
    Q3: 可视化梯度幅值
    """
    # 归一化到 [0, 255]
    mag_normalized = cv2.normalize(magnitudes, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 创建3通道图像用于显示
    mag_img = cv2.cvtColor(mag_normalized, cv2.COLOR_GRAY2BGR)
    
    # 被mask的像素显示为红色
    mag_img[~mask] = [0, 0, 255]
    
    cv2.imshow(window_name, mag_img)
    
    return mag_img