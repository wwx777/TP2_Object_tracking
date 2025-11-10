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


def visualize_hue_and_backprojection(frame, hist, track_window=None):
    """
    Q2可视化：显示Hue图像和反向投影
    
    Args:
        frame: 当前帧
        hist: 目标直方图
        track_window: 可选的跟踪窗口
        
    Returns:
        hue_img: Hue通道图像
        backproj: 反向投影图像
    """
    # 转换到HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue_img = hsv[:, :, 0]
    
    # 计算反向投影
    backproj = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
    
    # 如果有跟踪窗口，在图像上标注
    if track_window is not None:
        r, c, w, h = track_window
        cv2.rectangle(hue_img, (r, c), (r + h, c + w), 255, 2)
        cv2.rectangle(backproj, (r, c), (r + h, c + w), 255, 2)
    
    # 显示
    cv2.imshow('Hue Channel', hue_img)
    cv2.imshow('Back Projection', backproj)
    
    return hue_img, backproj