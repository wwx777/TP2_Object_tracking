"""
Classical tracking methods: Mean Shift and Hough Transform
"""
import cv2
import numpy as np
from .features import extract_color_histogram, compute_backprojection, visualize_hue_and_backprojection
from .utils import ROISelector, visualize_tracking, save_frame


class ClassicalTracker:
    """
    经典跟踪方法基类
    支持 Mean Shift 和 Hough Transform
    """
    def __init__(self, video_path, method='meanshift', **kwargs):
        """
        Args:
            video_path: 视频路径
            method: 'meanshift' 或 'hough'
            **kwargs: 其他配置参数
        """
        self.video_path = video_path
        self.method = method
        self.config = kwargs
        
        # 跟踪状态
        self.roi = None
        self.track_window = None
        self.model = None  # 直方图或R-Table
        self.velocity = np.array([0.0, 0.0])
        # Mean Shift 配置
        self.color_space = kwargs.get('color_space', 'hue')
        self.update_model = kwargs.get('update_model', False)
        self.update_rate = kwargs.get('update_rate', 0.05)
        
        # Mean Shift 终止条件
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                         10, 1)
        # Hough Transform 配置
        self.gradient_threshold = kwargs.get('gradient_threshold', 30)
    
    def select_roi(self, frame):
        """选择ROI"""
        selector = ROISelector()
        roi = selector.select_roi(frame)
        self.roi = roi
        return roi
    
    def initialize(self, frame, roi):
        """
        初始化跟踪器
        
        Args:
            frame: 第一帧
            roi: (r, c, w, h) - ROI坐标
        """
        self.track_window = roi
        
        if self.method == 'meanshift':
            self._init_meanshift(frame, roi)
        elif self.method == 'hough':
            raise NotImplementedError("Hough method will be implemented in Q3-Q5")
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _init_meanshift(self, frame, roi):
        """
        Q1: Mean Shift 初始化
        
        Args:
            frame: 第一帧
            roi: (r, c, w, h)
        """
        r, c, w, h = roi
        roi_region = frame[c:c+w, r:r+h]
        
        # 提取颜色直方图作为模型
        self.model = extract_color_histogram(
            roi_region, 
            feature_type=self.color_space
        )        
        print(f"Mean Shift initialized with {self.color_space} histogram")
    
    def update(self, frame):
        """
        更新跟踪
        
        Args:
            frame: 当前帧
            
        Returns:
            new_window: 新的跟踪窗口 (r, c, w, h)
        """
        if self.method == 'meanshift':
            return self._update_meanshift(frame)
        elif self.method == 'hough':
            raise NotImplementedError("Hough method will be implemented in Q3-Q5")
    
    def _update_meanshift(self, frame):
        """
        Q1: Mean Shift 更新
        
        Args:
            frame: 当前帧
            
        Returns:
            new_window: 新的跟踪窗口
        """
        # 计算反向投影（权重图）
        dst = compute_backprojection(
            frame, 
            self.model, 
            self.color_space
        )
        
        # 应用 Mean Shift 算法
        ret, new_window = cv2.meanShift(dst, self.track_window, self.term_crit)
        
        # Q2改进: 更新模型
        if self.update_model:
            self._update_model_meanshift(frame, new_window)
        
        return new_window
    
    def _update_model_meanshift(self, frame, window):
        """
        Q2改进: 更新直方图模型
        
        Args:
            frame: 当前帧
            window: 当前跟踪窗口
        """
        r, c, w, h = window
        roi_region = frame[c:c+w, r:r+h]
        
        # 提取当前ROI的直方图
        current_hist = extract_color_histogram(
            roi_region, 
            feature_type=self.color_space
        )
        
        # 加权更新: new_model = (1-α) * old_model + α * current_hist
        alpha = self.update_rate
        self.model = cv2.addWeighted(self.model, 1-alpha, 
                                     current_hist, alpha, 0)
    
    
    def _init_hough(self, frame, roi):
        """
        Q3-Q4: Hough Transform 初始化
        """
        raise NotImplementedError("Q4: Will build R-Table here")

# 添加 Hough 更新方法（Q4会用到）
    def _update_hough(self, frame):
        """
        Q3-Q4: Hough Transform 更新
        """
        raise NotImplementedError("Q4: Will compute Hough transform here")

    def track_video(self, visualize=True, save_result=False, 
               visualize_process=False, output_dir='results/frames'):
        """在整个视频上跟踪"""
        cap = cv2.VideoCapture(self.video_path)
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read video")
            return
        
        # 1. 选择ROI
        print("Step 1: Select ROI")
        roi = self.select_roi(frame)
        
        # 2. 初始化
        print("Step 2: Initialize tracker")
        self.initialize(frame, roi)
        
        # 3. 跟踪
        print("Step 3: Start tracking")
        print("Press 's' to save frame, 'ESC' to exit")
        
        frame_count = 1
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 更新跟踪
            new_window = self.update(frame)
            self.track_window = new_window
            
            # 可视化
            if visualize:
                frame_with_box = visualize_tracking(
                    frame, 
                    new_window, 
                    window_name='Tracking Result',
                    color=(255, 0, 0),
                    thickness=2
                )
                
                # Q2: 可视化中间过程
                if visualize_process and self.method == 'meanshift':
                    visualize_hue_and_backprojection(
                        frame, 
                        self.model, 
                        new_window,
                        save_dir=output_dir if save_result else None,  # ✅ 传递保存路径
                        frame_num=frame_count  # ✅ 传递帧编号
                    )
            
            # 保存
            if save_result:
                save_frame(frame_with_box, frame_count, output_dir)
            
            # ✅ 关键：在主循环处理按键
            key = cv2.waitKey(60) & 0xFF
            if key == 27:  # ESC
                print("\nTracking stopped by user")
                break
            elif key == ord('s'):
                save_frame(frame_with_box, frame_count, output_dir)
                print(f"Saved frame {frame_count}")
            
            frame_count += 1
        
        # ✅ 清理资源
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
        print(f"\n✅ Tracking completed. Total frames: {frame_count}")