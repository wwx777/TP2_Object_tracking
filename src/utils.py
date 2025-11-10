"""
Utility functions for tracking
"""
import cv2
import numpy as np
import os

"""
Utility functions for tracking
"""
import cv2
import numpy as np
import os


class ROISelector:
    """交互式ROI选择器"""
    
    def __init__(self):
        self.roi_defined = False
        self.r, self.c, self.w, self.h = 0, 0, 0, 0
        
    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 按下左键：记录起始点
            self.r, self.c = x, y
            self.roi_defined = False
            
        elif event == cv2.EVENT_LBUTTONUP:
            # 松开左键：记录终点并计算ROI
            r2, c2 = x, y
            self.h = abs(r2 - self.r)
            self.w = abs(c2 - self.c)
            self.r = min(self.r, r2)
            self.c = min(self.c, c2)
            self.roi_defined = True
    
    def select_roi(self, frame, window_name="First image"):
        """
        交互式选择ROI
        
        Args:
            frame: 输入图像
            window_name: 窗口名称
            
        Returns:
            roi: (r, c, w, h) - 左上角坐标和宽高
        """
        clone = frame.copy()
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        print("Instructions:")
        print("- Drag mouse to select ROI")
        print("- Press 'q' to confirm and start tracking")
        
        while True:
            # 每次循环都重新从 clone 复制，这样可以清除之前的矩形
            display_frame = clone.copy()
            
            # 如果ROI已定义，绘制绿色矩形
            if self.roi_defined:
                cv2.rectangle(display_frame, 
                            (self.r, self.c), 
                            (self.r + self.h, self.c + self.w), 
                            (0, 255, 0), 2)
            
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            # 按 'q' 键确认
            if key == ord('q'):
                break
        
        cv2.destroyWindow(window_name)
        return (self.r, self.c, self.w, self.h)


def visualize_tracking(frame, track_window, window_name='Tracking', 
                       color=(255, 0, 0), thickness=2):
    """在帧上绘制跟踪框"""
    r, c, w, h = track_window
    frame_with_box = frame.copy()
    cv2.rectangle(frame_with_box, (r, c), (r + h, c + w), color, thickness)
    cv2.imshow(window_name, frame_with_box)
    return frame_with_box


def save_frame(frame, frame_number, output_dir='results/frames'):
    """保存帧到文件"""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/Frame_{frame_number:04d}.png"
    cv2.imwrite(filename, frame)
    return filename
