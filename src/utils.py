"""
Utility functions for tracking
"""
import cv2
import os
import numpy as np



class ROISelector:
    """交互式ROI选择器"""
    
    def __init__(self):
        self.roi_defined = False
        self.r, self.c, self.w, self.h = 0, 0, 0, 0
        
    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.r, self.c = x, y
            self.roi_defined = False
            
        elif event == cv2.EVENT_LBUTTONUP:
            r2, c2 = x, y
            # 修复：r是x(列)，c是y(行)
            # w = 宽度(x方向), h = 高度(y方向)
            self.w = abs(r2 - self.r)
            self.h = abs(c2 - self.c)
            self.r = min(self.r, r2)
            self.c = min(self.c, c2)
            self.roi_defined = True
    
    def select_roi(self, frame, window_name="First image"):
        """Select ROI interactively"""
        print(f"DEBUG: Creating window '{window_name}'")
        print(f"DEBUG: Frame shape: {frame.shape}")
        clone = frame.copy()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        print("DEBUG: Window created, showing frame...")

        while True:
            frame_disp = clone.copy()
            if self.roi_defined:
                cv2.rectangle(frame_disp, (self.r, self.c),
                            (self.r + self.w, self.c + self.h), (0, 255, 0), 2)
                cv2.putText(frame_disp, 'Press SPACE/ENTER to confirm, c to reselect', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame_disp, 'Click and drag to select ROI', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(window_name, frame_disp)
            cv2.waitKey(1)  # Force window update
            key = cv2.waitKey(30) & 0xFF
        
            # Check if window is closed
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except:
                pass  # Window might not be visible yet
        
            # SPACE or ENTER to confirm selection
            if key in (32, 13) and self.roi_defined:  # SPACE=32, ENTER=13
                break
            
            # 'c' to cancel and reselect
            if key == ord('c'):
                self.roi_defined = False
            
            # ESC or 'q' to exit
            if key in (27, ord('q')):
                break

        cv2.destroyWindow(window_name)
        cv2.waitKey(1)
        return (self.r, self.c, self.w, self.h)


def visualize_tracking(frame, track_window, window_name='Tracking', 
                       color=(255, 0, 0), thickness=2):
    """
    在帧上绘制跟踪框
    注意：不要在这里处理按键，让主循环处理
    
    Args:
        window_name: If None, don't show window (for save-only mode)
    """
    r, c, w, h = track_window
    frame_with_box = frame.copy()
    cv2.rectangle(frame_with_box, (r, c), (r + w, c + h), color, thickness)
    
    if window_name is not None:
        cv2.imshow(window_name, frame_with_box)
    
    return frame_with_box  # ✅ 只显示，不处理按键


def save_frame(frame, frame_number, output_dir='results/frames'):
    """保存帧到文件"""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/Frame_{frame_number:04d}.png"
    cv2.imwrite(filename, frame)
    return filename