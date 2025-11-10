"""
Utility functions for tracking
"""
import cv2
import os


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
            self.h = abs(r2 - self.r)
            self.w = abs(c2 - self.c)
            self.r = min(self.r, r2)
            self.c = min(self.c, c2)
            self.roi_defined = True
    
    def select_roi(self, frame, window_name="First image"):
        """选择ROI"""
        clone = frame.copy()
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        while True:
            frame_disp = clone.copy()
            if self.roi_defined:
                cv2.rectangle(frame_disp, (self.r, self.c),
                            (self.r + self.w, self.c + self.h), (0, 255, 0), 2)

            cv2.imshow(window_name, frame_disp)
            key = cv2.waitKey(30) & 0xFF
        
            # 检查窗口是否关闭
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        
            # ESC 或 q 退出
            if key in (27, ord('q')):
                break

        cv2.destroyWindow(window_name)
        return (self.r, self.c, self.w, self.h)


def visualize_tracking(frame, track_window, window_name='Tracking', 
                       color=(255, 0, 0), thickness=2):
    """
    在帧上绘制跟踪框
    注意：不要在这里处理按键，让主循环处理
    """
    r, c, w, h = track_window
    frame_with_box = frame.copy()
    cv2.rectangle(frame_with_box, (r, c), (r + w, c + h), color, thickness)
    cv2.imshow(window_name, frame_with_box)
    return frame_with_box  # ✅ 只显示，不处理按键


def save_frame(frame, frame_number, output_dir='results/frames'):
    """保存帧到文件"""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/Frame_{frame_number:04d}.png"
    cv2.imwrite(filename, frame)
    return filename