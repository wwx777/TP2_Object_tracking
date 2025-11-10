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
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            # if the ROI is defined, draw it!
            if (self.roi_defined):
                # draw a green rectangle around the region of interest
                cv2.rectangle(frame, (self.r, self.c), (self.r+self.h, self.c+self.w), (0, 255, 0), 2)
            # else reset the image...
            else:
                frame = clone.copy()
            # if the 'q' key is pressed, break from the loop
            if key == ord("q"):
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
    cv2.waitKey(1)
    return frame_with_box


def save_frame(frame, frame_number, output_dir='results/frames'):
    """保存帧到文件"""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/Frame_{frame_number:04d}.png"
    cv2.imwrite(filename, frame)
    return filename