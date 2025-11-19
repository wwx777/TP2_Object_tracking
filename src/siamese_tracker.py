import os
from typing import Tuple
import cv2
import numpy as np
import torch
import sys
import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


SIAMFC_ROOT = os.path.join(PROJECT_ROOT, 'siamfc-pytorch')
if SIAMFC_ROOT not in sys.path:
    sys.path.append(SIAMFC_ROOT)


GOT10K_ROOT = os.path.join(PROJECT_ROOT,  'got10k-toolkit')
if GOT10K_ROOT not in sys.path:
    sys.path.append(GOT10K_ROOT)



PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIAMFC_ROOT = os.path.join(PROJECT_ROOT, 'siamfc-pytorch')
if SIAMFC_ROOT not in sys.path:
    sys.path.append(SIAMFC_ROOT)
from siamfc import TrackerSiamFC




class SiamFCTracker:
   
    
    def __init__(self, net_path=None, device='cpu', debug=False):
       
        self.device = device
        self.debug = debug
        
       
        print("Loading pretrained SiamFC model...")
        
        if net_path is None:
            
            self.tracker = TrackerSiamFC()
        else:
            self.tracker = TrackerSiamFC(net_path=net_path)
        
        print("✓ Pretrained Siamese network loaded!")
        
        self.current_box = None
        self.initialized = False
    
    def init(self, frame: np.ndarray, roi: Tuple[int, int, int, int]):
       
       
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        self.tracker.init(frame_rgb, roi)
        self.current_box = roi
        self.initialized = True
        
        if self.debug:
            print(f"[SiamFC] Initialized with bbox: {roi}")
    
    def update(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        if not self.initialized:
            raise RuntimeError("Tracker not initialized. Call init() first.")
        

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        

        bbox = self.tracker.update(frame_rgb)
        self.current_box = tuple(int(v) for v in bbox)
        
        if self.debug:
            print(f"[SiamFC] Updated bbox: {self.current_box}")
        
        return self.current_box


class SiameseTracker:
   
    
    def __init__(self, video_path: str, net_path=None, device: str = 'cpu', debug: bool = False):
       
        self.video_path = video_path
        self.device = device
        self.tracker = SiamFCTracker(net_path=net_path, device=device, debug=debug)
        self.initialized = False
    
    def select_roi(self, frame: np.ndarray):
        """交互式ROI选择"""
        print("\n=== ROI Selection ===")
        print("1. Drag to select the target")
        print("2. Press ENTER or SPACE to confirm")
        print("3. Press 'c' to cancel\n")
        
        roi = cv2.selectROI("Select Target - Real Siamese Network", frame, 
                           fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Target - Real Siamese Network")
        return roi
    
    def initialize(self, frame: np.ndarray, roi: Tuple[int, int, int, int]):
        """初始化跟踪器"""
        self.tracker.init(frame, roi)
        self.initialized = True
    
    def update(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """更新跟踪"""
        if not self.initialized:
            raise RuntimeError('Tracker not initialized. Call initialize() first.')
        return self.tracker.update(frame)
    
    def track_video(self, visualize=True, save_result=False, output_dir='results/siamfc_got10k'):
    
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        
        try:
            
            ret, frame = cap.read()
            if not ret:
                print('Cannot read video')
                return
            roi = self.select_roi(frame)
            if roi[2] == 0 or roi[3] == 0:
                print("Invalid ROI selected")
                return

            print("\nInitializing real Siamese network...")
            self.initialize(frame, roi)

            if save_result:
                from pathlib import Path
                import pandas as pd
                
                out_path = Path(output_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                print(f"Results will be saved to: {output_dir}")
            
            frame_idx = 1
            fps_list = []
            predictions = []
            
            print("\n" + "="*60)
            print("🎯 Tracking with REAL Siamese Network (GOT-10k trained)")
            print("="*60)
            print("Press ESC to quit | Press 'p' to pause")
            print("="*60 + "\n")
            
            x, y, w, h = roi
            out = frame.copy()
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(out, f'Real Siamese Network (GOT-10k)', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(out, f'Frame: {frame_idx}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if save_result:
                cv2.imwrite(str(out_path / f'Frame_{frame_idx:04d}.png'), out)
                predictions.append({'frame': frame_idx, 'x': x, 'y': y, 'w': w, 'h': h})
            
            if visualize:
                cv2.imshow('Real Siamese Network Tracker (GOT-10k)', out)
                cv2.waitKey(1)
            
            while True:
                start_time = cv2.getTickCount()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                
                bbox = self.update(frame)
                x, y, w, h = bbox
                
                out = frame.copy()
                
                cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 3)
                
                # 计算FPS
                end_time = cv2.getTickCount()
                fps = cv2.getTickFrequency() / (end_time - start_time)
                fps_list.append(fps)
                avg_fps = np.mean(fps_list[-30:])
                
                cv2.putText(out, f'Real Siamese Network (GOT-10k)', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(out, f'Frame: {frame_idx}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(out, f'FPS: {avg_fps:.1f}', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(out, f'BBox: ({x},{y},{w},{h})', (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if visualize:
                    cv2.imshow('Real Siamese Network Tracker (GOT-10k)', out)
                

                if save_result:
                    cv2.imwrite(str(out_path / f'Frame_{frame_idx:04d}.png'), out)
                    predictions.append({'frame': frame_idx, 'x': x, 'y': y, 'w': w, 'h': h})
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("\n\nTracking interrupted by user")
                    break
                elif key == ord('p'):  # Pause
                    print("\nPaused. Press any key to continue...")
                    cv2.waitKey(0)
                
                if frame_idx % 50 == 0:
                    print(f"📊 Frame {frame_idx} | Avg FPS: {avg_fps:.1f}")
            
            print("\n" + "="*60)
            print("✅ Tracking Completed!")
            print("="*60)
            frames_processed = max(0, frame_idx)
            print(f"Total frames processed: {frames_processed}")
            print(f"Average FPS: {np.mean(fps_list):.1f}")
            
            if save_result:
                df = pd.DataFrame(predictions)
                csv_path = out_path / "predictions.csv"
                df.to_csv(csv_path, index=False)
                print(f"Results saved to: {output_dir}")
                print(f"Saved {len(predictions)} frames and predictions.csv")
            print("="*60 + "\n")
        
        finally:
            try:
                cap.release()
            except:
                pass
            cv2.destroyAllWindows()
            try:
                cv2.waitKey(1)
            except:
                pass
def download_pretrained_model():
    print("\n" + "="*60)
    print("Downloading Pretrained Siamese Network Model")
    print("="*60 + "\n")
    
    try:
        from got10k.trackers import TrackerSiamFC
        
        print("Creating tracker (will auto-download if needed)...")
        tracker = TrackerSiamFC()
        
        print("\n✓ Model ready!")
        print("Model location: ~/.got10k/siamfc/")
        print("\nThis is a REAL Siamese network trained on:")
        print("  - GOT-10k dataset (10,000+ videos)")
        print("  - Using contrastive loss")
        print("  - Siamese architecture with shared weights")
        print("="*60 + "\n")
        
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease install got10k first:")
        print("  pip install got10k")
        return False


def verify_installation():
    print("\n" + "="*60)
    print("Verifying Installation")
    print("="*60 + "\n")
    
    try:
        from got10k.trackers import TrackerSiamFC
        print("✓ got10k installed")
    except ImportError:
        print("✗ got10k not found")
        print("  Install: pip install got10k")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("✗ PyTorch not found")
        print("  Install: pip install torch")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not found")
        print("  Install: pip install opencv-python")
        return False
    
    print("\n✓ All dependencies installed!")
    print("="*60 + "\n")
    return True


if __name__ == '__main__':
    import sys
    
    print("\n" + "="*60)
    print("🎯 Real Siamese Network Tracker (GOT-10k)")
    print("="*60)
    print("\nThis is a REAL Siamese network:")
    print("  ✓ Trained on GOT-10k tracking dataset")
    print("  ✓ Uses contrastive loss for similarity learning")
    print("  ✓ Siamese architecture (shared weights)")
    print("  ✓ NOT just a pretrained ImageNet classifier!")
    print("="*60 + "\n")
    
    if not verify_installation():
        sys.exit(1)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python siamese_tracker_got10k.py <video_path>")
        print("\nExample:")
        print("  python siamese_tracker_got10k.py test_video.mp4")
        print("\nFirst time? Download pretrained model:")
        print("  python siamese_tracker_got10k.py --download")
        
        if '--download' in sys.argv:
            download_pretrained_model()
        
        sys.exit(1)
    
    video_path = sys.argv[1]

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    

    tracker = SiameseTracker(
        video_path=video_path,
        device=device,
        debug=True
    )

    tracker.track_video(visualize=True, save_result=True)