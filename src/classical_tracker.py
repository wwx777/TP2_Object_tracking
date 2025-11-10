
import cv2
import numpy as np
from dataclasses import dataclass, field


@dataclass
class TrackState:
    # Python 3.10+ 可以用 | 语法；若你要兼容 3.9，用 Optional[Tuple[int,int,int,int]]
    track_window: tuple | None = None          # (r, c, w, h)
    model: np.ndarray | None = None            # 直方图 / R-Table
    # 关键修复：用 default_factory 创建默认的 numpy 数组
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))

    # 缓存梯度给 Hough / 可视化复用
    orientations: np.ndarray | None = None
    magnitudes:  np.ndarray | None = None
    grad_mask:   np.ndarray | None = None
    
    # Q4: Hough Transform 累加器可视化
    hough_accumulator: np.ndarray | None = None  # 累加器 H(x)
    search_region: tuple | None = None           # (r1, c1, r2, c2)



class TrackerStrategy:
    def init(self, state: TrackState, frame, roi): ...
    def update(self, state: TrackState, frame) -> tuple: ...  # return new window


class MeanShiftStrategy(TrackerStrategy):
    def __init__(self, *, color_space='hue', update_model=False, update_rate=0.05,
                 term_crit=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1),
                 extract_color_histogram=None, compute_backprojection=None):
        self.color_space = color_space
        self.update_model = update_model
        self.update_rate  = update_rate
        self.term_crit    = term_crit
        self.extract_color_histogram = extract_color_histogram
        self.compute_backprojection  = compute_backprojection

    def init(self, state: TrackState, frame, roi):
        r, c, w, h = roi
        roi_region = frame[c:c+h, r:r+w]
        state.model = self.extract_color_histogram(roi_region, feature_type=self.color_space)
        state.track_window = roi

    def update(self, state: TrackState, frame):
        dst = self.compute_backprojection(frame, state.model, self.color_space)
        _, new_window = cv2.meanShift(dst, state.track_window, self.term_crit)
        if self.update_model:
            r, c, w, h = new_window
            roi_region = frame[c:c+h, r:r+w]
            current_hist = self.extract_color_histogram(roi_region, feature_type=self.color_space)
            alpha = self.update_rate
            state.model = cv2.addWeighted(state.model, 1 - alpha, current_hist, alpha, 0)
        state.track_window = new_window
        return new_window
    

class HoughTransformStrategy(TrackerStrategy):
    """
    手写 Generalized Hough（无尺度/无旋转）
    依赖: compute_gradients(frame, threshold) -> (orientations, magnitudes, mask)
    用法: 在 _build_strategy 分支中返回本类实例即可。
    """
    def __init__(self, *, compute_gradients,
             gradient_threshold=30, angle_bins=36,
             gaussian_blur_ksize=5, search_window_expand=1.2,
             vote_weight='magnitude',
             show_orientations=False,
             save_orientations=False,
             orient_out_dir='results/q3_orient'):
        self.compute_gradients = compute_gradients
        self.gradient_threshold = gradient_threshold
        self.angle_bins = angle_bins
        self.gaussian_blur_ksize = gaussian_blur_ksize
        self.search_window_expand = search_window_expand
        self.vote_weight = vote_weight
        self.show_orientations = show_orientations
        self.save_orientations = save_orientations
        self.orient_out_dir = orient_out_dir

        self.rtable = None
        self.win_w = None
        self.win_h = None
        self._frame_idx = 0

    # ---------- helpers ----------
    def _angle_to_bin(self, theta):
        th = np.mod(theta, np.pi)  # [0, π)
        b = np.floor(self.angle_bins * th / (np.pi * (1 - 1e-9))).astype(int)
        return np.clip(b, 0, self.angle_bins - 1)

    def _smooth_argmax(self, acc):
        if self.gaussian_blur_ksize and self.gaussian_blur_ksize > 1:
            acc = cv2.GaussianBlur(acc, (self.gaussian_blur_ksize, self.gaussian_blur_ksize), 0)
        idx = int(np.argmax(acc))
        y, x = np.unravel_index(idx, acc.shape)
        return x, y
    def _make_orientation_vis(self, frame):
        import os
        ori, mag, msk = self.compute_gradients(frame, threshold=self.gradient_threshold)
        H = ((ori + np.pi) * 90.0 / np.pi).astype(np.uint8)
        S = np.full_like(H, 255, dtype=np.uint8)
        V = np.full_like(H, 255, dtype=np.uint8)
        hsv = cv2.merge([H, S, V])
        vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        vis[~msk] = (0, 0, 255)
        if self.show_orientations:
            cv2.imshow('Q3: Orientation (masked red)', vis)
        if self.save_orientations:
            os.makedirs(self.orient_out_dir, exist_ok=True)
            cv2.imwrite(f'{self.orient_out_dir}/orient_{self._frame_idx:04d}.png', vis)
        return vis


    def _build_rtable(self, frame, roi):
        """
        在 ROI 裁剪上构建 R-Table（全部用 ROI 局部坐标）
        向量 r = [x_edge - cx, y_edge - cy]，其中 cx=w/2, cy=h/2
        """
        r, c, w, h = roi
        roi_img = frame[c:c+h, r:r+w]
        ori, mag, msk = self.compute_gradients(roi_img, threshold=self.gradient_threshold)

        ys, xs = np.where(msk)
        if xs.size == 0:
            # 空表也要占位，保持索引一致
            return [np.empty((0, 2), np.float32) for _ in range(self.angle_bins)]

        cx, cy = w / 2.0, h / 2.0  # ROI 局部中心
        bins = self._angle_to_bin(ori[ys, xs])
        rt = [[] for _ in range(self.angle_bins)]
        for x, y, b in zip(xs, ys, bins):
            rt[b].append([x - cx, y - cy])

        # 压成 ndarray
        rt = [np.asarray(v, dtype=np.float32) if len(v) else np.empty((0, 2), np.float32)
              for v in rt]
        return rt

    def _search_region(self, window, W, H):
        if not window or not self.search_window_expand or self.search_window_expand <= 1.0:
            return 0, 0, W, H
        r0, c0, w0, h0 = window
        cx, cy = r0 + w0 / 2.0, c0 + h0 / 2.0
        sw, sh = int(w0 * self.search_window_expand), int(h0 * self.search_window_expand)
        r1 = max(0, int(cx - sw / 2)); c1 = max(0, int(cy - sh / 2))
        r2 = min(W, r1 + sw);          c2 = min(H, c1 + sh)
        return r1, c1, r2, c2

    # ---------- strategy interface ----------
    def init(self, state: TrackState, frame, roi):
        r, c, w, h = roi
        self.win_w, self.win_h = w, h
        self.rtable = self._build_rtable(frame, roi)
        state.track_window = roi

    def update(self, state: TrackState, frame):
        H, W = frame.shape[:2]
        if self.rtable is None:
            return state.track_window
        self._frame_idx += 1
        # 1) 搜索区
        r1, c1, r2, c2 = self._search_region(state.track_window, W, H)
        search = frame[c1:c2, r1:r2]
        if search.size == 0:
            return state.track_window

        # 2) 搜索区梯度
        ori, mag, msk = self.compute_gradients(search, threshold=self.gradient_threshold)
        ys, xs = np.where(msk)
        if xs.size == 0:
            return state.track_window

        # 3) 累加器投票（在搜索区局部坐标）
        acc = np.zeros((search.shape[0], search.shape[1]), dtype=np.float32)
        bins = self._angle_to_bin(ori[ys, xs])
        weights = mag[ys, xs].astype(np.float32) if self.vote_weight == 'magnitude' else np.ones_like(xs, dtype=np.float32)

        for b in range(self.angle_bins):
            sel = (bins == b)
            if not np.any(sel):
                continue
            rvecs = self.rtable[b]
            if rvecs.size == 0:
                continue
            pts = np.stack([xs[sel], ys[sel]], axis=1).astype(np.int32)  # [M,2]
            wv  = weights[sel]
            # 广播得到中心候选 [M,K,2]: center = edge - rvec
            cands = pts[:, None, :] - rvecs[None, :, :]                  # [M,K,2]
            cx = np.rint(cands[..., 0]).astype(np.int32).reshape(-1)
            cy = np.rint(cands[..., 1]).astype(np.int32).reshape(-1)
            ww = np.repeat(wv, rvecs.shape[0])
            ok = (cx >= 0) & (cx < acc.shape[1]) & (cy >= 0) & (cy < acc.shape[0])
            np.add.at(acc, (cy[ok], cx[ok]), ww[ok])

        # 4) 取峰
        cx_local, cy_local = self._smooth_argmax(acc)
        cx_abs, cy_abs = cx_local + r1, cy_local + c1

        # 5) 映射为窗口
        r_new = int(round(cx_abs - self.win_w / 2.0))
        c_new = int(round(cy_abs - self.win_h / 2.0))
        r_new = max(0, min(W - self.win_w, r_new))
        c_new = max(0, min(H - self.win_h, c_new))

        new_window = (r_new, c_new, self.win_w, self.win_h)
        state.track_window = new_window
        
        # Q4: 保存累加器和搜索区域用于可视化
        state.hough_accumulator = acc.copy()
        state.search_region = (r1, c1, r2, c2)
        
        if self.show_orientations or self.save_orientations:
            self._make_orientation_vis(frame)
        return new_window
 
class GradientSidecar:
    def __init__(self, *, enabled=False, threshold=30, save=False,
                 out_dir='results/q3_gradients',
                 compute_gradients=None, visualize_gradients=None, visualize_gradient_magnitude=None):
        self.enabled  = enabled
        self.threshold = threshold
        self.save = save
        self.out_dir = out_dir
        self.compute_gradients = compute_gradients
        self.visualize_gradients = visualize_gradients
        self.visualize_gradient_magnitude = visualize_gradient_magnitude

    def __call__(self, state: TrackState, frame, idx: int):
        if not self.enabled:
            return
        import os
        orientations, magnitudes, mask = self.compute_gradients(frame, threshold=self.threshold)
        state.orientations = orientations
        state.magnitudes   = magnitudes
        state.grad_mask    = mask

        ori_img = self.visualize_gradients(frame, orientations, magnitudes, mask, 'Q3: Gradient Orientation')
        mag_img = self.visualize_gradient_magnitude(magnitudes, mask, 'Q3: Gradient Magnitude')

        if self.save:
            os.makedirs(self.out_dir, exist_ok=True)
            cv2.imwrite(f'{self.out_dir}/Orientation_{idx:04d}.png', ori_img)
            cv2.imwrite(f'{self.out_dir}/Magnitude_{idx:04d}.png',  mag_img)


class ClassicalTracker:
    __slots__ = ('video_path', 'method', 'state', 'strategy', 'grad_sidecar',
                 'visualize', 'save_result', 'visualize_process', 'output_dir',
                 'visualize_tracking', 'visualize_hue_and_backprojection', 'save_frame')

    def __init__(self, video_path, method='meanshift', **kwargs):
        self.video_path = video_path
        self.method = method
        self.state = TrackState()


        from .features import extract_color_histogram, compute_backprojection, visualize_hue_and_backprojection
        from .utils import visualize_tracking, save_frame
        from .features import compute_gradients, visualize_gradients, visualize_gradient_magnitude

        self.strategy = self._build_strategy(method, kwargs,
                                             extract_color_histogram, compute_backprojection)

        self.grad_sidecar = GradientSidecar(
            enabled=kwargs.get('enable_gradients_vis', False),
            threshold=kwargs.get('gradient_threshold', 30),
            save=kwargs.get('save_gradients', False),
            out_dir=kwargs.get('grad_output_dir', 'results/q3_gradients'),
            compute_gradients=compute_gradients,
            visualize_gradients=visualize_gradients,
            visualize_gradient_magnitude=visualize_gradient_magnitude
        )


        self.visualize_tracking = visualize_tracking
        self.visualize_hue_and_backprojection = visualize_hue_and_backprojection
        self.save_frame = save_frame


        self.visualize = kwargs.get('visualize', True)
        self.save_result = kwargs.get('save_result', False)
        self.visualize_process = kwargs.get('visualize_process', False)
        self.output_dir = kwargs.get('output_dir', 'results/frames')

    def _build_strategy(self, method, kwargs, extract_color_histogram, compute_backprojection):
        if method == 'meanshift':
            return MeanShiftStrategy(
                color_space=kwargs.get('color_space', 'hsv'),
                update_model=kwargs.get('update_model', False),
                update_rate=kwargs.get('update_rate', 0.05),
                term_crit=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1),
                extract_color_histogram=extract_color_histogram,
                compute_backprojection=compute_backprojection
            )
        elif method == 'hough':
            from .features import compute_gradients
            return HoughTransformStrategy(
                compute_gradients=compute_gradients,
                gradient_threshold=kwargs.get('gradient_threshold', 30),
                angle_bins=kwargs.get('angle_bins', 36),
                gaussian_blur_ksize=kwargs.get('gaussian_blur_ksize', 5),
                search_window_expand=kwargs.get('search_window_expand', 1.2),
                vote_weight=kwargs.get('vote_weight', 'magnitude'),
                show_orientations=kwargs.get('show_orientations', False),
                save_orientations=kwargs.get('save_orientations', False),
                orient_out_dir=kwargs.get('orient_out_dir', 'results/q3_orient')
            )


            
        else:
            raise ValueError(f"Unknown method: {method}")

    def select_roi(self, frame):
        from .utils import ROISelector
        return ROISelector().select_roi(frame)


    def initialize(self, frame, roi):
        self.strategy.init(self.state, frame, roi)

    def update(self, frame):
        return self.strategy.update(self.state, frame)

    def track_video(self, visualize=True, save_result=False, visualize_process=False, output_dir='results/frames'):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read video")
            return

        print("Step 1: Select ROI")
        roi = self.select_roi(frame)

        print("Step 2: Initialize tracker")
        self.initialize(frame, roi)

        print("Step 3: Start tracking")
        print("Press 's' to save frame, 'ESC' to exit")

        frame_count = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            new_window = self.update(frame)

            # 梯度側掛：一行接入，可隨時關
            self.grad_sidecar(self.state, frame, frame_count)

            if visualize:
                frame_with_box = self.visualize_tracking(
                    frame, new_window, window_name='Tracking Result',
                    color=(255, 0, 0), thickness=2
                )
                
                # 根据不同方法使用不同的可视化
                if visualize_process:
                    if isinstance(self.strategy, MeanShiftStrategy):
                        # Mean-shift: 显示Hue和反向投影
                        self.visualize_hue_and_backprojection(
                            frame, self.state.model, new_window,
                            save_dir=output_dir if save_result else None,
                            frame_num=frame_count
                        )
                    elif isinstance(self.strategy, HoughTransformStrategy):
                        # Q3: 梯度四宫格可视化
                        from .features import render_gradient_quadrants, compute_gradients
                        orientations, magnitudes, mask = compute_gradients(frame, threshold=self.strategy.gradient_threshold)
                        
                        gradient_save_path = None
                        if save_result:
                            import os
                            os.makedirs(output_dir, exist_ok=True)
                            gradient_save_path = f"{output_dir}/gradient_quadrants_{frame_count:04d}.png"
                        
                        # 渲染并显示四宫格
                        panel = render_gradient_quadrants(frame, orientations, magnitudes, mask, gradient_save_path)
                        cv2.imshow('Q3: Gradient Analysis', panel)
                        
                        # Q4: Hough Transform 累加器可视化
                        from .features import visualize_hough_transform
                        if self.state.hough_accumulator is not None and self.state.search_region is not None:
                            hough_save_path = None
                            if save_result:
                                hough_save_path = f"{output_dir}/hough_transform_{frame_count:04d}.png"
                            
                            hough_vis = visualize_hough_transform(
                                frame, self.state.hough_accumulator, 
                                self.state.search_region, new_window,
                                hough_save_path
                            )
                            cv2.imshow('Q4: Hough Transform H(x)', hough_vis)

            if save_result:
                self.save_frame(frame_with_box, frame_count, output_dir)

            key = cv2.waitKey(60) & 0xFF
            if key == 27:
                print("\nTracking stopped by user")
                break
            elif key == ord('s'):
                self.save_frame(frame_with_box, frame_count, output_dir)
                print(f"Saved frame {frame_count}")

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print(f"\n✅ Tracking completed. Total frames: {frame_count}")
