# 🎯 Object Tracking Project

Implementation and comparison of classical and deep learning object tracking algorithms.

---

## 📂 Project Structure

```
```
object_tracking/
├── src/                          # Core source code
│   ├── classical_tracker.py     # Classical tracker (Strategy Pattern)
│   ├── features.py               # Feature extraction (color/gradient)
│   ├── utils.py                  # Utility functions (ROI selection, etc.)
│   ├── tracking_mean_shift.py   # Legacy mean-shift implementation
│   └── deep_tracker.py           # Deep learning tracker (TODO)
│
├── test/                         # Tests and demos
│   └── basic_questions.ipynb    # Q1-Q4 demonstrations
│
├── Test-Videos/                  # Test videos
│   ├── Antoine_Mug.mp4
│   └── VOT-ball.mp4
│
├── results/                      # Output results (gitignored)
│   ├── q1_basic/                # Q1 basic Mean-shift
│   ├── q2_*/                    # Q2 improvements
│   ├── q3_gradients/            # Q3 gradient visualization
│   └── q4_hough_transform/      # Q4 Hough Transform
│
└── docs/                         # Documentation (optional)
```

---

## 🏗️ Architecture Design

### Core Design Pattern: **Strategy Pattern**
```

---

## �️ 架构设计

### 核心设计模式：**策略模式 (Strategy Pattern)**

```python
# Architecture Overview
ClassicalTracker (Main Controller)
    ├── TrackerStrategy (Abstract Strategy Interface)
    │   ├── MeanShiftStrategy      # Mean-shift implementation
    │   └── HoughTransformStrategy # Hough Transform implementation
    │
    ├── TrackState (State Management)
    │   ├── track_window           # Current tracking window
    │   ├── model                  # Histogram / R-Table
    │   ├── hough_accumulator      # Hough accumulator (Q4)
    │   └── search_region          # Search region
    │
    └── GradientSidecar (Gradient Visualization Plugin)
```

### Why Strategy Pattern?

✅ **Easy to Extend**: Add new tracking methods by implementing `TrackerStrategy` interface  
✅ **Decoupled**: Tracking algorithms separated from main control logic  
✅ **Reusable**: State management and visualization shared across all strategies  

---

## 💻 Code Modules

### 1. `src/classical_tracker.py` - Main Controller

**Classes:**
- `TrackState`: Data class for tracking state management
  - `track_window`: (r, c, w, h) current tracking window
  - `model`: Color histogram or R-Table
  - `hough_accumulator`: Hough voting accumulator (Q4)
  - `search_region`: Search region bounds (Q4)

- `TrackerStrategy`: Abstract base class for tracking strategies
  - `init(state, frame, roi)`: Initialize tracker with first frame
  - `update(state, frame)`: Update tracking window for new frame

- `MeanShiftStrategy`: Mean-shift tracking implementation
  - Color histogram-based tracking
  - Supports single/dual channel histograms
  - Optional adaptive model update

- `HoughTransformStrategy`: Generalized Hough Transform tracking
  - Gradient-based R-Table model
  - Voting-based center localization
  - Gaussian smoothing for peak detection

- `GradientSidecar`: Optional gradient visualization plugin
  - Computes and displays gradients per frame
  - Decoupled from tracking logic

- `ClassicalTracker`: Main tracker class
  - Strategy selection and initialization
  - Video processing loop
  - Visualization management

### 2. `src/features.py` - Feature Extraction

**Color Features:**
- `extract_color_histogram(roi, feature_type='hue', mask=None)`
  - Extract Hue/HSV/RGB histograms
  - Automatic mask generation for HSV
  - Normalized output

- `compute_backprojection(frame, hist, feature_type='hue')`
  - Calculate back-projection from histogram
  - Supports Hue/HSV/RGB

- `visualize_hue_and_backprojection(frame, hist, track_window, save_dir, frame_num)`
  - Visualize Hue channel and back-projection
  - Draw tracking box
  - Optional save to file

**Gradient Features (Q3-Q4):**
- `compute_gradients(frame, threshold=30)`
  - Compute gradient orientation and magnitude using Sobel
  - Returns: (orientations, magnitudes, mask)
  - Mask filters low-magnitude gradients

- `visualize_gradients(frame, orientations, magnitudes, mask, window_name)`
  - Visualize gradient orientations as HSV image
  - Masked pixels shown in red

- `visualize_gradient_magnitude(magnitudes, mask, window_name)`
  - Visualize gradient magnitude
  - Masked pixels shown in red

- `render_gradient_quadrants(frame, orientations, magnitudes, mask, save_path)`
  - Create 2x2 panel visualization:
    - Original frame
    - Gradient orientation (grayscale)
    - Gradient magnitude (bone colormap)
    - Selected orientations (HSV with mask in red)

- `visualize_hough_transform(frame, accumulator, search_region, detected_window, save_path)`
  - Visualize Hough Transform accumulator (Q4)
  - JET colormap heatmap overlay
  - Show search region (green box)
  - Show detected window (red box)
  - Mark peak location (white cross)

### 3. `src/utils.py` - Utility Functions

**Classes:**
- `ROISelector`: Interactive ROI selection
  - Mouse callback for bounding box selection
  - Fixed x/y axis mapping issue
  - ESC/q to confirm selection

**Functions:**
- `visualize_tracking(frame, track_window, window_name, color, thickness)`
  - Draw tracking box on frame
  - Returns frame with box

- `save_frame(frame, frame_number, output_dir)`
  - Save frame to file with zero-padded numbering
  - Auto-create output directory

---

## 🎨 Usage

### Quick Start

```python
from src.classical_tracker import ClassicalTracker

# Q1: Basic Mean-shift
tracker = ClassicalTracker(
    video_path='Test-Videos/Antoine_Mug.mp4',
    method='meanshift',
    color_space='hue'
)
tracker.track_video(visualize=True, save_result=True)

# Q4: Hough Transform with visualization
tracker = ClassicalTracker(
    video_path='Test-Videos/VOT-ball.mp4',
    method='hough',
    gradient_threshold=30
)
tracker.track_video(
    visualize=True,
    visualize_process=True,  # Show gradients and accumulator
    save_result=True
)
```

### Full Demonstrations

See `test/basic_questions.ipynb` for complete implementations and comparisons of all questions.

---

## 📊 Implemented Methods

### Q1-Q2: Mean-shift Tracking

**Basic Implementation (Q1):**
- Hue histogram-based color tracking
- OpenCV `cv2.meanShift()` iterative optimization

**Improvements (Q2):**
1. **Dual-channel Histogram** (H+S): Better color discrimination
2. **Adaptive Model Update**: Adapt to lighting and pose changes
3. **Combined Approach**: Combine both improvements

**Visualizations:**
- Hue channel image
- Back-projection weight map

---

### Q3-Q4: Hough Transform Tracking

**Q3: Gradient Computation and Visualization**
- Sobel operator for gradient orientation and magnitude
- Gradient thresholding (threshold=30)
- **4-panel visualization:**
  - Original frame
  - Gradient orientation (grayscale)
  - Gradient magnitude (bone colormap)
  - Selected gradients (HSV, masked pixels in red)

**Q4: R-Table + Hough Transform**
- **R-Table Construction**: Build implicit model based on gradient orientations from initial ROI
- **Voting Process**: Each edge pixel votes for candidate centers based on gradient direction
- **Accumulator Visualization**: JET heatmap + search region + detection result

**Visualizations:**
- Q3: 4-panel gradient analysis
- Q4: Hough Transform accumulator heatmap

---

## 🎬 Visualization Effects

### Mean-shift
- **Tracking Result** - Basic tracking window
- **Hue Channel** - Hue channel grayscale image
- **Back Projection** - Back-projection weight map

### Hough Transform
- **Tracking Result** - Basic tracking window
- **Q3: Gradient Analysis** - 4-panel gradient visualization
- **Q4: Hough Transform H(x)** - Accumulator heatmap
  - Red heatmap: Voting intensity
  - Green box: Search region
  - Red box: Detection result
  - White cross: Accumulator peak location

---

## 💡 Key Improvements

### 1. **Strategy Pattern Decoupling**
Different tracking algorithms implement their own `init()` and `update()` methods, main controller only handles dispatching.

### 2. **State Management**
Use `TrackState` dataclass to uniformly manage tracking state, easy to extend and debug.

### 3. **Visualization Separation**
- Mean-shift uses Hue + Back Projection
- Hough uses 4-panel gradients + accumulator heatmap
- Controlled by `visualize_process` parameter

### 4. **Bug Fixes**
- Fixed x/y axis confusion in `ROISelector`
- Fixed coordinate mapping in Hough accumulator

---

## 🛠️ Environment Setup

### Requirements

```bash
# Python 3.10+ recommended
pip install opencv-python numpy matplotlib pillow
```

### Detailed Dependencies

- **opencv-python** (`cv2`): Core computer vision operations
  - Image processing
  - Mean-shift implementation
  - Video I/O
  
- **numpy**: Numerical computing
  - Array operations
  - Gradient computation
  
- **matplotlib**: Plotting and visualization
  - Result analysis in notebooks
  
- **pillow** (PIL): Image loading
  - Used in notebook visualizations

### Installation

```bash
# Clone repository
git clone https://github.com/wwx777/TP2_Object_tracking.git
cd TP2_Object_tracking

# Install dependencies
pip install -r requirements.txt  # If requirements.txt exists
# OR install manually:
pip install opencv-python numpy matplotlib pillow

# Run notebook
jupyter notebook test/basic_questions.ipynb
```

### Tested Environment

- Python: 3.10+
- OpenCV: 4.8.0+
- OS: macOS / Linux / Windows

---

## 📝 Examples

### Jupyter Notebook
```bash
cd test/
jupyter notebook basic_questions.ipynb
```

Run cells to see:
- Q1: Basic Mean-shift
- Q2: Three improvement approaches comparison
- Q3: Gradient computation and visualization
- Q4: Complete Hough Transform pipeline

### Python Script
```python
from src.classical_tracker import ClassicalTracker

# Q1
tracker = ClassicalTracker('Test-Videos/Antoine_Mug.mp4', method='meanshift')
tracker.track_video(visualize=True, save_result=True, output_dir='results/q1_basic')

# Q4
tracker = ClassicalTracker('Test-Videos/VOT-ball.mp4', method='hough')
tracker.track_video(visualize=True, visualize_process=True, save_result=True)
```

---

## 🚀 Advanced Features

### Q5: Predictive Tracking (✅ Implemented)

**Method**: `predictive_meanshift`

**Key Features**:
1. **Kalman Filter Prediction**: Predicts next frame position using state `[x, y, vx, vy]`
   - Exploits motion smoothness
   - Handles occlusion and fast motion
   - Reduces search space

2. **Adaptive Model Update**: Updates histogram model based on confidence
   - Confidence score: Bhattacharyya distance between current and model histograms
   - Update rate: `α = α_base × (1 - confidence)`
   - Only updates when confidence > threshold

3. **Confidence-based Search**: Expands search window when uncertain
   - High confidence: use predicted window
   - Low confidence: expand search by `search_expansion_factor`

**Usage**:
```python
tracker = ClassicalTracker(
    video_path='video.mp4',
    method='predictive_meanshift',
    color_space='hue',
    update_model=True,
    update_rate=0.05,
    confidence_threshold=0.6,
    search_expansion_factor=1.5
)
tracker.track_video(visualize=True)
```

**Advantages**:
- ✅ Robust to appearance changes
- ✅ Better handles fast motion
- ✅ Reduces drift over long sequences
- ✅ Adapts search strategy based on confidence

---

### Q6: Deep Learning (TODO)
- [ ] Implement `DeepTracker` class
- [ ] Integrate pre-trained CNN features
- [ ] Feature selection and dimensionality reduction

---

## 🤝 Contributing

To add a new tracking algorithm:

1. Create a new `Strategy` class in `classical_tracker.py`
2. Implement `init()` and `update()` methods
3. Register in `_build_strategy()`
4. (Optional) Add custom visualization functions

Example:
```python
class NewTrackerStrategy(TrackerStrategy):
    def init(self, state: TrackState, frame, roi):
        # Initialize model
        pass
    
    def update(self, state: TrackState, frame):
        # Update tracking window
        return new_window
```

---

## 📝 中文说明 (Chinese Notes)

### 如果要修改代码 (If You Want to Modify the Code)

#### 1. 修改跟踪参数 (Modify Tracking Parameters)
在 `test/basic_questions.ipynb` 中调整参数：

**Mean-shift 参数:**
```python
tracker = ClassicalTracker(
    video_path=VIDEO_PATH_MUG,
    method='meanshift',
    color_space='hue',        # 'hue', 'hsv', 'rgb'
    update_model=True,         # 是否自适应更新模型
    update_rate=0.05           # 更新率 (0.01-0.1)
)
```

**Hough Transform 参数:**
```python
tracker = ClassicalTracker(
    video_path=VIDEO_PATH_BALL,
    method='hough',
    gradient_threshold=30,     # 梯度阈值 (20-50)
    angle_bins=36,             # 角度分组数 (36-72)
    gaussian_blur_ksize=5,     # 高斯平滑核大小 (3, 5, 7)
    search_window_expand=1.25, # 搜索区域扩展倍数 (1.2-1.5)
    vote_weight='magnitude'    # 投票权重 ('magnitude' or 'uniform')
)
```

#### 2. 添加新的跟踪方法 (Add New Tracking Method)
在 `src/classical_tracker.py` 中：

```python
# Step 1: 定义新策略类
class YourNewStrategy(TrackerStrategy):
    def __init__(self, *, your_param1, your_param2):
        self.param1 = your_param1
        self.param2 = your_param2
    
    def init(self, state: TrackState, frame, roi):
        # 初始化你的模型
        state.model = your_initialization(frame, roi)
        state.track_window = roi
    
    def update(self, state: TrackState, frame):
        # 实现跟踪逻辑
        new_window = your_tracking_logic(state, frame)
        state.track_window = new_window
        return new_window

# Step 2: 在 _build_strategy() 中注册
def _build_strategy(self, method, kwargs, ...):
    if method == 'meanshift':
        return MeanShiftStrategy(...)
    elif method == 'hough':
        return HoughTransformStrategy(...)
    elif method == 'your_method':  # 添加这里
        return YourNewStrategy(
            your_param1=kwargs.get('your_param1', default_value),
            your_param2=kwargs.get('your_param2', default_value)
        )
```

#### 3. 修改可视化 (Modify Visualization)
在 `src/features.py` 中添加新的可视化函数，然后在 `classical_tracker.py` 的 `track_video()` 中调用。

#### 4. 修改 ROI 选择 (Modify ROI Selection)
在 `src/utils.py` 的 `ROISelector` 类中修改鼠标回调逻辑。

#### 5. 常见问题 (Common Issues)
- **ROI 选择框不正确**: 已修复 xy 轴问题，确保使用最新代码
- **窗口无法关闭**: 按 ESC 或 'q' 键退出
- **结果不保存**: 检查 `save_result=True` 和 `output_dir` 参数






