
# Object Tracking Toolkit — Minimal README

Purpose
-------
Small toolkit with reference implementations of classical trackers (Mean-shift, Hough) and a CNN-feature variant, plus simple annotation and evaluation tools.

Quickstart (minimal)
---------------------
1. Create a venv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the default Mean-shift demo and save results:

```bash
python - <<'PY'
from src.classical_tracker import ClassicalTracker
ClassicalTracker(video_path='Test-Videos/Antoine_Mug.mp4', method='meanshift', color_space='hue')\
  .track_video(visualize=False, save_result=True, output_dir='results/evaluation/meanshift_mug')
PY
```

Where to look
--------------
- `src/` — code (trackers, features, utils)
- `scripts/` — annotation and helper scripts
- `test/` — demo notebooks
- `results/` — output (not versioned)

Evaluation
----------
Use `src/evaluation.evaluate(pred_csv, gt_csv, cle_threshold=20.0)` to compute IoU, center error and common metrics.

Contributing
------------
Add a `TrackerStrategy` implementation in `src/classical_tracker.py` and register it in the factory. Keep changes focused and add a short demo.

License / Contact
-----------------
Provided for research and education. Open an issue for help or requests.

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

### Q6: Deep Learning-based Tracking (✅ Explained)

**Question**: How to improve histogram and Hough-based tracking using CNN features?

**Solution**:
- Replace color histograms/gradients with CNN feature channels
- Use pre-trained network (ResNet-50 recommended)
- Smart layer and channel selection strategies

#### Key Concepts:

**1. Layer Selection Strategy:**
- **Criteria**:
  - Receptive Field (RF) should match object size
  - Mid-level layers (layer3/conv4_3) provide best balance
  - Higher resolution → better localization
- **Recommendation**: 
  - ResNet-50 `layer3` (1024 channels, 14×14, RF=267px)
  - VGG-16 `conv4_3` (512 channels, 28×28, RF=52px)

**2. Channel Selection Methods:**
- **Variance-based**: Select high-variance channels (discriminative)
- **Max Response**: Select high-activation channels (relevant)
- **Gradient-based**: Select channels with strong edges (structural)
- **Recommendation**: Use variance or gradient method, select top K=32-64 channels

**3. Integration Approaches:**

**A. Deep Mean-shift:**
```
Traditional: Color Histogram → Back-projection → Mean-shift
Deep:        CNN Features → Feature "Histogram" → Mean-shift
```
- Build multi-dimensional histogram from selected channels
- Compute back-projection using feature similarity
- Apply standard mean-shift algorithm

**B. Deep Hough Transform:**
```
Traditional: Image Gradients → R-Table → Voting
Deep:        Feature Gradients → R-Table → Voting
```
- Compute gradients on CNN feature maps
- Build R-Table from feature orientations
- Vote using semantic edge information



#### Implementation:
See `src/deep_tracker.py` for complete implementation:
- `CNNFeatureExtractor`: Extract features from pre-trained networks
- `DeepMeanShiftTracker`: Mean-shift using CNN features
- Detailed documentation and selection strategies included

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

