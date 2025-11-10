
# TP_Tracking_2025 - Video Object Tracking

## 📋 Project Overview

This project implements and compares various video object tracking algorithms, from classical methods to state-of-the-art deep learning approaches.

## 🎯 Implemented Methods

### 1️⃣ **Classical Methods** (Q1-Q5)

#### **Mean Shift Tracking** (Q1-Q2)
- **Basic Implementation**: Hue histogram-based tracking
- **Improvements**:
  - Dual-channel (H+S) histogram
  - Adaptive model update
  - Enhanced density computation

#### **Hough Transform Tracking** (Q3-Q5)
- **Basic Implementation**: Gradient-based R-Table tracking
- **Improvements**:
  - Motion prediction using velocity
  - Adaptive model update

### 2️⃣ **Deep Learning Methods** (Q6)

#### **Feature-Based Tracking**
Uses pre-trained CNN features combined with classical methods.

**Supported Backbones:**
- ResNet-18, ResNet-50
- VGG-16
- MobileNet
- EfficientNet

**Feature Selection Methods:**
- Variance-based
- Correlation-based
- Mutual information

### 3️⃣ **State-of-the-Art Methods** (SOTA Comparison)

We integrate recent SOTA methods for comparison. **Note: Pick ONE to run due to computational constraints.**

#### **Option 1: SeqTrack (CVPR 2023)** ⭐ Recommended for beginners
- **Paper**: "Sequence to Sequence Learning for Visual Object Tracking"
- **GitHub**: https://github.com/chenxin-dlut/SeqTrackv2
- **Key Features**:
  - Simple encoder-decoder architecture
  - Sequence generation approach
  - Easy to understand and implement
- **Performance**: LaSOT AUC 69.9%, TrackingNet AUC 86.1%

**Usage:**
```bash
cd sota_methods/seqtrack
python demo.py --video_path ../../data/Sequences/Antoine_Mug.mp4
```

#### **Option 2: ARTrackV2 (CVPR 2024)** ⭐⭐ **Recommended - Latest SOTA**
- **Paper**: "ARTrackV2: Prompting Autoregressive Tracker Where to Look and How to Describe"
- **GitHub**: https://github.com/MIV-XJTU/ARTrack
- **Key Features**:
  - Autoregressive framework
  - Joint trajectory and appearance modeling
  - State-of-the-art performance
- **Performance**: GOT-10k AO 79.5%, TrackingNet AUC 86.1%, **3.6x faster than ARTrack**

**Usage:**
```bash
cd sota_methods/artrackv2
python demo.py --video_path ../../data/Sequences/Antoine_Mug.mp4
```

#### **Option 3: MixFormerV2 (NeurIPS 2023)** ⭐ Recommended for efficiency
- **Paper**: "MixFormerV2: Efficient Fully Transformer Tracking"
- **GitHub**: https://github.com/MCG-NJU/MixFormer
- **Key Features**:
  - Fully transformer architecture
  - Ultra-fast (165 FPS on GPU)
  - First CPU real-time transformer tracker
- **Performance**: LaSOT AUC 70.6%, TNL2k AUC 57.4%

**Usage:**
```bash
cd sota_methods/mixformerv2
python demo.py --video_path ../../data/Sequences/Antoine_Mug.mp4
```

#### **Other ICCV/CVPR 2024-2025 Methods** (Optional)

**For discussion in report only:**
- **ChatTracker (NeurIPS 2024)**: Multi-modal LLM for tracking
- **DiffusionTrack (CVPR 2024)**: Diffusion model-based tracking
- **VastTrack (NeurIPS 2024)**: Large-scale category tracking
- **MOTIP (CVPR 2025)**: MOT as ID prediction
- **SambaMOTR (ICLR 2025)**: Synchronized sequence modeling
- **CO-MOT (ICLR 2025)**: Coopetition label assignment


