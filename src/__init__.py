"""
Tracking package initialization
"""
from .classical_tracker import ClassicalTracker
from .utils import ROISelector, visualize_tracking, save_frame
from .features import (
    extract_color_histogram, 
    compute_backprojection,
    visualize_hue_and_backprojection
)

__all__ = [
    'ClassicalTracker',
    'ROISelector',
    'visualize_tracking',
    'save_frame',
    'extract_color_histogram',
    'compute_backprojection',
    'visualize_hue_and_backprojection'
]