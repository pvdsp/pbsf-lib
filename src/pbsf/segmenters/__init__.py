"""Segmentation strategies for partitioning time series into subsequences."""

from .base import Segmenter
from .sliding_window import SlidingWindow

__all__ = ["Segmenter", "SlidingWindow"]
