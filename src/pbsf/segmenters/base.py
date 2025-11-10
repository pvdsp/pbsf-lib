from abc import ABC, abstractmethod

import numpy as np

from ..utils import has_required


class Segmenter(ABC):
    """
    Abstract base class for data segmentation algorithms.

    This class defines the interface for segmenters that partition data
    into contiguous subsequences. Subclasses must implement the segment()
    method to define their specific segmentation strategy.
    """
    @abstractmethod
    def segment(self, data: np.ndarray) -> np.ndarray:
        """
        Segment input data into contiguous subsequences.

        Parameters
        ----------
        data : np.ndarray
            Input data to segment.

        Returns
        -------
        np.ndarray
            Segmented data.
        """
        pass