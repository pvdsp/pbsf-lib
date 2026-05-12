"""Abstract base class for segmentation algorithms."""

from abc import ABC, abstractmethod

import numpy as np


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


def _normalise(data: np.ndarray) -> np.ndarray:
    """
    Compute the z-score normalisation of the data.

    Parameters
    ----------
    data : np.ndarray
        1D array to normalise.

    Returns
    -------
    np.ndarray
        Z-score normalised data. Returns array of zeros if standard deviation is zero.

    Raises
    ------
    ValueError
        If data is empty or not 1D.
    """
    if len(data) == 0:
        raise ValueError("Cannot normalise empty sequence")
    if data.ndim != 1:
        raise ValueError(f"Can only normalise 1D data, got {data.ndim}D data")
    if np.std(data) == 0:
        return np.zeros(len(data))
    return (data - np.mean(data)) / np.std(data)
