from abc import ABC, abstractmethod

import numpy as np

from ..utils import has_required


class Discretiser(ABC):
    """
    Abstract base class for discretisation algorithms.

    This class defines the interface for discretisers that convert segmented data
    into discrete representations. Subclasses must implement the discretise()
    method to define their specific discretisation strategy.
    """
    @abstractmethod
    def discretise(self, segment: np.ndarray) -> np.ndarray:
        """
        Convert a segment into a discrete representation.

        Parameters
        ----------
        segment : np.ndarray
            Single segment to discretise.

        Returns
        -------
        np.ndarray
            Discretised representation of the segment.
        """
        pass


def _divide(begin: int, end: int, number: int) -> list[tuple[int, int]]:
    """
    Divide an interval into equally-sized frames.

    Parameters
    ----------
    begin : int
        Start of the interval (inclusive).
    end : int
        End of the interval (exclusive).
    number : int
        Number of frames to divide the interval into.

    Returns
    -------
    list[tuple[int, int]]
        List of (start, end) tuples representing each frame.

    Raises
    ------
    ValueError
        If number is less than 1 or greater than the interval length.
    """
    if number < 1:
        raise ValueError(f"Amount of frames must be greater than 0, got {number}")
    if number > end - begin:
        raise ValueError(f"Amount of frames must be smaller than sequence length ({end - begin}), got {number}")
    if number == 1:
        return [(begin, end)]
    step = (end - begin) / number
    return [(int(begin + segment * step), int(begin + (segment + 1) * step)) for segment in range(number)]


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


def _piecewise_linear(segment: np.ndarray, breakpoints: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute piecewise linear approximation of a segment.

    Fits a linear function to each frame defined by the breakpoints and returns
    the slopes and intercepts of these linear approximations.

    Parameters
    ----------
    segment : np.ndarray
        1D array representing the segment to approximate.
    breakpoints : list[tuple[int, int]]
        List of (start, end) tuples defining the frames within the segment.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - slopes: Array of slopes for each frame
        - intercepts: Array of intercepts for each frame
    """
    frames = len(breakpoints)
    slopes = np.empty(frames)
    intercepts = np.empty(frames)
    for index, (begin, end) in enumerate(breakpoints):
        x = np.arange(0, end - begin)
        y = segment[begin:end]
        slopes[index], intercepts[index] = np.polyfit(x, y, 1)
    return slopes, intercepts