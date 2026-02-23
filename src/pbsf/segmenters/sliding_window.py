from typing import Any

import numpy as np

from .base import Segmenter


class SlidingWindow(Segmenter):
    """
    Segmenter that uses a fixed-size sliding window to segment 1D data.

    Parameters
    ----------
    params : dict[str, Any] or None, optional
        Configuration dictionary with the following keys:

        - window_size (int): Size of the sliding window.
        - step_size (int, optional): Number of elements to move the window
          at each step. Default: 1.
        - differentiation (bool, optional): Whether to apply differentiation
          to the data before segmentation. Default: False.
        - autocorrelation (bool, optional): Whether to determine window size
          using autocorrelation. Default: False.
    """
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.autocorrelation = params.get("autocorrelation", False)
        self.differentiation = params.get("differentiation", False)
        self.window_size = params.get("window_size", None)
        if not self.window_size:
            if not self.autocorrelation:
                raise ValueError(
                    "`window_size` must be specified"
                    " if `autocorrelation` is False."
                )
            else:
                raise ValueError(
                    "`window_size` must be specified as a"
                    " fallback for when there is no"
                    " periodicity."
                )
        if self.window_size <= 0:
            raise ValueError("`window_size` must be greater than 0.")
        if self.autocorrelation:
            self.window_fallback = self.window_size
            self.window_size = None  # to be determined at first segmentation
        self.step_size = params.get("step_size", 1)
        if self.step_size <= 0:
            raise ValueError("`step_size` must be greater than 0.")

    def _ac_window_size(self, data: np.ndarray, minimum: int = 10) -> int:
        """
        Determine optimal window size using autocorrelation.

        This method detects periodic patterns in the data by computing the
        autocorrelation function and finding the lag with maximum correlation.

        Parameters
        ----------
        data : np.ndarray
            1D input data to analyze for periodicity.
        minimum : int, default=10
            Minimum lag to consider for periodicity detection. Lags shorter than
            this are ignored to avoid detecting trivial self-matches at small shifts
            as periodicity. Effectively acts as a minimum window size.

        Returns
        -------
        int
            The lag at which maximum autocorrelation occurs, representing the detected
            period length. Returns this value only if the autocorrelation coefficient
            exceeds 0.5, otherwise returns a fallback window size.

        References
        ----------
        Autocorrelation approach adapted from:
        https://stackoverflow.com/a/47369584
        """
        size = data.size
        norm = data - np.mean(data)
        autocorr = np.correlate(norm, norm, mode='same')
        autocorr = (
            autocorr[data.size // 2 + 1:]
            / (data.var() * np.arange(size - 1, size // 2, -1))
        )
        autocorr = autocorr[minimum:]
        lag = autocorr.argmax() + 1
        if autocorr[lag - 1] > 0.5:
            return lag + minimum
        return self.window_fallback

    def segment(self, data: np.ndarray) -> np.ndarray:
        """
        Segment 1D data using a sliding window approach.

        Parameters
        ----------
        data : np.ndarray
            1D array to segment.

        Returns
        -------
        np.ndarray
            2D array of shape (n_windows, window_size) containing
            the segmented data.

        Raises
        ------
        ValueError
            If data is not 1D, or if data length is less than window size.

        Notes
        -----
        If autocorrelation is enabled and this is the first call, the window size
        will be determined automatically and cached for subsequent calls.
        """
        if data.ndim != 1:
            raise ValueError("SlidingWindow only supports 1D data.")
        if self.autocorrelation and self.window_size is None:
            self.window_size = self._ac_window_size(data)
        if len(data) < self.window_size:
            raise ValueError(
                "Data length must be greater than or"
                " equal to window size."
            )
        if self.differentiation:
            data = np.diff(data)
        windows = np.lib.stride_tricks.sliding_window_view(
            data, self.window_size
        )
        return windows[::self.step_size]
