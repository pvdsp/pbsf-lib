"""Distance Profile of a query compared to a full time series."""

import numpy as np
from scipy.spatial import distance

from pbsf.segmenters.sliding_window import SlidingWindow


def distance_profile(query: np.ndarray, data: np.ndarray,
                     parameters: dict) -> np.ndarray:
    """
    Compute exact or approximate z-normalised distance from query to segments.

    Splits data up into z-normalised segments in size equal to the query.
    Returns the distance of each of these segments to the z-normalised query.
    Note: both the query and data segments are z-normalised.

    Parameters
    ----------
    query : np.ndarray
        The query segment.
    data : np.ndarray
        Data to calculate the distance from query.
    parameters : dict
        Algorithm configuration. Supported keys:

        - ``discretiser`` : Discretiser class
        - ``discretiser_params`` : dict passed to the discretiser constructor.
        See discretiser documentation for accepted discretiser parameters.

        If no ``discretiser`` is provided, returns the exact Euclidean distance profile.

    Returns
    -------
    np.ndarray
        Per-segment distance to the query; index i corresponds to data[i:i+window_size].
    """
    window_size = len(query)
    segmenter = SlidingWindow({
        "window_size": window_size,
        "z_normalisation": True
    })

    query = segmenter.segment(query)[0]  # preprocess query through segmenter
    segments = segmenter.segment(data)
    distances = np.full(len(data)-window_size+1, np.inf)

    if (discretiser_type := parameters.get("discretiser")) is None:
        for i, segment in enumerate(segments):
            distances[i] = distance.euclidean(query, segment)
    else:
        params = parameters.get("discretiser_params") or {}
        discretiser = discretiser_type(params)
        n1 = discretiser.discretise(query)[-1]
        for i, segment in enumerate(segments):
            n2 = discretiser.discretise(segment)[-1]
            distances[i] = n1.distance(n2)

    return distances
