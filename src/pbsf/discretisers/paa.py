from collections.abc import Callable
from typing import Any

import numpy as np

from pbsf.discretisers.base import Discretiser, _divide, _normalise
from pbsf.nodes import PAANode
from pbsf.utils import has_required


class PiecewiseAggregate(Discretiser):
    """
    Discretise contiguous subsequences in increasing granularity using Piecewise Aggregate Approximation.

    PAA reduces dimensionality by dividing a segment into frames and representing
    each frame by its mean value.

    Parameters
    ----------
    params : dict[str, Any] | None, default=None
        Configuration dictionary with the following required keys:

        - max_depth (Callable): Function that returns the maximum depth of the
          chain given the data.
        - frames (Callable): Function that returns the number of frames for a
          given depth.
        - node_type (type): The type of node to use for the chain. Must be PAANode.
        - node_params (dict): The parameters to pass to the node constructor.
    """
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        has_required(params, [
            ("max_depth", Callable),
            ("frames", Callable),
            ("node_type", [PAANode]),
            ("node_params", dict)
        ])
        self.max_depth = params["max_depth"]
        self.frames = params["frames"]
        self.node_type = params["node_type"]
        self.node_params = params["node_params"]

    def discretise(self, segment: np.ndarray) -> list:
        """
        Discretise a segment in increasing granularity using Piecewise Aggregate Approximation.

        Parameters
        ----------
        segment : np.ndarray
            The segment to discretise.

        Returns
        -------
        list
            A list of nodes representing the segment in increasing granularity,
            forming a chain from coarse to fine approximations.

        Raises
        ------
        ValueError
            If the segment is not 1D.
        """
        nodes = []
        segment = _normalise(segment)
        if segment.ndim != 1:
            raise ValueError(f"Can only discretise 1D data.")
        for depth in range(self.max_depth(segment)):
            breakpoints = _divide(0, len(segment), self.frames(depth))
            paa = np.array([np.mean(segment[start:end]) for start, end in breakpoints])
            node = self.node_type({
                "depth": depth,
                "segment_length": len(segment),
                "nr_of_frames": self.frames(depth),
                "breakpoints": breakpoints,
                "paa": paa,
                **self.node_params
            })
            nodes.append(node)
        return nodes