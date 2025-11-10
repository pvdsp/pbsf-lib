from collections.abc import Callable
from typing import Any

import numpy as np

from pbsf.discretisers.base import Discretiser, _divide, _normalise, _piecewise_linear
from pbsf.nodes import SlopeSignNode, StructuralProminenceNode, PLANode
from pbsf.utils import has_required


class PiecewiseLinear(Discretiser):
    """
    Discretise contiguous subsequences in increasing granularity using Piecewise Linear Approximation.

    Parameters
    ----------
    params : dict[str, Any] | None, default=None
        Configuration dictionary with the following required keys:

        - max_depth (Callable): Function that returns the maximum depth of the
          chain given the data.
        - frames (Callable): Function that returns the number of frames for a
          given depth.
        - node_type (type): The type of node to use for the chain. Must be one
          of SlopeSignNode, StructuralProminenceNode, or PLANode.
        - node_params (dict): The parameters to pass to the node constructor.
    """
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        has_required(params, [
            ("max_depth", Callable),
            ("frames", Callable),
            ("node_type", [SlopeSignNode, StructuralProminenceNode, PLANode]),
            ("node_params", dict)
        ])
        self.params = params
        self.max_depth = params["max_depth"]
        self.frames = params["frames"]
        self.node_type = params["node_type"]
        self.node_params = params["node_params"]

    def discretise(self, segment: np.ndarray) -> list:
        """
        Discretise a segment in increasing granularity using Piecewise Linear Approximation.

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
        std = np.std(segment)
        segment = _normalise(segment)
        if segment.ndim != 1:
            raise ValueError(f"Can only discretise 1D data.")
        for depth in range(self.max_depth(segment)):
            breakpoints = _divide(0, len(segment), self.frames(depth))
            slopes, intercepts = _piecewise_linear(segment, breakpoints)
            node = self.node_type({
                "depth": depth,
                "std": float(std),
                "slopes": slopes,
                "intercepts": intercepts,
                "breakpoints": breakpoints,
                **self.node_params
            })
            nodes.append(node)
        return nodes