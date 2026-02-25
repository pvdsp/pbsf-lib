"""Summation-based discretiser for demonstration purposes."""

from collections.abc import Callable
from typing import Any

import numpy as np

from pbsf.discretisers.base import Discretiser, _divide
from pbsf.nodes import SumNode
from pbsf.utils import has_required


class Summation(Discretiser):
    """
    Discretise subsequences by summing frame values.

    This is a toy Discretiser implementation example for demonstration purposes.

    Parameters
    ----------
    params : dict[str, Any] | None, default=None
        Configuration dictionary with the following required keys:

        - max_depth (Callable): Function that returns the maximum depth of the
          chain given the data.
        - frames (Callable): Function that returns the number of frames for a
          given depth.
        - node_type (type): The type of the node to use for the chain. Must be
          SumNode.
        - node_params (dict): The parameters to pass to the node constructor.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        has_required(params, [
            ("max_depth", Callable),
            ("frames", Callable),
            ("node_type", [SumNode]),
            ("node_params", dict)
        ])
        self.max_depth = params["max_depth"]
        self.frames = params["frames"]
        self.node_type = params["node_type"]
        self.node_params = params["node_params"]

    def discretise(self, segment: np.ndarray) -> list:
        """
        Discretise a segment by splitting into frames and summing values.

        Parameters
        ----------
        segment : np.ndarray
            The segment to discretise.

        Returns
        -------
        list
            A list of nodes representing the segment in increasing granularity,
            forming a chain from coarse to fine-grained approximations.

        Raises
        ------
        ValueError
            If the segment is not 1D.
        """
        nodes = []
        if segment.ndim != 1:
            raise ValueError("Can only discretise 1D data.")
        for depth in range(self.max_depth(segment)):
            breakpoints = _divide(0, len(segment), self.frames(depth))
            sums = np.array([np.sum(segment[i:j]) for (i, j) in breakpoints])
            nodes.append(
                self.node_type({
                    "depth": depth,
                    "sums": sums,
                    **self.node_params
                })
            )
        return nodes
