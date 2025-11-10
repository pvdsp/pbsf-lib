from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.stats

from pbsf.discretisers.base import Discretiser, _normalise
from pbsf.discretisers.paa import PiecewiseAggregate
from pbsf.nodes import SAXNode, PAANode
from pbsf.utils import has_required


class SymbolicAggregate(Discretiser):
    """
    Discretise contiguous subsequences in increasing granularity using Symbolic Aggregate Approximation.

    SAX converts a time series into a symbolic representation by first applying
    PAA (Piecewise Aggregate Approximation) and then discretising the resulting
    values into symbols using breakpoints based on dividing a Gaussian
    distribution into equiprobable regions.

    Parameters
    ----------
    params : dict[str, Any] | None, default=None
        Configuration dictionary with the following required keys:

        - max_depth (Callable): Function that returns the maximum depth of the
          chain given the data.
        - frames (Callable): Function that returns the number of frames for a
          given depth.
        - alphabet_size (int): Number of symbols in the alphabet (must be ≥ 2).
        - node_type (type): The type of node to use for the chain. Must be SAXNode.
        - node_params (dict): The parameters to pass to the node constructor.
    """
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        has_required(params, [
            ("max_depth", Callable),
            ("frames", Callable),
            ("alphabet_size", int),
            ("node_type", [SAXNode]),
            ("node_params", dict)
        ])
        self.max_depth = params["max_depth"]
        self.frames = params["frames"]
        self.alphabet_size = params["alphabet_size"]
        self.node_type = params["node_type"]
        self.node_params = params["node_params"]
        self.cut_points = self._cut_points()

    def _cut_points(self) -> np.ndarray:
        """
        Calculate Gaussian-based breakpoints for symbol discretisation.

        Generates breakpoints that divide the standard normal distribution into
        equally probable regions.

        Returns
        -------
        np.ndarray
            Array of cut points for discretising PAA values into symbols.

        Raises
        ------
        ValueError
            If alphabet_size is less than 2.
        """
        if self.alphabet_size < 2:
            raise ValueError(f"Alphabet size must be at least 2, got {self.alphabet_size}")
        if self.alphabet_size == 2:
            cut_points = np.array([0.0])
        else:
            quantiles = np.linspace(1 / self.alphabet_size, 1 - 1 / self.alphabet_size, self.alphabet_size - 1)
            cut_points = scipy.stats.norm.ppf(quantiles)
        return cut_points

    def discretise(self, segment: np.ndarray) -> list:
        """
        Discretise a segment in increasing granularity using Symbolic Aggregate Approximation.

        First applies PAA to reduce dimensionality, then converts PAA values to
        symbols using the pre-calculated cut points.

        Parameters
        ----------
        segment : np.ndarray
            The segment to discretise.

        Returns
        -------
        list
            A list of nodes representing the segment in increasing granularity,
            forming a chain from coarse to fine symbolic approximations.

        Raises
        ------
        ValueError
            If the segment is not 1D.
        """
        nodes = []
        segment = _normalise(segment)
        if segment.ndim != 1:
            raise ValueError(f"Can only discretise 1D data.")
        paa = PiecewiseAggregate({
                "max_depth": self.max_depth,
                "frames": self.frames,
                "node_type": PAANode,
                "node_params": self.node_params
            })
        paa_chain = paa.discretise(segment)
        for paa_node in paa_chain:
            node = self.node_type({
                "depth": paa_node.depth,
                "segment_length": len(segment),
                "nr_of_frames": self.frames(paa_node.depth),
                "alphabet_size": self.alphabet_size,
                "breakpoints": paa_node.breakpoints,
                "sax": np.digitize(paa_node.paa, self.cut_points) - 1,
                "cut_points": self.cut_points,
                **self.node_params
            })
            nodes.append(node)
        return nodes