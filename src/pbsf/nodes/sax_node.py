from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pbsf.nodes.base import Node
from pbsf.utils import has_required


class SAXNode(Node):
    """
    Node representing a Symbolic Aggregate Approximation (SAX) of a segment.

    SAX converts a time series into a symbolic representation by first applying
    PAA (Piecewise Aggregate Approximation) and then discretising the resulting
    values into symbols using predefined breakpoints. This node uses the SAX
    distance metric for comparison.

    Parameters
    ----------
    properties : dict[str, Any]
        Configuration dictionary with the following required keys:

        - depth (int): Depth of the node in the chain.
        - segment_length (int): Length of the original segment.
        - nr_of_frames (int): Number of frames in the SAX representation.
        - breakpoints (list): List of (start, end) tuples defining the frames.
        - cut_points (np.ndarray): Array of breakpoints for symbol discretisation.
        - sax (np.ndarray): Array of symbol indices for each frame.
        - alphabet_size (int): Number of symbols in the alphabet.
        - distance_threshold (Callable): Function that returns the distance
          threshold at a given depth.
    """
    def __init__(self, properties: dict[str, Any]) -> None:
        has_required(properties, [
            ("depth", int),
            ("segment_length", int),
            ("nr_of_frames", int),
            ("breakpoints", list),
            ("cut_points", np.ndarray),
            ("sax", np.ndarray),
            ("alphabet_size", int),
            ("distance_threshold", Callable)
        ])
        self.depth = properties["depth"]
        self.n = properties["segment_length"]
        self.frames = properties["nr_of_frames"]
        self.breakpoints = properties["breakpoints"]
        self.sax = properties["sax"]
        self.alphabet_size = properties["alphabet_size"]
        self.cut_points = properties["cut_points"]
        self.distance_threshold = properties["distance_threshold"](self.depth)

    def _is_comparable(self, node: 'SAXNode') -> None:
        """
        Validate that another node can be compared with this node.

        Parameters
        ----------
        node : SAXNode
            Node to validate for comparison.

        Raises
        ------
        ValueError
            If node is not a SAXNode or has different depth,
            thresholds, segment lengths, or number of frames.
        """
        if not isinstance(node, SAXNode):
            raise ValueError(
                f"Cannot compare node of type {type(self)}"
                f" with {type(node)}."
            )
        if self.depth != node.depth:
            raise ValueError(
                "Cannot compare nodes of different depths."
            )
        if self.distance_threshold != node.distance_threshold:
            raise ValueError(
                "Cannot compare nodes with different"
                " distance thresholds."
            )
        if self.n != node.n or self.frames != node.frames:
            raise ValueError(
                f"Cannot compare nodes with different segment"
                f" lengths or number of frames: segment"
                f" lengths: {self.n} and {node.n},"
                f" frames: {self.frames} and {node.frames}."
            )

    def _dist(self, s1: int, s2: int) -> float:
        """
        Calculate the distance between two symbols using SAX distance lookup.

        Parameters
        ----------
        s1 : int
            Index of the first symbol.
        s2 : int
            Index of the second symbol.

        Returns
        -------
        float
            Distance between the two symbols. Returns 0.0 if symbols are adjacent
            or identical, otherwise returns the difference between their cut points.

        Raises
        ------
        ValueError
            If either symbol index is out of bounds for the alphabet size.
        """
        if s1 > self.alphabet_size - 1 or s2 > self.alphabet_size - 1:
            raise ValueError(
                f"Symbol index out of bounds: {s1}, {s2}"
                f" with alphabet size {self.alphabet_size}."
            )
        if abs(s1 - s2) <= 1:
            return 0.0
        return self.cut_points[max(s1, s2) - 1] - self.cut_points[min(s1, s2)]

    def distance(self, node: 'SAXNode') -> float:
        """
        Calculate the scaled SAX distance between two symbolic aggregate nodes.

        The distance uses a symbol-wise distance lookup combined with scaling
        based on the segment length and number of frames, similar to PAA distance.

        Parameters
        ----------
        node : SAXNode
            Another SAXNode to calculate distance to.

        Returns
        -------
        float
            Scaled SAX distance between the symbolic representations.

        Raises
        ------
        ValueError
            If nodes are not comparable (different types, depths, thresholds,
            segment lengths, or number of frames).
        """
        self._is_comparable(node)
        return np.sqrt(self.n / self.frames) * np.sqrt(np.sum([
            self._dist(s1, s2) ** 2 for s1, s2 in zip(self.sax, node.sax)
        ]))

    def __eq__(self, other: 'SAXNode') -> bool:
        """
        Check equivalence between this node and another SAXNode.

        Two SAXNodes are considered equivalent if they have the same depth,
        distance threshold, and their scaled distance is within the threshold.

        Parameters
        ----------
        other : SAXNode
            Another node to compare with.

        Returns
        -------
        bool
            True if nodes are equivalent (distance ≤ threshold), False otherwise.
        """
        if not isinstance(other, SAXNode):
            return False
        if self.depth != other.depth:
            return False
        if self.distance_threshold != other.distance_threshold:
            return False
        return self.distance(other) <= self.distance_threshold

    def show(self) -> None:
        """
        Visualise the SAX representation.

        Draws vertical lines at frame breakpoints, horizontal lines at symbol
        cut points, and fills regions corresponding to each symbol with colour.
        """
        for (x1, x2) in self.breakpoints:
            plt.axvline(x1, color="lightgrey", linestyle=":")
            plt.axvline(x2, color="lightgrey", linestyle=":")
        for cut_point in self.cut_points:
            plt.axhline(y=cut_point, color="lightgrey", linestyle=":")
        for (x1, x2), symbol in zip(self.breakpoints, self.sax):
            y1 = self.cut_points[symbol] if symbol >= 0 else -5
            y2 = self.cut_points[symbol + 1] if symbol + 1 < len(self.cut_points) else 5
            plt.fill_between(
                x=np.linspace(x1, x2, 100),
                y1=y1, y2=y2, color="orangered", alpha=0.5
            )

    def __repr__(self) -> str:
        """
        Return string representation of the SAXNode.

        Returns
        -------
        str
            String representation showing depth, SAX symbols, and alphabet size.
        """
        return (
            f"SAXNode(depth={self.depth}, "
            f"sax={self.sax.tolist()}, "
            f"alphabet_size={len(self.cut_points) + 1})"
        )
