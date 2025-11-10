from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pbsf.nodes.base import Node
from pbsf.utils import has_required


class PAANode(Node):
    """
    Node representing a Piecewise Aggregate Approximation (PAA) of a segment.

    PAA reduces dimensionality by dividing a segment into frames and representing
    each frame by its mean value. This node uses Euclidean distance for comparison,
    scaled by the segment and frame parameters.

    Parameters
    ----------
    properties : dict[str, Any]
        Configuration dictionary with the following required keys:

        - depth (int): Depth of the node in the chain.
        - segment_length (int): Length of the original segment.
        - nr_of_frames (int): Number of frames in the PAA representation.
        - breakpoints (list): List of (start, end) tuples defining the frames.
        - paa (np.ndarray): Array of mean values for each frame.
        - distance_threshold (Callable): Function that returns the distance
          threshold at a given depth.
    """
    def __init__(self, properties: dict[str, Any]) -> None:
        has_required(properties, [
            ("depth", int),
            ("segment_length", int),
            ("nr_of_frames", int),
            ("breakpoints", list),
            ("paa", np.ndarray),
            ("distance_threshold", Callable)
        ])
        self.depth = properties["depth"]
        self.n = properties["segment_length"]
        self.frames = properties["nr_of_frames"]
        self.breakpoints = properties["breakpoints"]
        self.paa = properties["paa"]
        self.distance_threshold = properties["distance_threshold"](self.depth)

    def _euclidean_distance(self, other: 'PAANode') -> float:
        """
        Calculate the raw Euclidean distance between PAA representations.

        Parameters
        ----------
        other : PAANode
            Another PAANode to calculate distance to.

        Returns
        -------
        float
            Euclidean distance between the PAA arrays.

        Raises
        ------
        ValueError
            If nodes are not comparable (different types, depths, or thresholds).
        """
        if not isinstance(other, PAANode):
            raise ValueError(f"Cannot compare node of type {type(self)} with {type(other)}.")
        if self.depth != other.depth:
            raise ValueError(f"Cannot compare nodes of different depths.")
        if self.distance_threshold != other.distance_threshold:
            raise ValueError(f"Cannot compare nodes with different distance thresholds.")
        return np.linalg.norm(self.paa - other.paa)

    def show(self) -> None:
        """
        Visualise the PAA representation.

        Draws vertical lines at frame breakpoints and horizontal lines
        representing the mean value for each frame.
        """
        for (x1, x2) in self.breakpoints:
            plt.axvline(x1, color="lightgrey", linestyle=":")
            plt.axvline(x2, color="lightgrey", linestyle=":")
        for (x1, x2), y in zip(self.breakpoints, self.paa):
            plt.hlines(y=y, xmin=x1, xmax=x2, color="orangered", linestyle="--")

    def distance(self, node: 'PAANode') -> float:
        """
        Calculate the scaled Euclidean distance between two PAA nodes.

        The distance is scaled by sqrt(segment_length / nr_of_frames) to account
        for the dimensionality reduction performed by PAA.

        Parameters
        ----------
        node : PAANode
            Another PAANode to calculate distance to.

        Returns
        -------
        float
            Scaled Euclidean distance between the PAA representations.

        Raises
        ------
        ValueError
            If nodes are not comparable (different types, depths, or thresholds).
        """
        return np.sqrt(self.n / self.frames) * self._euclidean_distance(node)

    def __eq__(self, other: 'PAANode') -> bool:
        """
        Check equivalence between this node and another PAANode.

        Two PAANodes are considered equivalent if they have the same depth,
        distance threshold, and their scaled distance is within the threshold.

        Parameters
        ----------
        other : PAANode
            Another node to compare with.

        Returns
        -------
        bool
            True if nodes are equivalent (distance ≤ threshold), False otherwise.
        """
        if not isinstance(other, PAANode):
            return False
        if self.depth != other.depth:
            return False
        if self.distance_threshold != other.distance_threshold:
            return False
        return self.distance(other) <= self.distance_threshold

    def __repr__(self) -> str:
        """
        Return string representation of the PAANode.

        Returns
        -------
        str
            String representation showing depth, PAA values (rounded to 2 decimals),
            and distance threshold.
        """
        return f"PAANode(depth={self.depth}, " \
               f"paa={np.round(self.paa, 2)}, distance_threshold={self.distance_threshold})"