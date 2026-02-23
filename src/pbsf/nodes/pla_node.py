from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pbsf.nodes.base import Node
from pbsf.utils import has_required


class PLANode(Node):
    """
    Node representing a piecewise linear approximation of a segment.

    This node uses slopes and intercepts to represent segment via
    Piecewise Linear Approximation (PLA). Distance comparisons use the
    PLA distance metric (dist_PLA).

    Parameters
    ----------
    properties : dict[str, Any]
        Configuration dictionary with the following required keys:

        - depth (int): Depth of the node in the chain.
        - slopes (np.ndarray): Array of slopes of the linear segments.
        - intercepts (np.ndarray): Array of intercepts of the linear segments.
        - breakpoints (list): List of (start, end) tuples defining the segments.
        - distance_threshold (Callable): Function that returns the distance
          threshold for the PLA distance measure at a given depth.
    """
    def __init__(self, properties: dict[str, Any]) -> None:
        has_required(properties, [
            ("depth", int),
            ("slopes", np.ndarray),
            ("intercepts", np.ndarray),
            ("breakpoints", list),
            ("distance_threshold", Callable)
        ])
        self.depth = properties["depth"]
        self.slopes = properties["slopes"]
        self.intercepts = properties["intercepts"]
        self.breakpoints = properties["breakpoints"]
        self.distance_threshold = properties["distance_threshold"](self.depth)

    def show(self) -> None:
        """
        Visualise the piecewise linear approximation.

        Draws vertical lines at breakpoints and plots the linear segments
        defined by the slopes and intercepts.
        """
        pairs = zip(self.slopes, self.intercepts)
        for (x1, x2), (a, b) in zip(self.breakpoints, pairs):
            plt.axvline(x1, color="lightgrey", linestyle=":")
            plt.axvline(x2, color="lightgrey", linestyle=":")
            x = np.linspace(0, x2 - x1, 100)
            plt.plot(np.linspace(x1, x2, 100),
                     a * x + b, color="orangered", linestyle="--")

    def distance(self, node: 'PLANode') -> float:
        """
        Calculate the PLA distance between this node and another node.

        Parameters
        ----------
        node : PLANode
            Another PLANode to calculate distance to.

        Returns
        -------
        float
            PLA distance metric between the two nodes.

        Raises
        ------
        ValueError
            If nodes are not comparable (different types, depths, or thresholds).
        """
        if not isinstance(node, PLANode):
            raise ValueError(
                f"Cannot compare node of type {type(self)}"
                f" with {type(node)}."
            )
        if self.depth != node.depth:
            raise ValueError("Cannot compare nodes of different depths.")
        if self.distance_threshold != node.distance_threshold:
            raise ValueError("Cannot compare nodes with different distance thresholds.")
        slopes = self.slopes - node.slopes
        intercepts = self.intercepts - node.intercepts
        length = self.breakpoints[0][1] - self.breakpoints[0][0]
        j_values = np.arange(1, length + 1).reshape(-1, 1)
        result = slopes * j_values + intercepts
        return np.sqrt(np.sum(result ** 2))

    def __eq__(self, node: 'PLANode') -> bool:
        """
        Check equivalence between this node and another PLANode.

        Two PLANodes are considered equivalent if they have the same depth,
        distance threshold, and their PLA distance is within the threshold.

        Parameters
        ----------
        node : PLANode
            Another node to compare with.

        Returns
        -------
        bool
            True if nodes are equivalent (distance ≤ threshold), False otherwise.
        """
        if not isinstance(node, PLANode):
            return False
        if self.depth != node.depth:
            return False
        if self.distance_threshold != node.distance_threshold:
            return False
        return self.distance(node) <= self.distance_threshold

    def __repr__(self) -> str:
        """
        Return string representation of the PLANode.

        Returns
        -------
        str
            String representation showing depth, slopes,
            and intercepts (rounded to 2 decimals).
        """
        return (
            f"PLANode(depth={self.depth}, "
            f"slopes={np.round(self.slopes, 2)}, "
            f"intercepts={np.round(self.intercepts, 2)})"
        )
