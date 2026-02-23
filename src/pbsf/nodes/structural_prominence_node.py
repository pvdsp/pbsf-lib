from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pbsf.nodes.base import Node
from pbsf.utils import has_required


class StructuralProminenceNode(Node):
    """
    Node representing a piecewise linear approximation using
    structural and prominence distances.

    This node uses slopes and intercepts to represent segment discretisations via
    Piecewise Linear Approximation (PLA). Two distance metrics are used for comparison:
    structural distance (based on slopes and intercepts) and prominence distance
    (based on standard deviation).

    Parameters
    ----------
    properties : dict[str, Any]
        Configuration dictionary with the following required keys:

        - depth (int): Depth of the node in the chain.
        - std (float): Standard deviation of the segment.
        - slopes (np.ndarray): Array of slopes of the linear segments.
        - intercepts (np.ndarray): Array of intercepts of the linear segments.
        - breakpoints (list): List of (start, end) tuples defining the segments.
        - structural_threshold (Callable): Function that returns the threshold for
          structural distance at a given depth.
        - prominence_threshold (Callable): Function that returns the threshold for
          prominence distance at a given depth.

    Examples
    --------
    Constant thresholds:

        node = StructuralProminenceNode({
            'depth': 0,
            'structural_threshold': lambda depth: 0.1,
            'prominence_threshold': lambda depth: 0.1,
            ...
        })

    Thresholds that decrease with depth:

        node = StructuralProminenceNode({
            'depth': 0,
            'structural_threshold': lambda depth: 1 / (depth + 1),
            'prominence_threshold': lambda depth: 1 / (depth + 1),
            ...
        })
    """
    def __init__(self, properties: dict[str, Any]) -> None:
        has_required(properties, [
            ("depth", int),
            ("std", float),
            ("slopes", np.ndarray),
            ("intercepts", np.ndarray),
            ("breakpoints", list),
            ("structural_threshold", Callable),
            ("prominence_threshold", Callable)
        ])
        self.depth = properties["depth"]
        self.std = properties["std"]
        self.slopes = properties["slopes"]
        self.intercepts = properties["intercepts"]
        self.breakpoints = properties["breakpoints"]
        self.structural_threshold = properties["structural_threshold"](self.depth)
        self.prominence_threshold = properties["prominence_threshold"](self.depth)

    def _is_comparable(self, node: 'Node') -> None:
        """
        Validate that another node can be compared with this node.

        Parameters
        ----------
        node : Node
            Node to validate for comparison.

        Raises
        ------
        ValueError
            If node is not a StructuralProminenceNode or has
            different depth or thresholds.
        """
        if not isinstance(node, StructuralProminenceNode):
            raise ValueError(
                f"Cannot compare node of type {type(self)}"
                f" with {type(node)}."
            )
        if self.depth != node.depth:
            raise ValueError(
                "Cannot compare nodes of different depths."
            )
        if self.structural_threshold != node.structural_threshold:
            raise ValueError(
                "Cannot compare nodes with different"
                " structural thresholds."
            )
        if self.prominence_threshold != node.prominence_threshold:
            raise ValueError(
                "Cannot compare nodes with different"
                " prominence thresholds."
            )

    def structural_distance(self, node: 'StructuralProminenceNode') -> float:
        """
        Calculate the structural distance based on slopes and intercepts.

        Parameters
        ----------
        node : StructuralProminenceNode
            Another StructuralProminenceNode to calculate distance to.

        Returns
        -------
        float
            Mean of the sum of slope and intercept differences.

        Raises
        ------
        ValueError
            If nodes are not comparable (different types or depths).
        """
        self._is_comparable(node)
        slopes = self.slopes - node.slopes
        inters = self.intercepts - node.intercepts
        return np.mean(slopes + inters)

    def prominence_distance(self, node: 'StructuralProminenceNode') -> float:
        """
        Calculate the prominence distance based on standard deviation ratio.

        Parameters
        ----------
        node : StructuralProminenceNode
            Another StructuralProminenceNode to calculate distance to.

        Returns
        -------
        float
            Ratio of standard deviations minus 1.0.

        Raises
        ------
        ValueError
            If nodes are not comparable (different types or depths).
        """
        self._is_comparable(node)
        maximum = max(self.std, node.std)
        minimum = min(self.std, node.std)
        return maximum / (minimum + 1e-10) - 1.0

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

    def distance(self, node: 'StructuralProminenceNode') -> float:
        """
        Calculate the combined distance between this node and another node.

        The distance is the sum of the absolute structural and prominence distances.

        Parameters
        ----------
        node : StructuralProminenceNode
            Another StructuralProminenceNode to calculate distance to.

        Returns
        -------
        float
            Sum of absolute structural and prominence distances.

        Raises
        ------
        ValueError
            If nodes are not comparable (different types or depths).
        """
        self._is_comparable(node)
        structural = abs(self.structural_distance(node))
        prominence = abs(self.prominence_distance(node))
        return structural + prominence

    def __eq__(self, node: 'StructuralProminenceNode') -> bool:
        """
        Check equivalence between this node and another StructuralProminenceNode.

        Two StructuralProminenceNodes are considered equivalent if they have the same
        depth, thresholds, and both structural and prominence distances are within their
        respective thresholds.

        Parameters
        ----------
        node : StructuralProminenceNode
            Another node to compare with.

        Returns
        -------
        bool
            True if nodes are equivalent (both distances ≤ thresholds), False otherwise.
        """
        if not isinstance(node, StructuralProminenceNode):
            return False
        if self.depth != node.depth:
            return False
        if self.structural_threshold != node.structural_threshold:
            return False
        if self.prominence_threshold != node.prominence_threshold:
            return False

        structural = abs(self.structural_distance(node)) <= self.structural_threshold
        prominence = abs(self.prominence_distance(node)) <= self.prominence_threshold
        return structural and prominence

    def __repr__(self) -> str:
        """
        Return string representation of the StructuralProminenceNode.

        Returns
        -------
        str
            String representation showing depth, std, slopes,
            and intercepts (rounded to 2 decimals).
        """
        return (
            f"StructuralProminenceNode(depth={self.depth},"
            f" std={round(self.std, 2)},"
            f" slopes={np.round(self.slopes, 2)},"
            f" intercepts={np.round(self.intercepts, 2)})"
        )
