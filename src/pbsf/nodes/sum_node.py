from collections.abc import Callable
from typing import Any

import numpy as np

from pbsf.nodes.base import Node
from pbsf.utils import has_required


class SumNode(Node):
    """
    Node representing a segment by splitting it into multiple frames,
    and approximating each frame using their sum.

    This node uses the mean absolute difference of frame sums as the distance metric.
    Nodes are considered equivalent if their distance is within a given threshold.

    This is a toy Node implementation example for demonstration purposes.

    Parameters
    ----------
    properties : dict[str, Any]
        Configuration dictionary with the following required keys:

        - depth (int): Depth of the node in the chain.
        - sums (np.ndarray): Array of sum values.
        - distance_threshold (Callable): Function that returns the distance
          threshold at a given depth.
    """
    def __init__(self, properties: dict[str, Any]) -> None:
        has_required(properties, [
            ("sums", np.ndarray),
            ("depth", int),
            ("distance_threshold", Callable)
        ])
        self.depth = properties["depth"]
        self.sums = properties["sums"]
        self.distance_threshold = properties["distance_threshold"](self.depth)

    def distance(self, other: 'SumNode') -> float:
        """
        Calculate the mean absolute difference between sum arrays.

        Parameters
        ----------
        other : SumNode
            Another SumNode to calculate distance to.

        Returns
        -------
        float
            Mean absolute difference between the sum arrays.

        Raises
        ------
        ValueError
            If nodes are not comparable (different types, depths, or thresholds).
        """
        if not isinstance(other, SumNode):
            raise ValueError(
                f"Cannot compare node of type {type(self)}"
                f" with {type(other)}."
            )
        if self.depth != other.depth:
            raise ValueError("Cannot compare nodes of different depths.")
        if self.distance_threshold != other.distance_threshold:
            raise ValueError("Cannot compare nodes with different distance thresholds.")
        return np.mean(np.abs(self.sums - other.sums))

    def __eq__(self, other: 'SumNode') -> bool:
        """
        Check equivalence between this node and another SumNode.

        Two SumNodes are considered equivalent if they have the same depth
        and their distance is within the distance threshold.

        Parameters
        ----------
        other : SumNode
            Another node to compare with.

        Returns
        -------
        bool
            True if nodes are equivalent (distance ≤ threshold), False otherwise.
        """
        if not isinstance(other, SumNode):
            return False
        if self.depth != other.depth:
            return False
        return self.distance(other) <= self.distance_threshold

    def __repr__(self) -> str:
        """
        Return string representation of the SumNode.

        Returns
        -------
        str
            String representation showing depth and sums.
        """
        return f"SumNode(depth={self.depth}, sums={self.sums})"

    def show(self) -> None:
        """
        Visualise the node.

        This node type does not have a visual representation.
        """
        pass
