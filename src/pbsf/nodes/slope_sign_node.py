from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pbsf.nodes.base import Node
from pbsf.utils import has_required


class SlopeSignNode(Node):
    """
    Node representing slope signs of a segment discretisation.

    This node uses the signs of slopes (positive/negative) for comparison.
    Nodes are considered equivalent if all slope signs match.

    Parameters
    ----------
    properties : dict[str, Any]
        Configuration dictionary with the following required keys:

        - depth (int): Depth of the node in the chain.
        - slopes (np.ndarray): Array of slopes of the linear segments.
        - intercepts (np.ndarray): Array of intercepts of the linear segments.
        - breakpoints (list): List of (start, end) tuples defining the segments.
    """
    def __init__(self, properties: dict[str, Any]) -> None:
        has_required(properties, [
            ("depth", int),
            ("slopes", np.ndarray),
            ("intercepts", np.ndarray),
            ("breakpoints", list)
        ])
        self.depth = properties["depth"]
        self.slopes = properties["slopes"]
        self.intercepts = properties["intercepts"]
        self.breakpoints = properties["breakpoints"]

    def show(self) -> None:
        """
        Visualise the slope signs with colour-coded segments.

        Draws vertical lines at breakpoints and plots linear segments with
        colour indicating slope direction: green for positive/zero slopes,
        crimson for negative slopes. Fills the area between the line and zero.
        """
        for (x1, x2), (a, b) in zip(self.breakpoints, zip(self.slopes, self.intercepts)):
            plt.axvline(x1, color="lightgrey", linestyle=":")
            plt.axvline(x2, color="lightgrey", linestyle=":")
            x = np.linspace(0, x2 - x1, 100)
            color = "green" if a >= 0 else "crimson"
            plt.plot(np.linspace(x1, x2, 100),
                     a * x + b, color=color, linestyle="--")
            plt.fill_between(
                x=np.linspace(x1, x2, 100), y1=0, y2=a * x + b,
                color=color, alpha=0.5
            )

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
            If node is not a SlopeSignNode or has a different depth.
        """
        if not isinstance(node, SlopeSignNode):
            raise ValueError(f"Cannot compare node of type {type(self)} with {type(node)}.")
        if self.depth != node.depth:
            raise ValueError("Cannot compare nodes of different depths.")

    def distance(self, node: 'SlopeSignNode') -> float:
        """
        Calculate the proportion of differing slope signs between nodes.

        Parameters
        ----------
        node : SlopeSignNode
            Another SlopeSignNode to calculate distance to.

        Returns
        -------
        float
            Proportion of slope signs that differ (0.0 = all match, 1.0 = all differ).

        Raises
        ------
        ValueError
            If nodes are not comparable (different types or depths).
        """
        self._is_comparable(node)
        s1 = (self.slopes >= 0)
        s2 = (node.slopes >= 0)
        return float(np.sum(s1 != s2) / len(s1))

    def __eq__(self, node: 'SlopeSignNode') -> bool:
        """
        Check equivalence between this node and another SlopeSignNode.

        Two SlopeSignNodes are considered equivalent if they have the same depth
        and all slope signs match (distance = 0.0).

        Parameters
        ----------
        node : SlopeSignNode
            Another node to compare with.

        Returns
        -------
        bool
            True if nodes are equivalent (same depth and matching slope signs), False otherwise.
        """
        if not isinstance(node, SlopeSignNode):
            return False
        if self.depth != node.depth:
            return False
        return self.distance(node) == 0.0

    def __repr__(self) -> str:
        """
        Return string representation of the SlopeSignNode.

        Returns
        -------
        str
            String representation showing depth and slope signs ('+' or '-').
        """
        return f"SlopeSignNode(depth={self.depth}, slopes={['+' if s >= 0 else '-' for s in self.slopes]})"

    def __hash__(self) -> int:
        """
        Return hash of the node based on slope signs.

        Returns
        -------
        int
            Hash value computed from the tuple of slope signs (1 for positive, -1 for negative).
        """
        return tuple(1 if slope >= 0 else -1 for slope in self.slopes).__hash__()
