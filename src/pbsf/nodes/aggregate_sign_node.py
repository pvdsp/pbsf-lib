"""Aggregate sign node for comparing mean value signs."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pbsf.nodes.base import Node
from pbsf.utils import has_required


class AggregateSignNode(Node):
    """
    Node representing mean signs of a segment discretisation.

    This node uses the signs of the means (positive/negative) for comparison.
    Nodes are considered equivalent if all mean signs match.

    Parameters
    ----------
    properties : dict[str, Any]
        Configuration dictionary with the following required keys:

        - depth (int): Depth of the node in the chain.
        - paa (np.ndarray): Array of means of the frames.
        - breakpoints (list): List of (start, end) tuples defining the segments.
    """

    def __init__(self, properties: dict[str, Any]) -> None:
        has_required(properties, [
            ("depth", int),
            ("paa", np.ndarray),
            ("breakpoints", list)
        ])
        self.depth = properties["depth"]
        self.paa = properties["paa"]
        self.breakpoints = properties["breakpoints"]

    def show(self) -> None:
        """
        Visualise the mean signs with colour-coded segments.

        Draws horizontal lines for the means of the frames:
        green for positive/zero means, crimson for negative means.
        Fills the area between the line and zero.
        """
        for (x1, x2), mean in zip(self.breakpoints, self.paa):
            plt.axvline(x1, color="lightgrey", linestyle=":")
            plt.axvline(x2, color="lightgrey", linestyle=":")
            color = "green" if mean >= 0.0 else "crimson"
            plt.hlines(y=mean, xmin=x1, xmax=x2, color=color, linestyle="--")
            plt.fill_between(
                x=np.linspace(x1, x2, 100), y1=0, y2=mean,
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
            If node is not an AggregateSignNode or has a different depth.
        """
        if not isinstance(node, AggregateSignNode):
            raise ValueError(
                f"Cannot compare node of type {type(self)}"
                f" with {type(node)}."
            )
        if self.depth != node.depth:
            raise ValueError("Cannot compare nodes of different depths.")

    def distance(self, node: 'AggregateSignNode') -> float:
        """
        Calculate the proportion of differing mean signs between nodes.

        Parameters
        ----------
        node : AggregateSignNode
            Another AggregateSignNode to calculate distance to.

        Returns
        -------
        float
            Proportion of mean signs that differ.

        Raises
        ------
        ValueError
            If nodes are not comparable (different types or depths).
        """
        self._is_comparable(node)
        s1 = (self.paa >= 0)
        s2 = (node.paa >= 0)
        return float(np.sum(s1 != s2) / len(s1))

    def __eq__(self, node: 'AggregateSignNode') -> bool:
        """
        Check equivalence between this node and another AggregateSignNode.

        Two AggregateSignNodes are considered equivalent if they have the same depth
        and all mean signs match.

        Parameters
        ----------
        node : AggregateSignNode
            Another node to compare with.

        Returns
        -------
        bool
            True if nodes are equivalent, False otherwise.
        """
        if not isinstance(node, AggregateSignNode):
            return False
        if self.depth != node.depth:
            return False
        return self.distance(node) == 0.0

    def __repr__(self) -> str:
        """
        Return string representation of the AggregateSignNode.

        Returns
        -------
        str
            String representation showing depth and mean signs ('+' or '-').
        """
        signs = ['+' if s >= 0 else '-' for s in self.paa]
        return f"AggregateSignNode(depth={self.depth}, paa={signs})"

    def __hash__(self) -> int:
        """
        Return hash of the node based on mean signs.

        Zero means are treated as positive, matching `distance`,
        `__eq__` and `__repr__`.

        Returns
        -------
        int
            Hash value computed from the tuple of mean signs
            (1 for positive/zero, -1 for negative).
        """
        return hash(
            tuple(1 if s >= 0 else -1 for s in self.paa)
        )
