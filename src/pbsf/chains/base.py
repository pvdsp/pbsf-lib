"""Chain class for wrapping discretisation node sequences."""

from collections.abc import Iterator, Sequence

import numpy as np

from pbsf.nodes import Node


class Chain(Sequence):
    """
    A chain of discretisation nodes from coarse to fine granularity.

    Wraps a sequence of nodes. Distance between chains is the distance
    between their most fine-grained (last) nodes.

    Parameters
    ----------
    nodes : Sequence[Node]
        Optional sequence of Node instances, all of the same type.

    Raises
    ------
    ValueError
        If nodes contain non-Node instances, or contains mixed types.
    """

    def __init__(self, nodes: Sequence[Node] | None = None) -> None:
        if nodes is None:
            nodes = []
        if not all(isinstance(n, Node) for n in nodes):
            raise ValueError("All elements must be Node instances.")
        if len(nodes) > 0:
            self.type = type(nodes[0])
            if not all(type(n) is self.type for n in nodes):
                raise ValueError(
                    "All nodes must be the same type, got mixed types."
                )
        else:
            self.type = Node
        self._nodes = tuple(nodes)

    @property
    def nodes(self) -> tuple[Node, ...]:
        """The nodes in this chain."""
        return self._nodes

    def distance(self, other: 'Chain') -> float:
        """
        Compute distance between this chain and another chain.

        Returns the distance between the most fine-grained compatible nodes
        of the two chains.
        Empty chains are equal, and non-empty chains have
        infinite distance to empty chains.

        Raises ValueError if other is not a Chain or has different node type.

        Parameters
        ----------
        other : Chain
            Chain to compare against.

        Returns
        -------
        float
            Distance value.

        Raises
        ------
        ValueError
            If other is not a Chain or has different node type.
        """
        if not isinstance(other, Chain):
            raise ValueError("Can only compute distance to another Chain.")
        if self.empty and other.empty:
            return 0.0
        if self.empty or other.empty:
            return np.inf
        if self.type != other.type:
            expected = self.type.__name__
            got = other.type.__name__
            raise ValueError(f"Chains must contain the same node type. "
                             f"Expected {expected}, received {got} instead.")
        level = min(len(self), len(other)) - 1
        return self.nodes[level].distance(other.nodes[level])

    @property
    def empty(self) -> bool:
        """Check if Chain is empty."""
        return len(self.nodes) == 0

    def __len__(self) -> int:
        """Return the number of nodes in the chain."""
        return len(self.nodes)

    def __iter__(self) -> Iterator[Node]:
        """Iterate over the nodes in the chain."""
        return iter(self.nodes)

    def __getitem__(self, index):
        """Return the node at the given index, or a new Chain for slices."""
        if isinstance(index, slice):
            return Chain(list(self.nodes[index]))
        return self.nodes[index]

    def __repr__(self) -> str:
        """Return string representation of the chain."""
        return f"Chain(length={len(self)}, node_type={self.type.__name__})"

    def __eq__(self, other: object) -> bool:
        """Check equality based on node contents."""
        if not isinstance(other, Chain):
            return NotImplemented
        return self.nodes == other.nodes
