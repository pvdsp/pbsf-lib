"""Chain class for wrapping discretisation node sequences."""

from collections.abc import Iterator, Sequence

from pbsf.nodes import Node


class Chain(Sequence):
    """
    A chain of discretisation nodes from coarse to fine granularity.

    Wraps a sequence of nodes. Distance between chains is the distance
    between their most fine-grained (last) nodes.

    Parameters
    ----------
    nodes : list[Node]
        Non-empty list of Node instances, all of the same type.

    Raises
    ------
    ValueError
        If nodes is empty, contains non-Node instances, or contains mixed types.
    """

    def __init__(self, nodes: list[Node]) -> None:
        if not nodes:
            raise ValueError("Chain must contain at least one node.")
        if not all(isinstance(n, Node) for n in nodes):
            raise ValueError("All elements must be Node instances.")
        node_type = type(nodes[0])
        if not all(type(n) is node_type for n in nodes):
            raise ValueError(
                "All nodes must be the same type, got mixed types."
            )
        self._nodes = tuple(nodes)

    @property
    def nodes(self) -> tuple[Node, ...]:
        """The nodes in this chain."""
        return self._nodes

    @property
    def length(self) -> int:
        """Number of nodes in the chain."""
        return len(self._nodes)

    def distance(self, other: 'Chain') -> float:
        """
        Compute distance between this chain and another.

        Returns the distance between the most fine-grained (last) nodes
        of the two chains.

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
            If chains have different lengths or different node types.
        """
        if not isinstance(other, Chain):
            raise ValueError("Can only compute distance to another Chain.")
        if self.length != other.length:
            raise ValueError(
                f"Chains must have the same length,"
                f" got {self.length} and {other.length}."
            )
        if type(self._nodes[0]) is not type(other._nodes[0]):
            raise ValueError("Chains must contain the same node type.")
        return self._nodes[-1].distance(other._nodes[-1])

    def __len__(self) -> int:
        """Return the number of nodes in the chain."""
        return len(self._nodes)

    def __iter__(self) -> Iterator[Node]:
        """Iterate over the nodes in the chain."""
        return iter(self._nodes)

    def __getitem__(self, index):
        """Return the node at the given index, or a new Chain for slices."""
        if isinstance(index, slice):
            return Chain(list(self._nodes[index]))
        return self._nodes[index]

    def __repr__(self) -> str:
        """Return string representation of the chain."""
        node_type = type(self._nodes[0]).__name__
        return f"Chain(length={self.length}, node_type={node_type})"

    def __eq__(self, other: object) -> bool:
        """Check equality based on node contents."""
        if not isinstance(other, Chain):
            return NotImplemented
        return self._nodes == other._nodes

    __hash__ = None
