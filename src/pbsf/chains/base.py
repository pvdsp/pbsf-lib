"""Chain class for wrapping discretisation node sequences."""

from collections.abc import Callable, Iterator, Sequence

from pbsf.nodes import Node


class Chain(Sequence):
    """
    A chain of discretisation nodes from coarse to fine granularity.

    Wraps a sequence of nodes and provides chain-level distance computation
    with configurable distance strategies.

    Parameters
    ----------
    nodes : list[Node]
        Non-empty list of Node instances, all of the same type.
    distance_fn : Callable | None, default=None
        Custom distance function taking two Chain instances and returning a float.
        Defaults to exponential weighted distance.

    Raises
    ------
    ValueError
        If nodes is empty, contains non-Node instances, or contains mixed types.
    """

    def __init__(
        self,
        nodes: list[Node],
        distance_fn: Callable[['Chain', 'Chain'], float] | None = None,
    ) -> None:
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
        self._distance_fn = distance_fn or self._weighted_distance

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
        if self._distance_fn is not other._distance_fn:
            raise ValueError("Chains must use the same distance function.")
        return self._distance_fn(self, other)

    @staticmethod
    def _weighted_distance(a: 'Chain', b: 'Chain') -> float:
        """
        Exponential weighted distance: coarser levels weigh more.

        Weight for depth i = 2^(length - 1 - i), normalised by sum of weights.
        """
        total_weight = 0.0
        weighted_sum = 0.0
        for i in range(a.length):
            w = 2 ** (a.length - 1 - i)
            weighted_sum += w * a._nodes[i].distance(b._nodes[i])
            total_weight += w
        return weighted_sum / total_weight

    @staticmethod
    def _mean_distance(a: 'Chain', b: 'Chain') -> float:
        """Unweighted mean distance across all levels."""
        return sum(
            a._nodes[i].distance(b._nodes[i]) for i in range(a.length)
        ) / a.length

    def __len__(self) -> int:
        """Return the number of nodes in the chain."""
        return len(self._nodes)

    def __iter__(self) -> Iterator[Node]:
        """Iterate over the nodes in the chain."""
        return iter(self._nodes)

    def __getitem__(self, index):
        """Return the node at the given index, or a new Chain for slices."""
        if isinstance(index, slice):
            return Chain(list(self._nodes[index]), self._distance_fn)
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
