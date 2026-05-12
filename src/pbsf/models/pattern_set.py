"""Set-based model for tracking unique patterns at various depths."""

from collections.abc import Sequence

from bidict import bidict

from pbsf.models.base import Model
from pbsf.nodes import Node


class PatternSet(Model):
    """
    Track unique patterns at various depths using sets of discretised nodes.

    PatternSet maintains separate sets for each depth level, allowing efficient
    membership checking for patterns at different granularities. Nodes must be
    hashable for set operations.

    Parameters
    ----------
    params : dict | None, default=None
        Configuration dictionary (currently unused, reserved for future extensions).

    Attributes
    ----------
    ids : dict
        Integer identifiers for discretisations.
    nodes : list[set[int]]
        List of sets that stores discretisation identifiers of various depths.
    params : dict
        Configuration parameters.

    Notes
    -----
    Nodes added to the PatternSet must implement `__hash__` and `__eq__` to
    support set membership operations.
    """

    def __init__(self, params: dict | None = None) -> None:
        self.__next_free = 0
        self.params = params
        if self.params is None:
            self.params = {}
        self.ids = bidict()
        self.nodes = []

    def update(self, chain: Sequence[Node]) -> list[bool]:
        """
        Update the sets with a new chain of discretised data.

        For each depth level in the chain, checks if the node already exists in
        the corresponding set. If not present, adds the node to the set. Expands
        the depth structure as needed to accommodate the chain.

        Parameters
        ----------
        chain : Sequence[Node]
            A sequence of nodes representing the chain of discretised data.

        Returns
        -------
        list[bool]
            A list of booleans indicating whether each node was already present
            in its corresponding set (True = was present, False = newly added).
        """
        present = []
        while len(self.nodes) < len(chain):
            self.nodes.append(set())
        for level, node in enumerate(chain):
            was_present = node in self.ids
            present.append(was_present)
            if not was_present:
                self.ids[node] = self.__next_free
                self.nodes[level].add(self.__next_free)
                self.__next_free += 1
        return present

    def learn(self, chains: Sequence[Sequence[Node]]) -> list[list[bool]]:
        """
        Learn patterns from the provided dataset.

        Processes multiple chains and adds them to the pattern sets, tracking
        which patterns were previously seen.

        Parameters
        ----------
        chains : Sequence[Sequence[Node]]
            A sequence of chains of nodes representing the dataset.

        Returns
        -------
        list[list[bool]]
            For each chain, a list of booleans indicating whether each node was
            already present in its corresponding set.

        Raises
        ------
        ValueError
            If the input data is not a sequence, contains elements that are not
            sequences, or contains non-Node elements.
        """
        if not isinstance(chains, Sequence):
            raise ValueError("Data must be a sequence.")
        if not all(isinstance(chain, Sequence) for chain in chains):
            raise ValueError("Data must contain only sequences of Nodes.")
        if not all(all(isinstance(node, Node) for node in chain) for chain in chains):
            raise ValueError("Data must contain only nodes.")
        return [self.update(chain) for chain in chains]

    def contains(self, chain: Sequence[Node]) -> bool:
        """
        Check if the pattern set contains all nodes in a specific chain.

        Determines whether every node in the chain exists in its corresponding
        depth-level set. If the chain is longer than the tracked depths, the
        missing nodes are considered not present.

        Parameters
        ----------
        chain : Sequence[Node]
            Chain of nodes to check for membership.

        Returns
        -------
        bool
            True if all nodes in the chain are present in their corresponding sets,
            False otherwise.
        """
        for level, node in enumerate(chain):
            if level >= len(self.nodes):
                return False
            if node not in self.ids:
                return False
            if self.ids[node] not in self.nodes[level]:
                return False
        return True

    def get_node(self, identifier: int) -> Node:
        """Get the node for the given identifier."""
        if identifier not in self.ids.inverse:
            raise KeyError(f"Unknown identifier: {identifier}")
        return self.ids.inverse[identifier]

    def get_level(self, level: int) -> set[int]:
        """Get all identifiers at the given depth level."""
        if level < 0:
            raise ValueError("Level should be positive or zero.")
        if level >= len(self.nodes):
            return set()
        return set(self.nodes[level])

    def get_related(self, identifier: int, level: int) -> set[int]:
        """Get related identifiers at the given level for a vertex.

        Returns {identifier} when the requested level matches the
        identifier's own level. Otherwise returns an empty set, as
        relations between granularity levels are not stored in a
        PatternSet.
        """
        for depth, nodes in enumerate(self.nodes):
            if identifier in nodes:
                if level == depth:
                    return {identifier}
                return set()
        raise KeyError(f"Unknown identifier: {identifier}")

    def __repr__(self) -> str:
        """
        Return string representation of the PatternSet.

        Returns
        -------
        str
            String representation showing the number of depth levels and total
            unique patterns.
        """
        total_patterns = sum(len(node_set) for node_set in self.nodes)
        return f"PatternSet(depths={len(self.nodes)}, patterns={total_patterns})"
