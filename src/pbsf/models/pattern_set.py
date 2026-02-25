"""Set-based model for tracking unique patterns at various depths."""

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
    nodes : list[set[Node]]
        List of sets that stores discretisations at various depths. Each set
        contains unique nodes at a specific depth level.
    params : dict
        Configuration parameters.

    Notes
    -----
    Nodes added to the PatternSet must implement `__hash__` and `__eq__` to
    support set membership operations.
    """

    def __init__(self, params: dict | None = None) -> None:
        self.params = params
        if self.params is None:
            self.params = {}
        self.nodes = []

    def update(self, chain: list[Node]) -> list[bool]:
        """
        Update the sets with a new chain of discretised data.

        For each depth level in the chain, checks if the node already exists in
        the corresponding set. If not present, adds the node to the set. Expands
        the depth structure as needed to accommodate the chain.

        Parameters
        ----------
        chain : list[Node]
            A list of nodes representing the chain of discretised data.

        Returns
        -------
        list[bool]
            A list of booleans indicating whether each node was already present
            in its corresponding set (True = was present, False = newly added).
        """
        present = []
        while len(self.nodes) < len(chain):
            self.nodes.append(set())
        for node, node_set in zip(chain, self.nodes):
            was_present = node in node_set
            present.append(was_present)
            if not was_present:
                node_set.add(node)
        return present

    def learn(self, chains: list[list[Node]]) -> list[list[bool]]:
        """
        Learn patterns from the provided dataset.

        Processes multiple chains and adds them to the pattern sets, tracking
        which patterns were previously seen.

        Parameters
        ----------
        chains : list[list[Node]]
            A list of chains of nodes representing the dataset.

        Returns
        -------
        list[list[bool]]
            For each chain, a list of booleans indicating whether each node was
            already present in its corresponding set.

        Raises
        ------
        ValueError
            If the input data is not a list, contains elements that are not lists,
            or contains non-Node elements.
        """
        if not isinstance(chains, list):
            raise ValueError("Data must be a list.")
        if not all(isinstance(chain, list) for chain in chains):
            raise ValueError("Data must contain only lists of Nodes.")
        if not all(all(isinstance(node, Node) for node in chain) for chain in chains):
            raise ValueError("Data must contain only nodes.")
        return [self.update(chain) for chain in chains]

    def contains(self, chain: list[Node]) -> bool:
        """
        Check if the pattern set contains all nodes in a specific chain.

        Determines whether every node in the chain exists in its corresponding
        depth-level set. If the chain is longer than the tracked depths, the
        missing nodes are considered not present.

        Parameters
        ----------
        chain : list[Node]
            Chain of nodes to check for membership.

        Returns
        -------
        bool
            True if all nodes in the chain are present in their corresponding sets,
            False otherwise.
        """
        present = [node in node_set for node, node_set in zip(chain, self.nodes)]
        while len(present) < len(chain):
            present.append(False)
        return all(present)

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
