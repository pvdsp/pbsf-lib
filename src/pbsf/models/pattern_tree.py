"""Tree-based model for storing coarse-to-fine discretised patterns."""

from collections.abc import Sequence

from pbsf.chains import Chain
from pbsf.models.base import Model
from pbsf.nodes import Node
from pbsf.utils.digraph import Digraph


class PatternTree(Model):
    """
    Rooted tree for coarse-to-fine discretised patterns.

    A PatternTree stores chains of nodes in a directed graph structure, enabling
    efficient pattern matching and learning from discretised sequences. The tree
    grows incrementally as new patterns are learned, reusing existing nodes when
    patterns overlap.

    Parameters
    ----------
    params : dict | None, default=None
        Configuration dictionary. Currently not in use.

    Attributes
    ----------
    graph : Digraph
        Directed graph that stores the structure of the tree.
    root : int
        Identifier of the root node of the tree.
    params : dict
        Configuration parameters.
    """

    def __init__(self, params: dict | None = None) -> None:
        self.params = params
        if self.params is None:
            self.params = {}
        self.graph = Digraph()
        self.root = self.graph.add_vertex({
            "node": "root",
            "depth": -1
        })

    def __vertices_to_chain(self, vertices: Sequence[int]) -> Chain:
        """Convert a sequence of vertex identifiers to a chain of nodes."""
        if vertices and vertices[0] == self.root:
            vertices = vertices[1:]
        return Chain([self.get_node(idx) for idx in vertices])

    def __closest_path(self, paths, chain) -> tuple[int, ...]:
        """Return path closest to provided chain."""
        chains = [(path, self.__vertices_to_chain(path)) for path in paths]
        vertices, _ = min(chains, key=lambda c: chain.distance(c[1]))
        return vertices

    def chain_to_vertices(self, chain: Sequence[Node]) -> tuple[int, ...]:
        """
        Convert a chain of nodes to existing vertex identifiers.

        Returns the path corresponding to the closest Chain in the tree.

        Parameters
        ----------
        chain : Sequence[Node]
            Chain of nodes to match against the tree.

        Returns
        -------
        tuple[int, ...]
            Tuple of vertex identifiers representing the matched path, starting with
            the root. Length will be len(chain) + 1 if all nodes match, or shorter
            if matching stops early.
        """
        if not isinstance(chain, Chain):
            chain = Chain(chain)

        paths = set([(self.root,)])
        for node in chain:
            new_paths = set()
            for path in paths:
                neighbours = self.graph.outgoing(path[-1])
                for neighbour in neighbours:
                    candidate = self.graph.vertices[neighbour]
                    if node == candidate["node"]:
                        new_paths.add(path + (neighbour,))
            if len(new_paths) > 0:
                paths = new_paths
            else:
                return self.__closest_path(paths, chain)

        return self.__closest_path(paths, chain)

    def update(self, chain: Sequence[Node]) -> list[int]:
        """
        Update the tree with a new chain of discretised data.

        Reuses existing nodes if they are already in the tree, otherwise adds new
        nodes. Returns the vertex identifiers of the nodes in the PatternTree.

        Parameters
        ----------
        chain : Sequence[Node]
            A sequence of nodes representing the chain of discretised data.

        Returns
        -------
        list[int]
            Vertex identifiers of the nodes in the PatternTree, including the root.

        Raises
        ------
        ValueError
            If chain is not a sequence or contains non-Node elements.
        """
        if not isinstance(chain, Sequence):
            raise ValueError("Chain must be a sequence.")
        if not all(isinstance(node, Node) for node in chain):
            raise ValueError("Chain must contain only nodes.")

        vertices = list(self.chain_to_vertices(chain))
        while len(vertices) <= len(chain):
            current_vertex = vertices[-1]
            next_vertex = self.graph.add_vertex({
                "node": chain[len(vertices) - 1],
                "depth": chain[len(vertices) - 1].depth
            })
            self.graph.add_edge(current_vertex, next_vertex)
            vertices.append(next_vertex)
        return vertices

    def learn(self, chains: Sequence[Sequence[Node]]) -> list[list[int]]:
        """
        Learn patterns from the provided dataset.

        Processes multiple chains and adds them to the tree, reusing existing
        patterns where possible.

        Parameters
        ----------
        chains : Sequence[Sequence[Node]]
            A sequence of chains of nodes representing the dataset.

        Returns
        -------
        list[list[int]]
            Vertex identifiers for each chain in the PatternTree.

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
        if not all(
            all(isinstance(node, Node) for node in chain)
            for chain in chains
        ):
            raise ValueError("Data must contain only nodes.")
        return [self.update(chain) for chain in chains]

    def contains(self, chain: Sequence[Node]) -> bool:
        """
        Check if the tree contains a specific chain of nodes.

        Determines whether the entire chain can be matched in the tree by
        traversing from the root.

        Parameters
        ----------
        chain : Sequence[Node]
            Chain of nodes to check for membership.

        Returns
        -------
        bool
            True if the entire chain exists in the tree, False otherwise.
        """
        vertices = self.chain_to_vertices(chain)
        return len(vertices) == len(chain) + 1

    def get_node(self, identifier: int) -> Node:
        """Get the node for the given vertex identifier."""
        if identifier < 0 or identifier >= len(self.graph.vertices):
            raise KeyError(f"Unknown identifier: {identifier}")
        if identifier == self.root:
            raise KeyError(f"Identifier {identifier} refers to the root.")
        return self.graph.vertices[identifier]["node"]

    def get_level(self, level: int) -> set[int]:
        """Get all vertex identifiers at the given depth level."""
        if level < 0:
            raise ValueError("Level should be positive or zero.")
        return {
            v for v in range(len(self.graph.vertices))
            if self.graph.vertices[v]["depth"] == level
        }

    def get_related(self, identifier: int, level: int) -> set[int]:
        """Get related vertex identifiers at the given level for a vertex."""
        if identifier < 0 or identifier >= len(self.graph.vertices):
            raise KeyError(f"Unknown identifier: {identifier}")
        if identifier == self.root:
            raise KeyError(f"Identifier {identifier} refers to the root.")
        depth = self.graph.vertices[identifier]["depth"]
        if level < depth:
            msg = "Level should be equal to or deeper than the node's depth."
            raise ValueError(msg)
        return self._get_descendants(identifier, level)

    def _get_descendants(self, identifier: int, level: int) -> set[int]:
        """Recursively collect descendants at the given depth level."""
        if self.graph.vertices[identifier]["depth"] == level:
            return {identifier}
        result = set()
        for child in self.graph.outgoing(identifier):
            result |= self._get_descendants(child, level)
        return result

    def __repr__(self) -> str:
        """
        Return string representation of the PatternTree.

        Returns
        -------
        str
            String representation showing the number of vertices in the tree.
        """
        return f"PatternTree(vertices={len(self.graph.vertices)})"
