import math

from pbsf.models.base import Model
from pbsf.nodes import Node
from pbsf.utils.digraph import Digraph


class PatternTree(Model):
    """
    Rooted tree-like structure for representing coarse- to
    fine-grained discretised data.

    A PatternTree stores chains of nodes in a directed graph structure, enabling
    efficient pattern matching and learning from discretised sequences. The tree
    grows incrementally as new patterns are learned, reusing existing nodes when
    patterns overlap.

    Parameters
    ----------
    params : dict | None, default=None
        Configuration dictionary. Optional keys:

        - closest_match (bool): If True, use best match strategy based on distance;
          if False, use first match strategy. Default is False.

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

    def _first_match(self, node: Node, candidates: set[int]) -> int | None:
        """
        Find the first matching node among candidates.

        Returns the first vertex from candidates whose node is equivalent to the
        given node, using the node's equivalence operator.

        Parameters
        ----------
        node : Node
            Node to match against candidates.
        candidates : set[int]
            Set of vertex identifiers to search through.

        Returns
        -------
        int | None
            Vertex identifier of the first match, or None if no match found.
        """
        for vertex in candidates:
            candidate = self.graph.vertices[vertex]["node"]
            if node == candidate:
                return vertex
        return None

    def _best_match(self, node: Node, candidates: set[int]) -> int | None:
        """
        Find the closest matching node among candidates.

        Returns the vertex from candidates whose node is equivalent to the given
        node and has the smallest distance, according to the node's distance metric.

        Parameters
        ----------
        node : Node
            Node to match against candidates.
        candidates : set[int]
            Set of vertex identifiers to search through.

        Returns
        -------
        int | None
            Vertex identifier of the best match, or None if no match found.
        """
        current_match = None
        current_distance = math.inf
        for vertex in candidates:
            candidate = self.graph.vertices[vertex]["node"]
            distance = node.distance(candidate)
            if node == candidate and distance < current_distance:
                current_match = vertex
                current_distance = distance
        return current_match

    def chain_to_vertices(self, chain: list[Node]) -> list[int]:
        """
        Convert a chain of nodes to a list of existing vertex
        identifiers in the PatternTree.

        Traverses the tree from the root, matching each node in the chain to an
        existing vertex. At the first mismatch, the traversal stops and returns
        the partial path. The match strategy (first match or best match) is determined
        by the 'closest_match' parameter.

        Parameters
        ----------
        chain : list[Node]
            Chain of nodes to match against the tree.

        Returns
        -------
        list[int]
            List of vertex identifiers representing the matched path, starting with
            the root. Length will be len(chain) + 1 if all nodes match, or shorter
            if matching stops early.
        """
        traversal = [self.root]
        closest_match = self.params.get("closest_match", False)
        match_strategy = self._best_match if closest_match else self._first_match
        for node in chain:
            neighbours = self.graph.outgoing(traversal[-1])
            candidate = match_strategy(node, neighbours)
            if candidate is None:
                break
            traversal.append(candidate)
        return traversal

    def update(self, chain: list[Node]) -> list[int]:
        """
        Update the tree with a new chain of discretised data.

        Reuses existing nodes if they are already in the tree, otherwise adds new
        nodes. Returns the vertex identifiers of the nodes in the PatternTree.

        Parameters
        ----------
        chain : list[Node]
            A list of nodes representing the chain of discretised data.

        Returns
        -------
        list[int]
            Vertex identifiers of the nodes in the PatternTree, including the root.

        Raises
        ------
        ValueError
            If chain is not a list or contains non-Node elements.
        """
        if not isinstance(chain, list):
            raise ValueError("Chain must be a list.")
        if not all(isinstance(node, Node) for node in chain):
            raise ValueError("Chain must contain only nodes.")

        vertices = self.chain_to_vertices(chain)
        while len(vertices) <= len(chain):
            current_vertex = vertices[-1]
            next_vertex = self.graph.add_vertex({
                "node": chain[len(vertices) - 1],
                "depth": chain[len(vertices) - 1].depth
            })
            self.graph.add_edge(current_vertex, next_vertex)
            vertices.append(next_vertex)
        return vertices

    def learn(self, chains: list[list[Node]]) -> list[list[int]]:
        """
        Learn patterns from the provided dataset.

        Processes multiple chains and adds them to the tree, reusing existing
        patterns where possible.

        Parameters
        ----------
        chains : list[list[Node]]
            A list of chains of nodes representing the dataset.

        Returns
        -------
        list[list[int]]
            Vertex identifiers for each chain in the PatternTree.

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
        if not all(
            all(isinstance(node, Node) for node in chain)
            for chain in chains
        ):
            raise ValueError("Data must contain only nodes.")
        return [self.update(chain) for chain in chains]

    def contains(self, chain: list[Node]) -> bool:
        """
        Check if the tree contains a specific chain of nodes.

        Determines whether the entire chain can be matched in the tree by
        traversing from the root.

        Parameters
        ----------
        chain : list[Node]
            Chain of nodes to check for membership.

        Returns
        -------
        bool
            True if the entire chain exists in the tree, False otherwise.
        """
        vertices = self.chain_to_vertices(chain)
        return len(vertices) == len(chain) + 1

    def __repr__(self) -> str:
        """
        Return string representation of the PatternTree.

        Returns
        -------
        str
            String representation showing the number of vertices in the tree.
        """
        return f"PatternTree(vertices={len(self.graph.vertices)})"
