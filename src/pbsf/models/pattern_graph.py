import math

from pbsf.models.base import Model
from pbsf.utils.layered_digraph import LayeredDigraph
from pbsf.nodes import Node


class PatternGraph(Model):
    """
    Directed acyclic graph (DAG) for representing coarse- to fine-grained discretised patterns.

    A PatternGraph uses a layered directed graph structure to store chains of nodes,
    where each layer corresponds to a specific depth level. Unlike PatternTree, this
    allows multiple parent-child relationships and more flexible pattern matching across
    different sequences.

    Parameters
    ----------
    params : dict | None, default=None
        Configuration dictionary. Optional keys:

        - closest_match (bool): If True, use best match strategy based on distance;
          if False, use first match strategy. Default is False.

    Attributes
    ----------
    graph : LayeredDigraph
        Layered directed graph that stores the structure of the pattern graph.
    params : dict
        Configuration parameters.
    """

    def __init__(self, params: dict | None = None) -> None:
        self.params = params
        if self.params is None:
            self.params = {}
        self.graph = LayeredDigraph()

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
            if (node == candidate) and ((distance := node.distance(candidate)) < current_distance):
                current_match = vertex
                current_distance = distance
        return current_match

    def _find_matching_vertex(self, node: Node, depth: int, parent: int | None = None) -> int | None:
        """
        Find a vertex in the graph that matches the given node at the specified depth.

        Searches for a matching vertex first among the parent's children (if provided),
        then among all vertices at the specified depth layer.

        Parameters
        ----------
        node : Node
            Node to find a match for.
        depth : int
            Depth level to search in.
        parent : int | None, default=None
            Optional parent vertex identifier to prioritise searching among its children.

        Returns
        -------
        int | None
            Vertex identifier if a match is found, otherwise None.
        """
        if depth >= self.graph.max_depth:
            return None

        # Determine the matching strategy based on parameters:
        closest_match = self.params.get("closest_match", False)
        match_strategy = self._best_match if closest_match else self._first_match

        # If parent provided, first check all children as potential matches:
        if parent is not None:
            candidates = self.graph.outgoing(parent)
            if (match := match_strategy(node, candidates)) is not None:
                return match

        # If no parent provided (or no match found), check all vertices at the current depth:
        candidates = self.graph.get_layer(depth)
        if (match := match_strategy(node, candidates)) is not None:
            return match

        # If no match found, return None:
        return None

    def _check_connection(self, v1: int | None, v2: int | None) -> bool:
        """
        Check if there is an edge from vertex v1 to vertex v2.

        Parameters
        ----------
        v1 : int | None
            Source vertex identifier.
        v2 : int | None
            Target vertex identifier.

        Returns
        -------
        bool
            True if there is an edge from v1 to v2, False otherwise (including
            if either vertex is None).
        """
        if v1 is None or v2 is None:
            return False
        children = self.graph.outgoing(v1)
        return v2 in children

    def chain_to_vertices(self, chain: list[Node]) -> tuple[list[int | None], list[bool]]:
        """
        Convert a chain of nodes to vertex identifiers and their connection status.

        Matches each node in the chain to existing vertices in the graph and tracks
        whether consecutive vertices are connected by edges.

        Parameters
        ----------
        chain : list[Node]
            Chain of nodes to match against the graph.

        Returns
        -------
        tuple[list[int | None], list[bool]]
            A tuple containing:
            - traversal: List of vertex identifiers (or None if no match found)
              corresponding to the nodes in the chain.
            - connection: List of booleans indicating if there is an edge between
              consecutive vertices in the traversal (length is len(chain) - 1).
        """
        traversal = []
        connection = []
        for depth, node in enumerate(chain):
            parent = traversal[-1] if len(traversal) > 0 else None
            matched_vertex_id = self._find_matching_vertex(node, depth, parent)
            if matched_vertex_id is not None:
                if len(traversal) > 0:
                    parent_id = traversal[-1]
                    connection.append(self._check_connection(parent_id, matched_vertex_id))
                traversal.append(matched_vertex_id)
            else:
                if len(traversal) > 0:
                    connection.append(False)
                traversal.append(None)
        return traversal, connection

    def update(self, chain: list[Node]) -> list[int]:
        """
        Update the graph with a new chain of discretised data.

        Reuses existing nodes if they are already in the graph, otherwise adds
        new nodes. Connects unconnected vertices to represent the chain structure.

        Parameters
        ----------
        chain : list[Node]
            A list of nodes representing the chain of discretised data.

        Returns
        -------
        list[int]
            Vertex identifiers of the nodes in the PatternGraph.

        Raises
        ------
        ValueError
            If chain is not a list or contains non-Node elements.
        """
        if not isinstance(chain, list):
            raise ValueError("Chain must be a list.")
        if not all(isinstance(node, Node) for node in chain):
            raise ValueError("Chain must contain only nodes.")

        vertices, connections = self.chain_to_vertices(chain)
        # Create vertices for gaps in the traversal:
        for idx, vertex in enumerate(vertices):
            if vertex is None:
                vertices[idx] = self.graph.add_vertex({
                    "node": chain[idx]
                })
        # Create edges between vertices that represent the chain:
        for idx, connected in enumerate(connections):
            if not connected:
                v1 = vertices[idx]
                v2 = vertices[idx + 1]
                self.graph.add_edge(v1, v2)

        return vertices

    def learn(self, chains: list[list[Node]]) -> list[list[int]]:
        """
        Learn patterns from the provided dataset.

        Processes multiple chains and adds them to the graph, reusing existing
        patterns and creating new connections where possible.

        Parameters
        ----------
        chains : list[list[Node]]
            A list of chains of nodes representing the dataset.

        Returns
        -------
        list[list[int]]
            Vertex identifiers for each chain in the PatternGraph.

        Raises
        ------
        ValueError
            If the input data is not a list, contains elements that are not lists,
            or contains non-Node elements.
        """
        if not isinstance(chains, list):
            raise ValueError("Chains must be a list.")
        if not all(isinstance(chain, list) for chain in chains):
            raise ValueError("Chains must contain only lists of Nodes.")
        if not all(all(isinstance(node, Node) for node in chain) for chain in chains):
            raise ValueError("Chains must contain only nodes.")
        return [self.update(chain) for chain in chains]

    def contains(self, chain: list[Node]) -> bool:
        """
        Check if the graph contains a specific chain of nodes and all its connections.

        Determines whether the entire chain exists in the graph with all edges
        between consecutive nodes properly connected.

        Parameters
        ----------
        chain : list[Node]
            Chain of nodes to check for membership.

        Returns
        -------
        bool
            True if all nodes exist and all consecutive nodes are connected by edges,
            False otherwise.
        """
        vertices, connections = self.chain_to_vertices(chain)
        return all(vertex is not None for vertex in vertices) and all(connections)

    def __repr__(self) -> str:
        """
        Return string representation of the PatternGraph.

        Returns
        -------
        str
            String representation showing the number of vertices and edges in the graph.
        """
        num_edges = sum(len(self.graph.outgoing(v)) for v in self.graph.vertices)
        return f"PatternGraph(vertices={len(self.graph.vertices)}, edges={num_edges})"