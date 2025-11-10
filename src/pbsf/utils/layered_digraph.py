from pbsf.utils.digraph import Digraph


class LayeredDigraph(Digraph):
    """
    A directed graph that enforces a layered structure.

    Vertices of layer n can only point to vertices in layer n+1, creating a
    directed acyclic graph with explicit depth levels.

    Attributes
    ----------
    vertices : list[dict]
        List of dictionaries storing properties of each vertex.
    edges : list[set[int]]
        List storing sets of outgoing edge vertex IDs for each vertex.
    max_depth : int
        Maximum depth (number of layers) in the graph.
    """
    def __init__(self) -> None:
        super().__init__()
        self._layers = [set()]  # Initialize with the root layer
        self.max_depth = len(self._layers)

    def _update_layer(self, vertex_id: int, layer: int) -> None:
        """
        Update the layer assignment for a vertex.

        Removes the vertex from its current layer (if assigned) and adds it to
        the specified layer. Creates new layers as needed and updates max_depth.

        Parameters
        ----------
        vertex_id : int
            The ID of the vertex to update.
        layer : int
            The layer to assign the vertex to.
        """
        vertex = self.vertices[vertex_id]
        if vertex.get("layer") is not None:
            self._layers[vertex["layer"]].remove(vertex_id)
        if len(self._layers) <= layer:
            self._layers.append(set())
            self.max_depth += 1
        self._layers[layer].add(vertex_id)
        vertex["layer"] = layer

    def add_vertex(self, properties: dict | None = None) -> int:
        """
        Add a new vertex to the graph and update the layer mapping.

        New vertices are initially placed in the root layer (layer 0) and may be
        moved to deeper layers when edges are added.

        Parameters
        ----------
        properties : dict | None, default=None
            Dictionary of properties for the new vertex. Empty dict if None.

        Returns
        -------
        int
            The identifier of the new vertex.

        Raises
        ------
        ValueError
            If properties is not a dictionary.
        """
        vertex_id = super().add_vertex(properties)
        self._update_layer(vertex_id, 0)  # New vertices start in the root layer
        return vertex_id

    def add_edge(self, from_v: int, to_v: int) -> None:
        """
        Add a directed edge between two vertices and update the layer mapping.

        Enforces layered structure: edges can only go from layer n to layer n+1.
        Automatically updates the target vertex's layer if needed.

        Parameters
        ----------
        from_v : int
            The ID of the source vertex.
        to_v : int
            The ID of the target vertex.

        Raises
        ------
        ValueError
            If the edge would violate the layered structure constraint.
        """
        v1 = self.vertices[from_v]
        v2 = self.vertices[to_v]
        if v2["layer"] == 0:
            if len(self.outgoing(to_v)) > 0:
                raise ValueError(f"Cannot add edge from node {from_v} in layer {v1['layer']} "
                                 f"to root layer node {to_v} with outgoing edges.")
            else:
                super().add_edge(from_v, to_v)
        elif v2["layer"] == v1["layer"] + 1:
            super().add_edge(from_v, to_v)
        else:
            raise ValueError(f"Cannot add edge from node {from_v} of layer {v1['layer']} "
                                f"to node {to_v} of layer {v2['layer']}.")
        self._update_layer(to_v, v1["layer"] + 1)

    def outgoing(self, vertex: int) -> set[int]:
        """
        Get the outgoing edges of a vertex.

        Parameters
        ----------
        vertex : int
            The ID of the vertex.

        Returns
        -------
        set[int]
            The IDs of the vertices that the input vertex points to.

        Raises
        ------
        ValueError
            If the vertex does not exist.
        """
        return super().outgoing(vertex)

    def get_layer(self, layer: int) -> set[int]:
        """
        Get the set of vertex IDs at a specific layer.

        Parameters
        ----------
        layer : int
            The layer number.

        Returns
        -------
        set[int]
            The IDs of the vertices at the specified layer.

        Raises
        ------
        IndexError
            If the layer does not exist.
        """
        if layer >= len(self._layers):
            raise IndexError(f"Layer {layer} does not exist.")
        return self._layers[layer]

    def __repr__(self) -> str:
        """
        Return string representation of the LayeredDigraph.

        Returns
        -------
        str
            String representation showing number of vertices, edges, and layers.
        """
        num_edges = sum([len(self.outgoing(node)) for node in range(len(self.vertices))])
        return f"LayeredDigraph(vertices={len(self.vertices)}, edges={num_edges}, layers={self.max_depth})"