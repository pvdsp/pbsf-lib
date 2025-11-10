class Digraph:
    """
    A simple directed graph implementation.

    Attributes
    ----------
    vertices : list[dict]
        List of dictionaries storing properties of each vertex.
    edges : list[set[int]]
        List storing sets of outgoing edge vertex IDs for each vertex.
    """
    def __init__(self) -> None:
        self.vertices = []
        self.edges = []

    def add_vertex(self, properties: dict | None = None) -> int:
        """
        Add a new vertex to the graph.

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
        if properties is None:
            properties = {}
        if not isinstance(properties, dict):
            raise ValueError(f"Properties {properties} must be a dictionary.")
        vertex_id = len(self.vertices)  # identifier of the new vertex
        self.vertices.append(properties)
        self.edges.append(set())
        return vertex_id

    def add_edge(self, from_v: int, to_v: int) -> None:
        """
        Add a directed edge between two vertices.

        Parameters
        ----------
        from_v : int
            The ID of the source vertex.
        to_v : int
            The ID of the target vertex.

        Raises
        ------
        ValueError
            If either vertex does not exist.
        """
        if from_v >= len(self.vertices):
            raise ValueError(f"Vertex {from_v} does not exist.")
        if to_v >= len(self.vertices):
            raise ValueError(f"Vertex {to_v} does not exist.")
        self.edges[from_v].add(to_v)

    def outgoing(self, vertex_id: int) -> set[int]:
        """
        Get the outgoing edges of a vertex.

        Parameters
        ----------
        vertex_id : int
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
        if vertex_id >= len(self.vertices):
            raise ValueError(f"Vertex with identifier {vertex_id} does not exist.")
        return self.edges[vertex_id]

    def __repr__(self) -> str:
        """
        Return string representation of the Digraph.

        Returns
        -------
        str
            String representation showing number of vertices and edges.
        """
        return (f"Digraph(vertices={len(self.vertices)}, "
                f"edges={sum([len(self.outgoing(node)) for node in range(len(self.vertices))])})")