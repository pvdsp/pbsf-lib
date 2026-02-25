"""Graphviz-based visualisation for nested words and pattern models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import graphviz

from pbsf.utils.digraph import Digraph
from pbsf.utils.nested_word import NestedWord

if TYPE_CHECKING:
    from pbsf.models import PatternGraph, PatternTree


def _apply_styling(graph: graphviz.Digraph, label: str = "") -> None:
    """
    Apply consistent styling to the graph.

    Parameters
    ----------
    graph : graphviz.Digraph
        The graph to apply styling to.
    label : str, default=""
        Optional label for the graph.
    """
    graph.graph_attr.update({
        "fontname": "monospace",
        "fontsize": "10",
        "color": "gray80",
        "style": "dotted",
        "label": label,
        "labelloc": "t",
        "labeljust": "l"
    })
    graph.node_attr["fontname"] = "monospace"
    graph.edge_attr["fontname"] = "monospace"

def _show_nested_word(nw: NestedWord) -> graphviz.Digraph:
    """
    Visualise a nested word.

    Parameters
    ----------
    nw : NestedWord
        The nested word to visualise.

    Returns
    -------
    graphviz.Digraph
        A graphviz Digraph representing the nested word.
    """
    # Create a new graph
    graph = graphviz.Digraph()
    graph.edge_attr["arrowhead"] = "normal"

    # Set graph styling attributes
    _apply_styling(graph, label=f"Nested Word: {nw.to_tagged()}")
    graph.graph_attr["rankdir"] = "LR"
    graph.node_attr["shape"] = "circle"
    graph.node_attr["style"] = "filled"
    graph.node_attr["fillcolor"] = "black"
    graph.node_attr["fontcolor"] = "white"
    graph.node_attr["rank"] = "same"

    # Add nodes for each symbol in the word, with edges connecting them
    for i, symbol in enumerate(nw.word):
        c = "black" if not nw.matching.is_internal(i) else "gray"
        graph.node(str(i), label=f"{symbol}", fillcolor=c)
    for i in range(len(nw.word) - 1):
        graph.edge(str(i), str(i + 1))

    # Add dashed edges for call and return positions
    for call, ret in nw.matching.get_matches():
        if call is None:
            graph.node("pending call", label="", style="invis")
        if ret is None:
            graph.node("pending ret", label="", style="invis")
        call = call if call is not None else "pending call"
        ret = ret if ret is not None else "pending ret"
        graph.edge(str(call), str(ret), style="dashed", arrowhead="empty")

    return graph

def _show_digraph(digraph: Digraph) -> graphviz.Digraph:
    """
    Visualise a directed graph.

    Parameters
    ----------
    digraph : Digraph
        The directed graph to visualise.

    Returns
    -------
    graphviz.Digraph
        A graphviz Digraph representing the directed graph.
    """
    graph = graphviz.Digraph()
    graph.graph_attr["rankdir"] = "LR"

    # Apply styling
    _apply_styling(graph, label="Directed Graph")
    graph.node_attr["shape"] = "circle"
    graph.edge_attr["arrowhead"] = "normal"

    # Add vertex nodes
    for vertex_id in range(len(digraph.vertices)):
        label = str(vertex_id)
        graph.node(str(vertex_id), label=label)

    # Add edges
    for vertex_id in range(len(digraph.vertices)):
        for target_vertex in digraph.outgoing(vertex_id):
            graph.edge(str(vertex_id), str(target_vertex))

    return graph

def show(obj: NestedWord | 'PatternTree' | 'PatternGraph') -> graphviz.Digraph:
    """
    Visualise a nested word, pattern tree, or pattern graph.

    Parameters
    ----------
    obj : NestedWord | PatternTree | PatternGraph
        The object to visualise.

    Returns
    -------
    graphviz.Digraph
        A graphviz Digraph representing the object.

    Raises
    ------
    NotImplementedError
        If the object type is not supported.
    """
    # Import here to avoid circular dependency
    from pbsf.models import PatternGraph, PatternTree

    if isinstance(obj, NestedWord):
        return _show_nested_word(obj)
    elif (isinstance(obj, PatternTree) or
          isinstance(obj, PatternGraph)):
        return _show_digraph(obj.graph)
    else:
        raise NotImplementedError(
            f"`show` is not implemented for object of type `{type(obj).__name__}`."
        )
