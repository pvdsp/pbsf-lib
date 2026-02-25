"""Node types representing segment approximations at various granularities."""

from .base import Node
from .paa_node import PAANode
from .pla_node import PLANode
from .sax_node import SAXNode
from .slope_sign_node import SlopeSignNode
from .structural_prominence_node import StructuralProminenceNode
from .sum_node import SumNode

__all__ = [
    "Node",
    "PAANode",
    "PLANode",
    "SAXNode",
    "SlopeSignNode",
    "StructuralProminenceNode",
    "SumNode",
]
