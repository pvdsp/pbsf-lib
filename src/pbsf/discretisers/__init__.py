"""Discretisation strategies for converting segments into symbolic representations."""

from .base import Discretiser
from .paa import PiecewiseAggregate
from .pla import PiecewiseLinear
from .sax import SymbolicAggregate

__all__ = ["Discretiser", "PiecewiseLinear", "PiecewiseAggregate", "SymbolicAggregate"]
