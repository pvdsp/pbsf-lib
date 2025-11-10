from abc import ABC, abstractmethod

from pbsf.utils import has_required


class Node(ABC):
    """
    Abstract base class for discretisation nodes.

    A node represents an approximation of a segment at one specific level of
    granularity. A list of nodes forms a chain, which represents a segment
    using approximations ranging from coarse- to fine-grained granularity.

    Subclasses must implement methods for equivalence comparison, representation,
    visualisation, and distance calculation between nodes.
    """

    @abstractmethod
    def __eq__(self, other: 'Node') -> bool:
        """
        Check equivalence between this node and another node.

        Parameters
        ----------
        other : Node
            Another node to compare with.

        Returns
        -------
        bool
            True if nodes are equivalent, False otherwise.
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """
        Return string representation of the node.

        Returns
        -------
        str
            String representation of the node.
        """
        pass

    @abstractmethod
    def show(self) -> None:
        """
        Visualise the node.

        This method should draw a visual representation of the approximation, if applicable.
        """
        pass

    @abstractmethod
    def distance(self, node: 'Node') -> float:
        """
        Calculate the distance between this node and another node.

        Parameters
        ----------
        node : Node
            Another node to calculate distance to.

        Returns
        -------
        float
            Distance metric between the two nodes.
        """
        pass