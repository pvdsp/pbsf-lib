from abc import ABC, abstractmethod
from collections.abc import Iterable


class FiniteAcceptor(ABC):
    """
    Abstract base class for finite acceptors.

    Subclasses must implement methods that return their size,
    the result of a transition given a state and symbol or sequence,
    and the result of acceptance of rejection given a Sequence.
    """
    @abstractmethod
    def size(self) -> tuple[int, int]:
        """
        Get the number of states and number of transitions of the acceptor.

        Returns
        -------
        tuple[int, int]
            Tuple containing number of states and number of transitions
        """
        pass

    @abstractmethod
    def step(self, state: int, symbol: int) -> set[int]:
        """
        Get the set of states reachable from a given state and symbol
        given their identifiers.

        Parameters
        ----------
        state : int
            Integer identifier of a specific state.
        symbol : int
            Integer identifier of a given symbol.

        Returns
        -------
        set[int]
            Set of the integer identifiers of states reachable from
            the given state and symbol.
        """
        pass

    @abstractmethod
    def follow(self, state: int, sequence: Iterable[int]) -> set[int]:
        """
        Get the set of states reachable from a given start state and
        a sequence of symbols given their identifiers.

        Parameters
        ----------
        state : int
            Integer identifier of a specific start state.
        sequence : Iterable[int]
            An iterable of integer symbol identifiers.

        Returns
        -------
        set[int]
            Set of the integer identifiers of states reachable from
            the given start state and sequence of symbols.
        """
        pass

    @abstractmethod
    def accept(self, sequence: Iterable[int]) -> bool:
        """
        Returns whether the acceptor accepts the given sequence of symbols
        given an iterable with their identifiers.

        Parameters
        ----------
        sequence : Iterable[int]
            An iterable of integer symbol identifiers

        Returns
        -------
        bool
            Acceptance of the given sequence.
        """
        pass
