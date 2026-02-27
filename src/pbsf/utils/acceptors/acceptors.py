"""Abstract base class for finite acceptors."""

from abc import ABC, abstractmethod

from pbsf.utils.words import Word


class FiniteAcceptor(ABC):
    """
    Abstract base class for finite acceptors.

    Subclasses must implement methods that return their size,
    the result of a transition given a state and symbol or word,
    and the result of acceptance or rejection given a word.
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
        Get the set of states reachable from a state and symbol.

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
    def follow(self, state: int, word: Word) -> set[int]:
        """
        Get the set of states reachable from a start state and word.

        Parameters
        ----------
        state : int
            Integer identifier of a specific start state.
        word : Word
            Sequence of symbols.

        Returns
        -------
        set[int]
            Set of the integer identifiers of states reachable from
            the given start state and word.
        """
        pass

    @abstractmethod
    def accept(self, word: Word) -> bool:
        """
        Return whether the acceptor accepts the given word.

        Parameters
        ----------
        word : Word
            Sequence of symbols.

        Returns
        -------
        bool
            Acceptance of the given word.
        """
        pass
