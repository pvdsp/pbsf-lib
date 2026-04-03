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

    def next_position(self, state: int, word: Word) -> int:
        """
        Return the index in `word` that this acceptor will consume next.

        Subclasses can override this to implement non-left-to-right reading
        strategies. For example, a bidirectional DFA may return `len(word) - 1`
        when in a right state.

        Parameters
        ----------
        state : int
            Integer identifier of the current state.
        word : Word
            The word about to be processed.

        Returns
        -------
        int
            Index of the symbol to consume. The default implementation
            always returns `0`, corresponding to the leftmost symbol.
        """
        return 0

    @abstractmethod
    def step(self, state: int, word: Word) -> tuple[set[int], Word]:
        """
        Consume symbol from `word`, return resulting states and remaining word.

        The position consumed is determined by `next_position`. On success
        the symbol is removed from `word` and the remaining word is returned.
        On failure (unknown symbol or no valid transition) the original
        `word` is returned unchanged together with an empty state set.

        Parameters
        ----------
        state : int
            Integer identifier of the current state.
        word : Word
            The word to consume one symbol from.

        Returns
        -------
        tuple[set[int], Word]
            Reachable states (empty set on failure) and remaining word.
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
