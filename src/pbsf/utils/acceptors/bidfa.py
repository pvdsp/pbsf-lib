"""Bidirectional Deterministic Finite Automaton implementation."""

from typing import Any, Iterable, Optional

from .dfa import DFA


class biDFA(DFA):
    """
    Bidirectional Deterministic Finite Automaton (biDFA).

    Automaton states are partitioned into two sets: left states and right states.
    biDFAs describe symmetric languages, and recognise all sequences of symbols
    that are part of their language.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.left: set[int] = {0}
        self.right: set[int] = set()

    def add_state(self, state: Optional[Any] = None) -> int:
        """
        Add object as a left state of the biDFA.

        If no specific object is given in case of an abstract state,
        the internal integer identifier is used as the state object.

        Parameters
        ----------
        state : Optional[Any]
            Optional object to associate with the new state.

        Returns
        -------
        int
            Identifier of the state.

        Raises
        ------
        ValueError
            State is already part of the biDFA.
        """
        return self.add_left(state)

    def add_states(self, states: Iterable[Any]) -> list[int]:
        """
        Add a collection of objects as distinct left states of the biDFA.

        Each object in ``states`` is associated with a new left state.
        The order of the resulting state identifiers matches the order
        of the input objects.

        Parameters
        ----------
        states : Iterable[Any]
            Iterable of objects to associate with new left states.

        Returns
        -------
        list[int]
            List of integer identifiers for the newly added left states,
            in the same order as the input ``states`` iterable.

        Raises
        ------
        ValueError
            If any object in ``states`` is already associated to an
            existing state in the biDFA.
        """
        states = list(states)
        for state in states:
            if state in self.states:
                raise ValueError(f"{state} is already associated to a state.")
        return [self.add_state(state) for state in states]

    def swap(self, state: int) -> None:
        """
        Swap a left state to a right state or vice versa, given the state identifier.

        Parameters
        ----------
        state : int
            Identifier of the state to be swapped.

        Raises
        ------
        ValueError
            If provided state identifier is not a valid state.
        """
        if state in self.left:
            self.left.remove(state)
            self.right.add(state)
        elif state in self.right:
            self.right.remove(state)
            self.left.add(state)
        else:
            raise ValueError(f"{state} is not a valid state identifier.")

    def add_left(self, state: Optional[Any] = None) -> int:
        """
        Add object as a left state of the biDFA.

        If no specific object is given in case of an abstract state,
        the internal integer identifier is used as the state object.

        Parameters
        ----------
        state : Optional[Any]
            Optional object to associate with the new state.

        Returns
        -------
        int
            Identifier of the state

        Raises
        ------
        ValueError
            State is already part of the biDFA.
        """
        identifier = super().add_state(state)
        self.left.add(identifier)
        return identifier

    def add_right(self, state: Optional[Any] = None) -> int:
        """
        Add object as a right state of the biDFA.

        If no specific object is given in case of an abstract state,
        the internal integer identifier is used as the state object.

        Parameters
        ----------
        state : Optional[Any]
            Optional object to associate with the new state.

        Returns
        -------
        int
            Identifier of the state

        Raises
        ------
        ValueError
            State is already part of the biDFA.
        """
        identifier = super().add_state(state)
        self.right.add(identifier)
        return identifier

    def follow(self, state: int, sequence: Iterable[int]) -> set[int]:
        """
        Get the state reachable from a start state and symbol sequence.

        In left states, the leftmost symbol of the sequence is consumed.
        In right states, the rightmost symbol of the sequence is consumed.

        biDFAs return a singleton set with the reachable state if
        each symbol in the sequence corresponds to a relevant transition,
        otherwise returns the empty set.

        Parameters
        ----------
        state : int
            Integer identifier of a specific state.
        sequence : Iterable[int]
            Iterable of integer symbol identifiers.

        Returns
        -------
        set[int]
            Singleton set with integer identifier of reachable state
            or empty set if there is no outgoing transition somewhere
            along the path of followed transitions.

        Raises
        ------
        ValueError
            If provided state of one of the symbol identifiers in
            the sequence is respectively not a valid state or symbol.
        """
        symbols = list(sequence)
        while len(symbols) > 0:
            index = 0 if state in self.left else -1
            current = symbols.pop(index)
            state_set = self.step(state, current)
            if not state_set:
                return set()
            state = next(iter(state_set))
        return {state}

    def accept(self, sequence: Iterable[int]) -> bool:
        """
        Return whether the biDFA accepts the given symbol sequence.

        Parameters
        ----------
        sequence : Iterable[int]
            An iterable of integer symbol identifiers

        Returns
        -------
        bool
            Acceptance of the given sequence.

        Raises
        ------
        TypeError
            If one of the provided symbol identifiers is not an integer.
        """
        sequence = list(sequence)
        for symbol in set(sequence):
            if not isinstance(symbol, int):
                raise TypeError(
                    f"Symbol identifier must be int, got {type(symbol).__name__}"
                )
            if symbol not in self.alphabet.inverse:
                return False

        state_set = self.follow(self.initial, sequence)
        if not state_set:
            return False
        return any(state in self.final for state in state_set)
