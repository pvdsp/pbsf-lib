"""Bidirectional Deterministic Finite Automaton implementation."""

from typing import Any, Iterable, Optional

from pbsf.utils.words import Word

from .dfa import DFA


class biDFA(DFA):
    """
    Bidirectional Deterministic Finite Automaton (biDFA).

    Automaton states are partitioned into two sets: left states and right states.
    biDFAs describe symmetric languages, and recognise all words that are part
    of their language.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.left: set[int] = {0}
        self.right: set[int] = set()

    @classmethod
    def from_description(cls, description: str) -> 'biDFA':
        """
        Create a biDFA from a textual description.

        All states should first explicitly be defined as left or right state.

        Example
        -------
        ```
        <name of automaton>
            left <identifier(s) of left state(s)>
            right <identifier(s) of right state(s)>
            initial <identifier of initial state>
            final <identifier(s) of final state(s)>
            <id from> <id to> <symbol>
            <id from> <id to> <symbol>
        ```

        ```
        a1
            left 0
            right 1
            initial 0
            final 0
            0 1 a
            1 0 b
        ```

        Parameters
        ----------
        description : str
            Textual description of the biDFA.

        Returns
        -------
        biDFA
            biDFA built from the textual description.
        """
        # Parse the description
        lines = description.strip().split('\n')
        name = lines.pop(0)

        # Create instance of biDFA
        d = cls(name=name)
        d.states.clear()
        d.left = set()
        d.initial = None

        # Go over the description line by line
        for line in lines:
            parts = line.split()
            if not parts:
                continue

            # Adding left states
            if parts[0] == 'left':
                states = parts[1:]
                for state in states:
                    if state in d.states:
                        raise ValueError(f"State {state} is already defined; {d.states}.")
                    d.add_left(state)

            # Adding right states
            elif parts[0] == 'right':
                states = parts[1:]
                for state in states:
                    if state in d.states:
                        raise ValueError(f"State {state} is already defined.")
                    d.add_right(state)

            # Setting initial state
            elif parts[0] == 'initial':
                states = parts[1:]
                if not states:
                    raise ValueError("State expected after 'initial'.")
                if len(states) > 1:
                    raise ValueError("biDFA can only have one initial state.")
                if d.initial is not None:
                    raise ValueError(f"biDFA initial state already set as {d.initial};"
                                     f" multiple 'initial' lines not allowed.")
                if (q := states[0]) not in d.states:
                    raise ValueError(f"State {q} should first be defined"
                                     f" as a left or right state.")
                d.initial = d.states[q]

            # Setting final states
            elif parts[0] == 'final':
                states = parts[1:]
                for state in states:
                    if state not in d.states:
                        raise ValueError(f"State {state} should first be defined"
                                         f" as a left or right state.")
                    d.final.add(d.states[state])

            # Setting transitions
            elif len(parts) == 3:
                q1, q2, symbol = parts
                if (q1 not in d.states) or (q2 not in d.states):
                    raise ValueError("States should first be defined"
                                     " as a left or right state.")
                if symbol not in d.alphabet:
                    d.add_symbol(symbol)
                q1_id = d.states[q1]
                q2_id = d.states[q2]
                sym_id = d.alphabet[symbol]
                d.set_transition(q1_id, q2_id, sym_id)

            # Raise error for unrecognised lines
            else:
                raise ValueError(f"Unrecognised line: {line}")

        # Raise error if there is no initial state set
        if d.initial is None:
            raise ValueError("biDFA expects exactly one initial state.")
        return d

    def __validate_symbol(self, symbol: int | None) -> None:
        if symbol is None:
            raise ValueError("Symbol is not in the alphabet")
        if not isinstance(symbol, int):
            raise TypeError(f"Symbol identifier {symbol} should be an integer.")
        if symbol not in self.alphabet.inverse:
            raise ValueError(f"Symbol {symbol} is not in the alphabet.")

    def __validate_state(self, state: int) -> None:
        if not isinstance(state, int):
            raise TypeError(f"State identifier {state} should be an integer.")
        if state not in self.states.inverse:
            raise ValueError(f"State {state} is not a valid state.")

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

    def follow(self, state: int, word: Word) -> set[int]:
        """
        Get the state reachable from a start state and word.

        In left states, the leftmost symbol of the word is consumed.
        In right states, the rightmost symbol of the word is consumed.

        biDFAs return a singleton set with the reachable state if
        each symbol of the word corresponds to a relevant transition,
        otherwise returns the empty set.

        Parameters
        ----------
        state : int
            Integer identifier of a specific state.
        word : Word
            Sequence of symbols.

        Returns
        -------
        set[int]
            Singleton set with integer identifier of reachable state
            or empty set if there is no outgoing transition somewhere
            along the path of visited states.

        Raises
        ------
        ValueError
            If provided state identifier or one of the symbols in the word
            is respectively not a valid state or symbol.
        """
        self.__validate_state(state)
        if not isinstance(word, Word):
            raise TypeError(f"Expected Word, received {type(word).__name__}.")
        symbols = list(word)
        while len(symbols) > 0:
            index = 0 if state in self.left else -1
            symbol = symbols.pop(index)
            sid = self.alphabet.get(symbol)
            self.__validate_symbol(sid)
            state_set = self.step(state, sid)
            if not state_set:
                return set()
            state = next(iter(state_set))
        return {state}

    def accept(self, word: Word) -> bool:
        """
        Return whether the biDFA accepts the given word.

        Parameters
        ----------
        word : Word
            Sequence of symbols.

        Returns
        -------
        bool
            Acceptance of the given word.

        Raises
        ------
        TypeError
            If the provided word is not a Word.
        """
        if not isinstance(word, Word):
            raise TypeError(f"Expected Word, received {type(word).__name__}.")

        symbols = list(word)
        for symbol in set(symbols):
            if symbol not in self.alphabet:
                return False

        state_set = self.follow(self.initial, word)
        if not state_set:
            return False
        return any(state in self.final for state in state_set)
