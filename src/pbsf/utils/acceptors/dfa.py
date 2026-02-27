"""Deterministic Finite Automaton implementation."""

from typing import Any, Iterable, Optional

from bidict import bidict

from pbsf.utils.words import Word

from .acceptors import FiniteAcceptor


class DFA(FiniteAcceptor):
    """
    Deterministic Finite Automaton (DFA).

    Automaton states and symbols can be linked to any object.
    DFAs describe regular languages, and thus recognise all words
    that are part of their language.
    """

    def __init__(self, name: Optional[str] = None):
        self.name: Optional[str] = name
        self.states: bidict[Any, int] = bidict({None: 0})
        self.alphabet: bidict[Any, int] = bidict()
        self.initial: int = 0
        self.final: set[int] = set()
        self.transitions: dict[int, dict[int, set[int]]] = {}
        self.__free_identifier = 0

    @classmethod
    def from_description(cls, description: str) -> 'DFA':
        """
        Create a DFA from a textual description.

        Example
        -------
        ```
        name of automaton
            initial <identifier of initial state>
            final <identifier(s) of final state(s)>
            <id from> <id to> <character>
            <id from> <id to> <character>
            ...
        ```

        ```
        a1
            initial 0
            final 0 1
            0 1 a
            1 0 b
        ```

        Parameters
        ----------
        description : str
            Textual description of the DFA.

        Returns
        -------
        DFA
            DFA built from the textual description.
        """
        # Parse the description
        lines = description.strip().split('\n')
        name = lines.pop(0)

        # Create instance of DFA
        dfa = cls(name=name)
        dfa.states.clear()
        dfa.initial = None

        # Go over description line by line
        for line in lines:
            parts = line.split()
            if not parts:
                continue

            # Setting initial state
            if parts[0] == 'initial':
                states = parts[1:]
                if not states:
                    raise ValueError("State expected after 'initial'.")
                if len(states) > 1:
                    raise ValueError("DFA can only have one initial state.")
                if dfa.initial is not None:
                    raise ValueError(f"DFA initial state already set as {dfa.initial};"
                                     f" multiple 'initial' lines not allowed.")
                if (q := states[0]) not in dfa.states:
                    dfa.add_state(q)
                dfa.initial = dfa.states[q]

            # Setting final states
            elif parts[0] == 'final':
                states = parts[1:]
                for state in states:
                    if state not in dfa.states:
                        dfa.add_state(state)
                    state_id = dfa.states[state]
                    dfa.final.add(state_id)

            # Setting transitions
            elif len(parts) == 3:
                q1, q2, symbol = parts
                if q1 not in dfa.states:
                    dfa.add_state(q1)
                if q2 not in dfa.states:
                    dfa.add_state(q2)
                if symbol not in dfa.alphabet:
                    dfa.add_symbol(symbol)
                q1_id = dfa.states[q1]
                q2_id = dfa.states[q2]
                sym_id = dfa.alphabet[symbol]
                dfa.set_transition(q1_id, q2_id, sym_id)

            # Raise error for unrecognised lines
            else:
                raise ValueError(f"Unrecognised line: {line}")

        # Raise error if there is no initial state set
        if dfa.initial is None:
            raise ValueError("DFA expects exactly one initial state.")
        return dfa

    def __next_free_identifier(self) -> int:
        identifier = self.__free_identifier
        while ((identifier in self.states.inverse) or
               (identifier in self.alphabet.inverse)):
            identifier += 1
        self.__free_identifier = identifier
        return identifier

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

    def size(self) -> tuple[int, int]:
        """
        Get the number of states and number of transitions of the DFA.

        Returns
        -------
        tuple[int, int]
            Tuple containing number of states and number of transitions
        """
        states = len(self.states)
        transitions = sum(len(t) for t in self.transitions.values())
        return states, transitions

    def add_symbol(self, symbol: Optional[Any] = None) -> int:
        """
        Add object as a symbol in the DFA's alphabet.

        If no specific object is given in case of an abstract symbol,
        the internal integer identifier is used as the symbol object.

        Parameters
        ----------
        symbol : Optional[Any]
            Optional object to associate with the new symbol.

        Returns
        -------
        int
            Identifier of the new symbol in the alphabet.

        Raises
        ------
        ValueError
            Symbol is already part of the DFA's alphabet.
        """
        identifier = self.__next_free_identifier()
        if symbol is not None:
            if symbol not in self.alphabet:
                self.alphabet[symbol] = identifier
            else:
                raise ValueError(f"Symbol {symbol} is already part of the alphabet.")
        else:
            self.alphabet[identifier] = identifier
        return identifier

    def add_symbols(self, symbols: Iterable[Any]) -> list[int]:
        """
        Add a collection of objects as distinct symbols in the DFA's alphabet.

        Parameters
        ----------
        symbols : Iterable[Any]
            Iterable collection of objects to associate with new symbols

        Returns
        -------
        list[int]
            An ordered list of identifiers corresponding to the new symbols.

        Raises
        ------
        ValueError
            One of the symbols is already part of the DFA's alphabet.
        """
        symbols = list(symbols)
        for symbol in symbols:
            if symbol in self.alphabet:
                raise ValueError(f"Symbol {symbol} already in alphabet.")
        return [self.add_symbol(symbol) for symbol in symbols]

    def add_state(self, state: Optional[Any] = None) -> int:
        """
        Add object as a state of the DFA.

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
            State is already part of the DFA.
        """
        identifier = self.__next_free_identifier()
        if state is not None:
            if state not in self.states:
                self.states[state] = identifier
            else:
                raise ValueError(f"{state} is already associated to a state.")
        else:
            self.states[identifier] = identifier
        return identifier

    def add_states(self, states: Iterable[Any]) -> list[int]:
        """
        Add a collection of objects as distinct states of the DFA.

        Parameters
        ----------
        states : Iterable[Any]
            Iterable collection of objects to associate with new states.

        Returns
        -------
        list[int]
            An ordered list of identifiers corresponding to the new states.

        Raises
        ------
        ValueError
            One of the objects is already associated to a state of the DFA.
        """
        states = list(states)
        for state in states:
            if state in self.states:
                raise ValueError(f"{state} is already associated to a state.")
        return [self.add_state(state) for state in states]

    def set_transition(self, s1: int, s2: int, symbol: int) -> None:
        """
        Set a transition from `s1` to `s2` labelled with `symbol`.

        Parameters
        ----------
        s1 : int
            Identifier of state with outgoing transition
        s2 : int
            Identifier of state with incoming transition
        symbol : int
            Identifier of the symbol that labels the transition

        Raises
        ------
        ValueError
            If provided states or symbol identifiers are respectively not
            a valid state or symbol, or if there is already an outgoing
            transition from state with identifier `s1` labelled with the symbol
            with identifier `symbol`.
        """
        self.__validate_state(s1)
        self.__validate_state(s2)
        self.__validate_symbol(symbol)
        self.transitions.setdefault(s1, {})
        if symbol in self.transitions[s1]:
            raise ValueError(f"There already exists a transition"
                             f" from {s1} labelled {symbol}")
        self.transitions[s1][symbol] = {s2}

    def step(self, state: int, symbol: int) -> set[int]:
        """
        Get the state reachable from a state and symbol.

        DFAs return a singleton set with the reachable state if
        there is a relevant transition, otherwise return the empty set.

        Parameters
        ----------
        state : int
            Integer identifier of a specific state.
        symbol : int
            Integer identifier of a specific symbol.

        Returns
        -------
        set[int]
            Singleton set with integer identifier of reachable state
            or empty set if there is no outgoing transition from state
            labelled with symbol.

        Raises
        ------
        ValueError
            If provided state or symbol identifier is respectively not
            a valid state or symbol.
        """
        self.__validate_state(state)
        self.__validate_symbol(symbol)
        transitions = self.transitions.get(state, {})
        # Return a defensive copy to prevent external mutation of internal state
        return set(transitions.get(symbol, set()))

    def follow(self, state: int, word: Word) -> set[int]:
        """
        Get the state reachable from a start state and word.

        DFAs return a singleton set with the reachable state if
        each symbol in the word corresponds to a relevant transition,
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

        sequence = list(word)
        for symbol in sequence:
            sid = self.alphabet.get(symbol)
            self.__validate_symbol(sid)
            state_set = self.step(state, sid)
            if not state_set:
                return set()
            state = next(iter(state_set))
        return {state}

    def accept(self, word: Word) -> bool:
        """
        Return whether the DFA accepts the given word.

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
