from typing import Any, Iterable, Optional

from bidict import bidict

from pbsf.utils.acceptors import FiniteAcceptor


class DFA(FiniteAcceptor):
    """
    Implementation of the Deterministic Finite Automaton (DFA).
    Automaton states and symbols can be linked to any object.

    DFAs describe regular languages, and thus recognise all sequences
    of symbols that are part of their language.
    """
    def __init__(self, name: Optional[str]):
        self.name: Optional[str] = name
        self.states: bidict[Any, int] = bidict({None: 0})
        self.alphabet: bidict[Any, int] = bidict()
        self.initial: int = 0
        self.final: set[int] = set()
        self.transitions: dict[int, dict[int, set[int]]] = {}
        self.__free_identifier = 0

    def __next_free_identifier(self) -> int:
        id = self.__free_identifier
        while ((id in self.states.inverse) or
               (id in self.alphabet.inverse)):
            id += 1
        self.__free_identifier = id
        return id

    def __validate_symbol(self, symbol: int) -> None:
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

    def add_symbol(self, symbol: Optional[Any]) -> int:
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
        id = self.__next_free_identifier()
        if symbol is not None:
            if symbol not in self.alphabet:
                self.alphabet[symbol] = id
            else:
                raise ValueError(f"Symbol {symbol} is already part of the alphabet.")
        else:
            self.alphabet[id] = id
        return id

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
        ---------
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
        id = self.__next_free_identifier()
        if state is not None:
            if state not in self.states:
                self.states[state] = id
            else:
                raise ValueError(f"{state} is already associated to a state.")
        else:
            self.states[id] = id
        return id

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
        Set a transition from state with identifier `s1` to state with
        identifier `s2` labelled with symbol `symbol`.

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
        Get the state reachable from a state and symbol
        given their identifier.

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
        return transitions.get(symbol, set())

    def follow(self, state: int, sequence: Iterable[int]) -> set[int]:
        """
        Get the state reachable from a given start state and
        a sequence of symbols given their identifier.

        DFAs return a singleton set with the reachable state if
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
        self.__validate_state(state)
        sequence = list(sequence)
        for symbol in set(sequence):
            self.__validate_symbol(symbol)
        for symbol in sequence:
            state_set = self.step(state, symbol)
            if not state_set:
                return set()
            state = state_set.pop()
        return {state}

    def accept(self, sequence: Iterable[int]) -> bool:
        """
        Returns whether the DFA accepts the given sequence of symbols
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
        sequence = list(sequence)
        for symbol in set(sequence):
            if symbol not in self.alphabet.inverse:
                return False

        state = self.initial
        for symbol in sequence:
            state_set = self.step(state, symbol)
            if not state_set:  # no matching transition
                return False
            state = state_set.pop()
        return state in self.final
