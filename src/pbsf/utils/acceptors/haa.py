"""Hierarchical-Alphabet Automaton implementation."""

import re
import textwrap
from typing import Any, NamedTuple

from pbsf.utils.acceptors import FiniteAcceptor
from pbsf.utils.sets import MutablePoset
from pbsf.utils.words import NestedWord, Word
from pbsf.utils.words.nested_word import MatchingRelation

from .bidfa import biDFA
from .dfa import DFA


class MappingCondition(NamedTuple):
    """
    A single condition in the HAA mapping.

    A condition specifies when the k-th acceptor in a chain should process
    the next nested subword: the active states of the first (k-1) acceptors
    must each belong to their respective state subset, and the current
    position label must belong to the symbol set.

    Attributes
    ----------
    acceptors : tuple[FiniteAcceptor, ...]
        Full k-tuple of acceptors forming a chain in the poset,
        starting with the greatest acceptor. The last element is the
        acceptor that will process the next nested subword.
    states : tuple[frozenset[int], ...]
        (k-1)-tuple of state subsets, one per acceptor in the chain,
        excluding the last. Each frozenset contains integer state IDs.
    symbols : frozenset[Any]
        Set of symbols corresponding to position labels for which
        this mapping applies.
    """

    acceptors: tuple[FiniteAcceptor, ...]
    states: tuple[frozenset[int], ...]
    symbols: frozenset[Any]


class HAA(FiniteAcceptor):
    """
    Hierarchical-Alphabet Automaton (HAA).

    A modular finite acceptor for languages of nested words.
    Consists of a partially ordered set of FiniteAcceptors, where each
    acceptor models patterns found at different nesting layers of a nested word.

    A mapping function determines which HAA substructure processes the nested subword
    corresponding to a deeper nesting layer, based on the chain of active acceptors,
    their active states, and the current position label.
    """

    def __init__(
        self,
        name: str | None = None,
        acceptors: MutablePoset | None = None,
        mapping: dict[tuple[FiniteAcceptor, ...], set[MappingCondition]] | None = None,
    ):
        self.name: str | None = name
        if acceptors is None:
            acceptors = MutablePoset()
        self.acceptors: MutablePoset[FiniteAcceptor] = acceptors

        # Mapping from a (k-1)-tuple prefix to the set of MappingConditions whose
        # acceptor chain starts with that prefix. Each condition's acceptors
        # tuple has length k, with the last element being the acceptor
        # of the relevant HAA substructure.
        if mapping is None:
            mapping = {}
        self.__mapping: dict[
            tuple[FiniteAcceptor, ...], set[MappingCondition]
        ] = mapping

    @classmethod
    def from_description(
        cls,
        description: str,
        types: dict[str, type[FiniteAcceptor]] | None = None,
    ) -> 'HAA':
        """
        Create a HAA from a textual description.

        The description has up to four sections, in order:

        1. An optional HAA name (first bare non-indented line).
        2. Acceptor blocks: each opened by a header line `<name> [<type>]:`
           (where `<type>` defaults to `dfa`) followed by an indented body
           passed to that acceptor's `from_description`.
        3. Covering relations — one per line: `<name1> > <name2>`.
        4. Mapping conditions — one per line:
           `<a_1> <a_2> … <a_k> (<states_1>) … (<states_{k-1}>) (<symbols_k>)`
           where each `(<states_i>)` is a space- or comma-separated subset of
           states of acceptor `<a_i>`, and ``(<symbols_k>)`` is the triggering
           symbol subset.

        Parameters
        ----------
        description : str
            Textual description of the HAA.
        types : dict[str, type[FiniteAcceptor]] | None
            Mapping from lowercase type keyword to acceptor class. Defaults to
            `{'dfa': DFA, 'bidfa': biDFA}`. Pass additional entries to
            support custom acceptor classes with a `from_description` method.

        Returns
        -------
        HAA
            HAA built from the textual description.

        Examples
        --------
        ::

            even-odd

            main dfa:
                initial 0
                final 0
                0 1 s
                1 2 s
                2 3 s
                3 0 s
            even dfa:
                initial 0
                final 2
                0 1 s
                1 2 s
                2 1 s
            odd dfa:
                initial 0
                final 3
                0 1 s
                1 2 s
                2 3 s
                3 2 s

            main > even
            main > odd

            main even (0) (s)
            main odd (2) (s)
        """
        if types is None:
            types = {'dfa': DFA, 'bidfa': biDFA}

        haa = cls()
        acceptors_dict: dict[str, FiniteAcceptor] = {}
        current_name: str | None = None
        current_type: type[FiniteAcceptor] | None = None
        current_lines: list[str] = []

        def _flush() -> None:
            nonlocal current_name, current_type, current_lines
            if current_name is None:
                return
            body = '\n'.join(current_lines)
            acceptor = current_type.from_description(current_name + '\n' + body)
            acceptors_dict[current_name] = acceptor
            haa.add_acceptor(acceptor)
            current_name = None
            current_type = None
            current_lines = []

        for line in textwrap.dedent(description).strip().split('\n'):
            stripped = line.strip()
            if not stripped:
                continue

            if line[0] in (' ', '\t'):
                if current_name is not None:
                    current_lines.append(line)
                continue

            _flush()

            if stripped.endswith(':'):
                words = stripped[:-1].split()
                type_key = words[-1].lower() if len(words) > 1 else 'dfa'
                name = ' '.join(words[:-1]) if len(words) > 1 else words[0]
                if name in acceptors_dict:
                    raise ValueError(f"Acceptor `{name}` is already defined.")
                if type_key not in types:
                    raise ValueError(
                        f"Unknown acceptor type `{type_key}`;"
                        f" known types: {list(types)}."
                    )
                current_name = name
                current_type = types[type_key]
                current_lines = []

            elif '>' in stripped:
                parts = stripped.split('>')
                if len(parts) != 2:
                    raise ValueError(f"Unrecognised line: {line!r}")
                left_name, right_name = parts[0].strip(), parts[1].strip()
                left = acceptors_dict.get(left_name)
                right = acceptors_dict.get(right_name)
                if left is None:
                    raise ValueError(f"Acceptor `{left_name}` not found.")
                if right is None:
                    raise ValueError(f"Acceptor `{right_name}` not found.")
                haa.acceptors.add_covering(left, right)

            elif '(' in stripped:
                paren_groups = re.findall(r'\(([^)]*)\)', stripped)
                names_part = re.sub(r'\([^)]*\)', '', stripped).strip()
                acceptor_names = names_part.split()
                if len(acceptor_names) < 2:
                    raise ValueError(
                        f"Mapping requires at least 2 acceptors,"
                        f" got {len(acceptor_names)}."
                    )
                if len(paren_groups) != len(acceptor_names):
                    raise ValueError(
                        f"Expected {len(acceptor_names)} parenthesized groups"
                        f" ({len(acceptor_names) - 1} state subset(s) + 1"
                        f" symbol subset), got {len(paren_groups)}."
                    )
                chain = []
                for name in acceptor_names:
                    a = acceptors_dict.get(name)
                    if a is None:
                        raise ValueError(f"Acceptor `{name}` not found.")
                    chain.append(a)
                chain = tuple(chain)
                state_subsets = []
                for group, acceptor in zip(paren_groups[:-1], chain[:-1]):
                    labels = group.replace(',', ' ').split()
                    ids = set()
                    for label in labels:
                        if label not in acceptor.states:
                            raise ValueError(
                                f"State `{label}` not found in"
                                f" acceptor `{acceptor.name}`."
                            )
                        ids.add(acceptor.states[label])
                    state_subsets.append(ids)
                symbol_group = paren_groups[-1]
                symbols = {s for s in symbol_group.replace(',', ' ').split() if s}
                haa.add_mapping(chain, tuple(state_subsets), symbols)

            else:
                if haa.name is not None:
                    raise ValueError(f"Unrecognised line: {line!r}")
                haa.name = stripped

        _flush()
        return haa

    def __validate_ordering(self, acceptors: tuple[FiniteAcceptor, ...]) -> None:
        # The chain must start with the greatest acceptor
        if acceptors[0] != self.acceptors.greatest:
            raise ValueError(
                f"k-tuple must start with the greatest acceptor,"
                f" got {acceptors[0]} instead of {self.acceptors.greatest}."
            )
        # Each acceptor in chain must succeed the next: a_{i-1} >= a_i in the poset
        for i in range(1, len(acceptors)):
            a2 = acceptors[i]
            a1 = acceptors[i - 1]
            if not self.acceptors.succeeds(a1, a2):
                raise ValueError(
                    f"k-tuple must be a chain in the poset, but"
                    f" {a1} does not succeed {a2}."
                )

    def __validate_length(
        self,
        acceptors: tuple[FiniteAcceptor, ...],
        states: tuple[set[int], ...],
    ) -> None:
        # k-tuple of acceptors pairs with a (k-1)-tuple of state subsets
        if len(acceptors) != len(states) + 1:
            raise ValueError(
                f"Expected a k-tuple of FiniteAcceptors and (k-1)-tuple of"
                f" state subsets, but acceptors has length {len(acceptors)}"
                f" and states has length {len(states)}."
            )

    def __validate_states(
        self,
        acceptors: tuple[FiniteAcceptor, ...],
        states: tuple[set[int], ...],
    ) -> None:
        # Each state subset must only contain valid state IDs for its acceptor.
        # The last acceptor in the tuple has no corresponding state subset.
        for acceptor, state_ids in zip(acceptors, states):
            valid = set(acceptor.states.inverse.keys())
            if not state_ids <= valid:  # state_ids must be subset of valid
                raise ValueError(
                    f"State subset {state_ids} contains IDs not in"
                    f" acceptor {acceptor}: valid IDs are {valid}."
                )

    @property
    def alphabet(self) -> set[Any]:
        """
        Get the set of all symbols across all acceptors in the poset.

        Returns
        -------
        set[Any]
            Union of the alphabets of every FiniteAcceptor in the poset.
        """
        return {s for a in self.acceptors for s in a.alphabet.keys()}

    def add_mapping(
        self,
        acceptors: tuple[FiniteAcceptor, ...],
        states: tuple[set[int], ...],
        symbols: set[Any],
    ) -> None:
        """
        Add a mapping condition to the HAA.

        Registers a condition under which the HAA substructure where the last acceptor
        in `acceptors` is the greatest acceptor of its partially ordered set will
        process the next nested subword corresponding to the deeper nesting layer.
        This condition is triggered when the active states of the first (k-1) acceptors
        each belong to their respective subset in `states`, and the relevant
        position label belongs to `symbols`.

        Parameters
        ----------
        acceptors : tuple[FiniteAcceptor, ...]
            A k-tuple forming a chain in the poset, starting with the
            greatest acceptor. The last element is the acceptor to invoke.
        states : tuple[set[int], ...]
            A (k-1)-tuple of state subsets, one per acceptor excluding the last.
            Each set contains integer state IDs of that acceptor.
        symbols : set[Any]
            Set of symbols that trigger this condition.

        Raises
        ------
        ValueError
            If the acceptor chain is not a valid ordered chain in the poset,
            if the lengths of `acceptors` and `states` are inconsistent,
            or if any state ID is not valid for its acceptor.
        """
        self.__validate_ordering(acceptors)
        self.__validate_length(acceptors, states)
        self.__validate_states(acceptors, states)
        condition = MappingCondition(
            acceptors=acceptors,
            states=tuple(frozenset(s) for s in states),
            symbols=frozenset(symbols),
        )
        prefix = acceptors[:-1]
        self.__mapping.setdefault(prefix, set()).add(condition)

    def find_mappings(
        self, acceptors: tuple[FiniteAcceptor, ...]
    ) -> set[MappingCondition]:
        """
        Get mapping conditions with k acceptors given their (k-1)-tuple prefix.

        Parameters
        ----------
        acceptors : tuple[FiniteAcceptor, ...]
            A (k-1)-tuple forming a chain in the poset, starting with the
            greatest acceptor.

        Returns
        -------
        set[MappingCondition]
            All conditions registered under this prefix. Empty set if none.

        Raises
        ------
        ValueError
            If `acceptors` is not a valid ordered chain in the poset.
        """
        self.__validate_ordering(acceptors)
        return set(self.__mapping.get(acceptors, set()))

    def _find_mapping(
        self,
        acceptors: tuple[FiniteAcceptor, ...],
        states: tuple[int, ...],
        symbol: Any,
    ) -> FiniteAcceptor | None:
        """
        Find the unique next acceptor triggered by the current state chain and symbol.

        Given a (k-1)-tuple prefix of active acceptors, their current states, and a
        symbol, returns the k-th acceptor from the unique matching MappingCondition,
        or None if no condition applies.

        Parameters
        ----------
        acceptors : tuple[FiniteAcceptor, ...]
            Current (k-1)-tuple prefix, starting with the greatest acceptor.
        states : tuple[int, ...]
            Current active state of each acceptor in the chain.
        symbol : Any
            Surface symbol at the current position.

        Returns
        -------
        FiniteAcceptor | None
            The next acceptor to invoke, or None if no condition matches.

        Raises
        ------
        ValueError
            If multiple conditions match: specific type of non-determinism.
        """
        found = None
        for condition in self.__mapping.get(acceptors, set()):
            if all(q in Q for q, Q in zip(states, condition.states)):
                if symbol in condition.symbols:
                    if found is not None:
                        raise ValueError(
                            f"Non-deterministic HAA: multiple conditions match"
                            f" for chain {acceptors}, states {states},"
                            f" symbol {symbol}."
                        )
                    found = condition.acceptors[-1]
        return found

    @staticmethod
    def _make_subword(word: NestedWord, call: int, ret: int) -> NestedWord:
        """
        Slice word[call:ret+1] and strip the outer match.

        The outer (call, ret) pair is removed from the matching relation so
        that both boundary positions are treated as internal positions during
        recursive recognition.

        Parameters
        ----------
        word : NestedWord
            The nested word to slice.
        call : int
            Index of the call position (becomes position 0 in the subword).
        ret : int
            Index of the matched return position.

        Returns
        -------
        NestedWord
            Subword with the outer call-return match removed.
        """
        sub = word[call:ret + 1]
        inner = {m for m in sub.matching.get_matches() if m[0] != 0}
        return NestedWord(sub.word, MatchingRelation(len(sub.word), inner))

    @staticmethod
    def _step_call_return(
        acceptor: FiniteAcceptor,
        state: int,
        subword: NestedWord,
    ) -> int | None:
        """
        Step `acceptor` through the call and return labels of a subword.

        Parameters
        ----------
        acceptor : FiniteAcceptor
            The active acceptor to step.
        state : int
            Current state of the acceptor.
        subword : NestedWord
            The subword whose first symbol is the call label and last symbol
            is the return label.

        Returns
        -------
        int | None
            New state after stepping through both labels, or None if either
            symbol is unknown or no transition exists.
        """
        call = subword.sequence[0]
        ret = subword.sequence[-1]
        next_states = acceptor.follow(state, Word((call, ret)))
        if not next_states:
            return None  # No transition on call or return label
        return next(iter(next_states))

    def __follow(
        self,
        acceptors: tuple[FiniteAcceptor, ...],
        states: tuple[int, ...],
        word: NestedWord,
    ) -> set[int]:
        """
        Process a nested word given the current acceptor chain and states.

        Parameters
        ----------
        acceptors : tuple[FiniteAcceptor, ...]
            Current chain of active acceptors, starting with the greatest.
            The last element is the active acceptor at the current depth.
        states : tuple[int, ...]
            Current state of each acceptor in the chain.
        word : NestedWord
            The nested word to process.

        Returns
        -------
        set[int]
            Singleton set with the final state of the active acceptor after
            processing the word, or empty set if the word is rejected.
        """
        states = list(states)
        acceptor = acceptors[-1]

        while len(word) > 0:
            pos = acceptor.next_position(states[-1], word)
            symbol = word.sequence[pos]
            next_acceptor = self._find_mapping(acceptors, tuple(states), symbol)

            if word.matching.is_call(pos) or word.matching.is_return(pos):
                if next_acceptor is None:
                    return set()  # no relevant mapping: reject
                call, ret = word.matching.get_match(pos)
                if (call is None) or (ret is None):
                    return set()  # pending position: reject
                subword = self._make_subword(word, call, ret)
                sub_follow = self.__follow(
                    acceptors + (next_acceptor,),
                    tuple(states) + (next_acceptor.initial,),
                    subword,
                )
                if not (sub_follow & next_acceptor.final):
                    return set()  # sub-HAA rejected: reject
                new_state = self._step_call_return(acceptor, states[-1], subword)
                if new_state is None:
                    return set()  # no relevant transitions in acceptor: reject
                states[-1] = new_state
                word = word[:call] + word[ret + 1:]

            else:  # internal position
                if (next_acceptor is not None
                        and next_acceptor.initial not in next_acceptor.final):
                    return set()  # mapping exists but sub-automaton rejects ε: reject
                next_states, word = acceptor.step(states[-1], word)
                if not next_states:
                    return set()  # no transition from state labelled symbol: reject
                states[-1] = next(iter(next_states))

        return {states[-1]}

    def size(self) -> tuple[int, int]:
        """
        Get the total number of states and transitions across all acceptors.

        Returns
        -------
        tuple[int, int]
            Sum of (states, transitions) over every acceptor in the poset.
        """
        states, transitions = 0, 0
        for a in self.acceptors:
            s, t = a.size()
            states += s
            transitions += t
        return states, transitions

    def step(self, state: int, word: Word) -> tuple[set[int], Word]:
        """
        Not supported: HAA processes nested words via `follow`.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("HAA does not support `step`; use `follow`.")

    def follow(self, state: int, word: NestedWord) -> set[int]:
        """
        Get the set of states reachable from an initial state over a nested word.

        Parameters
        ----------
        state : int
            Initial state identifier in the greatest acceptor.
        word : NestedWord
            The nested word to process.

        Returns
        -------
        set[int]
            Singleton set with the reachable state, or empty set if rejected.

        Raises
        ------
        TypeError
            If word is not a NestedWord.
        """
        if not isinstance(word, NestedWord):
            raise TypeError(f"Expected NestedWord, got {type(word).__name__}.")
        return self.__follow(
            (self.acceptors.greatest,),
            (state,),
            word
        )

    def accept(self, word: NestedWord) -> bool:
        """
        Return whether the HAA accepts the given nested word.

        Parameters
        ----------
        word : NestedWord
            The nested word to recognise.

        Returns
        -------
        bool
            True if the word is accepted, False otherwise.

        Raises
        ------
        TypeError
            If word is not a NestedWord.
        """
        greatest = self.acceptors.greatest
        result = self.follow(greatest.initial, word)
        if greatest is None:
            raise AttributeError(f"{self.acceptors} has no unique greatest element.")
        return bool(result & greatest.final)

    def add_acceptor(self, acceptor: FiniteAcceptor) -> None:
        """
        Add a FiniteAcceptor to the poset of the HAA.

        Parameters
        ----------
        acceptor : FiniteAcceptor
            The acceptor to add.

        Raises
        ------
        ValueError
            If the acceptor is already present in the poset.
        """
        if acceptor in self.acceptors:
            raise ValueError(f"Acceptor {acceptor} is already in the poset.")
        self.acceptors.add_element(acceptor)
