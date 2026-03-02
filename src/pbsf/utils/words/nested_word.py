"""Nested word and matching relation data structures."""

from __future__ import annotations

from typing import Any, Iterable, Iterator

from pbsf.utils.words.word import Word


def _validate_indices(i: int | None, j: int | None) -> None:
    if i is None and j is None:
        raise ValueError(
            f"At least one position in match ({i}, {j})"
            f" must be an actual position."
        )


def _validate_order(i: int | None, j: int | None) -> None:
    if i is not None and j is not None:
        if i == j:
            raise ValueError(
                f"Position {i} in match ({i}, {j})"
                f" cannot be both call and return."
            )
        if i >= j:
            raise ValueError(
                f"Nestings only go forward, but position"
                f" {j} precedes position {i}."
            )


def _raise_crossing_error(
    c1: int | None, r1: int | None,
    c2: int | None, r2: int | None,
) -> None:
    raise ValueError(
        f"Match ({c1}, {r1}) crosses with"
        f" existing match ({c2}, {r2})."
    )


class MatchingRelation:
    """Represents a matching relation."""

    def __init__(
        self,
        length: int,
        matching: set[tuple[int | None, int | None]] | None = None,
    ) -> None:
        """
        Initialise a matching relation.

        Parameters
        ----------
        length : int
            The length of the matching relation.
        matching : set[tuple[int | None, int | None]] | None, default=None
            Optional set of (call, return) position pairs. Call or return can be None.
        """
        if matching is None:
            matching = set()
        if length < 0:
            raise ValueError(f"Length must be non-negative, got {length}.")
        self.__length: int = length
        self.__return_successors: list[int | None] = [-1] * self.__length
        self.__call_predecessors: list[int | None] = [-1] * self.__length
        self.__matches: set[tuple[int | None, int | None]] = set()
        for call, ret in matching:
            self.__validate_properties(call, ret)
            if call is not None:
                self.__return_successors[call] = ret
            if ret is not None:
                self.__call_predecessors[ret] = call
            self.__matches.add((call, ret))

    def __validate_properties(self, i: int | None, j: int | None) -> None:
        _validate_indices(i, j)
        for pos in (i, j):
            if pos is not None:
                self.__validate_position(pos)
        _validate_order(i, j)
        if i is not None and self.is_call(i):
            raise ValueError(f"Position {i} is already a call position.")
        if j is not None and self.is_return(j):
            raise ValueError(f"Position {j} is already a return position.")
        self.__validate_crossing(i, j)

    def __validate_position(self, pos: int) -> None:
        if not isinstance(pos, int):
            raise ValueError(
                f"Position {pos} must be an integer,"
                f" got {type(pos).__name__}."
            )
        if pos < 0:
            raise ValueError(f"Position {pos} must be non-negative.")
        if pos >= self.__length:
            raise ValueError(
                f"Position {pos} is out of bounds"
                f" for matching of length {self.__length}."
            )

    def __validate_crossing(self, i: int | None, j: int | None) -> None:
        for call, ret in self.__matches:
            if (i is None and call is None) or (j is None and ret is None):
                # (None, j) and (None, ret) will never cross
                # (i, None) and (call, None) will never cross
                continue
            elif i is None:  # (None, j)
                if ret is None:  # (None, j) and (call, None)
                    if call < j:
                        _raise_crossing_error(i, j, call, ret)
                else:  # (None, j) and (call, ret)
                    if call < j < ret:
                        _raise_crossing_error(i, j, call, ret)
            elif j is None:  # (i, None)
                if call is None:  # (i, None) and (None, ret)
                    if i < ret:
                        _raise_crossing_error(i, j, call, ret)
                else:  # (i, None) and (call, ret)
                    if call < i < ret:
                        _raise_crossing_error(i, j, call, ret)
            elif call is None:  # (i, j) and (None, ret)
                if i < ret < j:
                    _raise_crossing_error(i, j, call, ret)
            elif ret is None:  # (i, j) and (call, None)
                if i < call < j:
                    _raise_crossing_error(i, j, call, ret)
            else:  # (i, j) and (call, ret)
                if (i < call <= j <= ret) or (call < i <= ret <= j):
                    _raise_crossing_error(i, j, call, ret)

    def is_call(self, i: int) -> bool:
        """
        Check if a position is a call position.

        Parameters
        ----------
        i : int
            The position to check.

        Returns
        -------
        bool
            True if the position is a call position, False otherwise.
        """
        return self.__return_successors[i] != -1

    def is_return(self, i: int) -> bool:
        """
        Check if a position is a return position.

        Parameters
        ----------
        i : int
            The position to check.

        Returns
        -------
        bool
            True if the position is a return position, False otherwise.
        """
        return self.__call_predecessors[i] != -1

    def is_internal(self, i: int) -> bool:
        """
        Check if a position is an internal position.

        Parameters
        ----------
        i : int
            The position to check.

        Returns
        -------
        bool
            True if the position is an internal position, False otherwise.
        """
        return not self.is_call(i) and not self.is_return(i)

    def is_pending(self, i: int) -> bool:
        """
        Check if a position is a pending call or return position.

        Parameters
        ----------
        i : int
            The position to check.

        Returns
        -------
        bool
            True if the position is a pending call or return
            position, False otherwise.
        """
        return ((self.__return_successors[i] is None) or
                (self.__call_predecessors[i] is None))

    def get_match(self, i: int) -> tuple[int | None, int | None] | None:
        """
        Get the match for a call or return position.

        Parameters
        ----------
        i : int
            The call or return position.

        Returns
        -------
        tuple[int | None, int | None] | None
            The return successor or call predecessor for the given position.
            Returns (i, None) or (None, i) if the call or return
            is pending, respectively.
            Returns None if position is an internal position.

        Raises
        ------
        ValueError
            If position is out of bounds.
        """
        if not (0 <= i < self.__length):
            raise ValueError(
                f"Position {i} is out of bounds"
                f" for length {self.__length}."
            )
        if self.is_call(i):
            return i, self.__return_successors[i]
        if self.is_return(i):
            return self.__call_predecessors[i], i
        return None

    def get_matches(self) -> set[tuple[int | None, int | None]]:
        """
        Get all matches in the matching relation.

        Returns
        -------
        set[tuple[int | None, int | None]]
            A set of all matches in the matching relation.
        """
        return self.__matches

    def get_pending(self) -> set[tuple[int | None, int | None]]:
        """
        Get the set of all pending positions in the matching relation.

        Returns
        -------
        set[tuple[int | None, int | None]]
            Set of all pending positions.
        """
        return {(i, j) for i, j in self.__matches if i is None or j is None}

    def get_pending_calls(self) -> set[int]:
        """
        Get the set of all pending calls of the matching relation.

        Returns
        -------
        set[int]
            Set of all pending call positions.
        """
        return {i for i, j in self.get_matches() if j is None}

    def get_pending_returns(self) -> set[int]:
        """
        Get the set of all pending returns of the matching relation.

        Returns
        -------
        set[int]
            Set of all pending return positions.
        """
        return {j for i, j in self.get_matches() if i is None}

    def __len__(self) -> int:
        """
        Return the length of the matching relation.

        Returns
        -------
        int
            The length of the matching relation.
        """
        return self.__length

    def __iter__(self) -> Iterator[int]:
        """
        Iterate over the matching relation.

        Returns
        -------
        Iterator[int]
            An iterator over the matching relation.
        """
        for i in range(self.__length):
            yield i

    def __eq__(self, other: Any) -> bool:
        """
        Check if the matching relation is equal to another object.

        Parameters
        ----------
        other : Any
            The object to compare.

        Returns
        -------
        bool
            True if the matching relation is equal to the other
            object, False otherwise.
        """
        if not isinstance(other, MatchingRelation):
            return False
        return self.__length == other.__length and self.__matches == other.__matches

    def __hash__(self) -> int:
        """
        Return the hash of the matching relation.

        Returns
        -------
        int
            Hash of the matching relation.
        """
        return hash((self.__length, frozenset(self.__matches)))

    def __repr__(self) -> str:
        """
        Return the string representation of the matching relation.

        Returns
        -------
        str
            String representation of the matching relation.
        """
        return f"MatchingRelation({self.__matches})"

    def __getitem__(
        self, key: int | slice,
    ) -> 'MatchingRelation' | tuple[int | None, int | None]:
        """
        Get the match at a position, or a subrelation from a slice.

        Parameters
        ----------
        key : int | slice
            The index to retrieve the match, or the slice to
            retrieve the matching relation.

        Returns
        -------
        MatchingRelation | tuple[int | None, int | None]
            The match at the given index, or the matching relation
            from start to stop position.
        """
        if isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else self.__length
            if start < 0 or stop > self.__length or start > stop:
                raise ValueError(
                    f"Slice {key} is out of bounds"
                    f" for length {self.__length}."
                )
            length = stop - start
            matches = set()
            for pos in range(start, stop):
                if self.is_call(pos):
                    call = pos - start
                    if self.__return_successors[pos] is None:
                        ret = None
                    else:
                        return_pos = self.__return_successors[pos] - start
                        ret = None if return_pos >= length else return_pos
                    matches.add((call, ret))
                elif self.is_return(pos):
                    if self.__call_predecessors[pos] is None:
                        call = None
                    else:
                        call_pos = self.__call_predecessors[pos] - start
                        call = None if call_pos < 0 else call_pos
                    ret = pos - start
                    if call is None:
                        matches.add((call, ret))
            return MatchingRelation(length, matches)
        elif isinstance(key, int):
            return self.get_match(key)


class NestedWord(Word):
    """
    Represents a nested word.

    Attributes
    ----------
        word: The word of the nested word.
        matching: The matching relation of the nested word.
    """

    def __init__(
        self,
        word: Word | None = None,
        matching: MatchingRelation | None = None,
    ) -> None:
        """
        Initialise a nested word.

        Parameters
        ----------
        word : Word | None, default=None
            The word of the nested word. Defaults to an empty Word.
        matching : MatchingRelation | None, default=None
            The matching relation of the nested word. Defaults to an empty matching.

        Raises
        ------
        ValueError
            If word and matching relation have different lengths.
        """
        if word is None:
            word = Word()
        if matching is None:
            matching = MatchingRelation(len(word))
        elif len(word) != len(matching):
            raise ValueError("Word and matching relation must have the same length.")
        super().__init__(word.sequence)
        self._word: Word = word
        self._matching: MatchingRelation = matching
        self._tagged: tuple[Any, ...] | None = None

    @property
    def word(self) -> Word:
        """The word of the nested word."""
        return self._word

    @property
    def matching(self) -> MatchingRelation:
        """The matching relation of the nested word."""
        return self._matching

    @property
    def tagged(self) -> tuple[Any, ...]:
        """The tagged representation of the nested word."""
        if self._tagged is None:
            tagged = []
            for i, symbol in enumerate(self.word):
                if self.matching.is_call(i):
                    tagged.append("<")
                tagged.append(symbol)
                if self.matching.is_return(i):
                    tagged.append(">")
            self._tagged = tuple(tagged)
        return self._tagged

    @classmethod
    def from_tagged(cls, tagged: Iterable[Any]) -> 'NestedWord':
        """
        Create a nested word from a tagged sequence.

        Parameters
        ----------
        tagged : Iterable[Any]
            The tagged sequence to create the nested word from.

        Returns
        -------
        NestedWord
            A nested word created from the tagged sequence.
        """
        tagged = tuple(tagged)
        stack = []
        counter = 0
        word = [s for s in tagged if s != "<" and s != ">"]
        matches = set()
        for symbol in tagged:
            if symbol == "<":
                stack.append(counter)
            elif symbol == ">":
                call = stack.pop() if len(stack) > 0 else None
                matches.add((call, max(0, counter - 1)))
            else:
                counter += 1
        while len(stack) > 0:
            matches.add((stack.pop(), None))
        nw = cls(Word(word), MatchingRelation(len(word), matches))
        nw._tagged = tagged
        return nw

    def __getitem__(
        self, key: int | slice,
    ) -> tuple[Any, tuple[int | None, int | None] | None] | 'NestedWord':
        """
        Get the symbol and match at a position, or a nested subword from a slice.

        Parameters
        ----------
        key : int | slice
            The index to retrieve the symbol and matching relation.

        Returns
        -------
        tuple[Any, tuple[int | None, int | None] | None] | NestedWord
            A tuple of the symbol and matching relation at the
            given index, or a nested subword.
        """
        if isinstance(key, slice):
            return type(self)(self.word[key], self.matching[key])
        elif isinstance(key, int):
            return self.sequence[key], self.matching.get_match(key)
        else:
            raise TypeError(
                f"NestedWord indices must be integers or slices,"
                f" not {type(key).__name__}"
            )

    def __str__(self) -> str:
        """
        Return the string representation of the nested word.

        Returns
        -------
        str
            The string representation of the nested word.
        """
        return f"NestedWord({self.tagged})"

    def __repr__(self) -> str:
        """
        Return the string representation of the nested word.

        Returns
        -------
        str
            The string representation of the nested word.
        """
        return f"NestedWord({self.word}, {self.matching})"

    def __eq__(self, other: Any) -> bool:
        """
        Check if the nested word is equal to another object.

        Parameters
        ----------
        other : Any
            The object to compare.

        Returns
        -------
        bool
            True if the nested word is equal to the other object,
            False otherwise.
        """
        if not isinstance(other, NestedWord):
            return False
        return (
            self.sequence == other.sequence
            and self.matching == other.matching
        )

    def __add__(self, other: 'NestedWord') -> 'NestedWord':
        """
        Concatenate two nested words.

        Parameters
        ----------
        other : NestedWord
            The nested word to concatenate with.

        Returns
        -------
        NestedWord
            A new nested word that is the concatenation of the two nested words.
        """
        return type(self).from_tagged(self.tagged + other.tagged)

    def __hash__(self) -> int:
        """
        Return hash of the nested word based on its tagged representation.

        Returns
        -------
        int
            Hash value of the nested word.
        """
        return hash(self.tagged)
