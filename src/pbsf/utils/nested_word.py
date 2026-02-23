from __future__ import annotations

from typing import Any, Iterator


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
    """
    Represents a matching relation.
    """

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
            Optional list of (call, return) position pairs. Call or return can be None.
        """
        if matching is None:
            matching = []
        self._length: int = length
        self._return_successors: list[int | None] = [-1] * self._length
        self._call_predecessors: list[int | None] = [-1] * self._length
        for i, j in matching:
            self.set_match(i, j)

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
        return self._return_successors[i] != -1

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
        return self._call_predecessors[i] != -1

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
        return ((self._return_successors[i] is None) or
                (self._call_predecessors[i] is None))

    def _validate_position(self, pos: int) -> None:
        if not isinstance(pos, int):
            raise ValueError(
                f"Position {pos} must be an integer,"
                f" got {type(pos).__name__}."
            )
        if pos < 0:
            raise ValueError(f"Position {pos} must be non-negative.")
        if pos >= self._length:
            raise ValueError(
                f"Position {pos} is out of bounds"
                f" for length {self._length}."
            )

    def _validate_crossing(self, i: int | None, j: int | None) -> None:
        for call, ret in self.get_matches():
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

    def _validate_properties(self, i: int | None, j: int | None) -> None:
        _validate_indices(i, j)
        _validate_order(i, j)
        for pos in (i, j):
            if pos is not None:
                self._validate_position(pos)
        self._validate_crossing(i, j)

    def set_match(
        self, call: int | None, ret: int | None,
    ) -> tuple[int | None, int | None]:
        """
        Set a match between a call and return position.

        Parameters
        ----------
        call : int | None
            The call position. If None, the return position is pending.
        ret : int | None
            The return position. If None, the call position is pending.

        Returns
        -------
        tuple[int | None, int | None]
            The match tuple (call, ret).
        """
        self._validate_properties(call, ret)
        if call is not None:
            self._return_successors[call] = ret
        if ret is not None:
            self._call_predecessors[ret] = call
        return call, ret

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
        if not (0 <= i < self._length):
            raise ValueError(
                f"Position {i} is out of bounds"
                f" for length {self._length}."
            )
        if self.is_call(i):
            return i, self._return_successors[i]
        if self.is_return(i):
            return self._call_predecessors[i], i
        return None

    def get_matches(self) -> set[tuple[int | None, int | None]]:
        """
        Get all matches in the matching relation.

        Returns
        -------
        set[tuple[int | None, int | None]]
            A set of all matches in the matching relation.
        """
        return set(
            match
            for match in (
                self.get_match(i) for i in range(self._length)
            )
            if match is not None
        )

    def get_pending(self) -> set[tuple[int | None, int | None]]:
        """
        Get the set of all pending positions in the matching relation.

        Returns
        -------
        set[tuple[int | None, int | None]]
            Set of all pending positions.
        """
        return {
            (i, j) for i, j in self.get_matches()
            if i is None or j is None
        }

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

    def remove_match(self, i: int) -> None:
        """
        Remove a match at a call or return position.

        Parameters
        ----------
        i : int
            The call or return position to remove the match from.

        Raises
        ------
        ValueError
            If the position is an internal position.
        """
        if not self.is_internal(i):
            call, ret = self.get_match(i)
            if call is not None:
                self._return_successors[call] = -1
            if ret is not None:
                self._call_predecessors[ret] = -1
        else:
            raise ValueError(f"Position {i} is an internal position")

    def extend(self, length: int) -> None:
        """
        Extend the matching relation with `length` new positions.

        Parameters
        ----------
        length : int
            The number of positions to extend the matching relation with.
        """
        self._length += length
        self._return_successors.extend([-1] * length)
        self._call_predecessors.extend([-1] * length)

    def __len__(self) -> int:
        """
        Return the length of the matching relation.

        Returns
        -------
        int
            The length of the matching relation.
        """
        return self._length

    def __iter__(self) -> Iterator[int]:
        """
        Iterate over the matching relation.

        Returns
        -------
        Iterator[int]
            An iterator over the matching relation.
        """
        for i in range(self._length):
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
        return (self._return_successors == other._return_successors and
                self._call_predecessors == other._call_predecessors)

    def __getitem__(
        self, key: int | slice,
    ) -> 'MatchingRelation' | tuple[int | None, int | None]:
        """
        Get the match at the given index, or the matching relation
        from start to stop position.

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
            stop = key.stop if key.stop is not None else self._length
            if start < 0 or stop > self._length or start > stop:
                raise ValueError(
                    f"Slice {key} is out of bounds"
                    f" for length {self._length}."
                )
            length = stop - start
            submatch = MatchingRelation(length)
            for pos in range(start, stop):
                if self.is_call(pos):
                    call = pos - start
                    if self._return_successors[pos] is None:
                        ret = None
                    else:
                        return_pos = self._return_successors[pos] - start
                        ret = None if return_pos >= length else return_pos
                    submatch.set_match(call, ret)
                elif self.is_return(pos):
                    if self._call_predecessors[pos] is None:
                        call = None
                    else:
                        call_pos = self._call_predecessors[pos] - start
                        call = None if call_pos < 0 else call_pos
                    ret = pos - start
                    if call is None:
                        submatch.set_match(call, ret)
            return submatch
        elif isinstance(key, int):
            return self.get_match(key)


class NestedWord:
    """
    Represents a nested word.

    Attributes:
        word: The word of the nested word.
        matching: The matching relation of the nested word.
    """

    def __init__(
        self,
        word: list[Any] | None = None,
        matching: MatchingRelation | None = None,
    ) -> None:
        """
        Initialise a nested word.

        Parameters
        ----------
        word : list[Any] | None, default=None
            The word of the nested word.
        matching : MatchingRelation | None, default=None
            The matching relation of the nested word. Defaults to an empty matching.

        Raises
        ------
        ValueError
            If word and matching relation have different lengths.
        """
        if word is None:
            word = []
        if matching is None:
            matching = MatchingRelation(len(word))
        elif len(word) != len(matching):
            raise ValueError("Word and matching relation must have the same length.")
        self.word = word
        self.matching = matching

    @classmethod
    def from_tagged_sequence(cls, tagged_sequence: list[Any]) -> 'NestedWord':
        """
        Create a nested word from a tagged sequence.

        Parameters
        ----------
        tagged_sequence : list[Any]
            The tagged sequence to create the nested word from.

        Returns
        -------
        NestedWord
            A nested word created from the tagged sequence.
        """
        stack = []
        counter = 0
        word = [
            s for s in tagged_sequence
            if s != "<" and s != ">"
        ]
        matching = MatchingRelation(len(word))
        for symbol in tagged_sequence:
            if symbol == "<":
                stack.append(counter)
            elif symbol == ">":
                call = stack.pop() if len(stack) > 0 else None
                matching.set_match(call, max(0, counter - 1))
            else:
                counter += 1
        while len(stack) > 0:
            matching.set_match(stack.pop(), None)
        return cls(list(word), matching)

    @classmethod
    def from_tagged_word(cls, tagged_word: str) -> 'NestedWord':
        """
        Create a nested word from a tagged word.

        Parameters
        ----------
        tagged_word : str
            The tagged word to create the nested word from.

        Returns
        -------
        NestedWord
            A nested word created from the tagged word.
        """
        return cls.from_tagged_sequence(list(tagged_word))

    def to_tagged(self) -> list[Any]:
        """
        Convert the nested word to a tagged word.

        Returns
        -------
        list[Any]
            The tagged word representation of the nested word.
        """
        tagged = []
        for i, symbol in enumerate(self.word):
            if self.matching.is_call(i):
                tagged.append("<")
            tagged.append(symbol)
            if self.matching.is_return(i):
                tagged.append(">")
        return tagged

    def add_internals(self, symbols: list[Any]) -> None:
        """
        Extend the nested word with a list of internal positions labelled by `symbols`.

        Parameters
        ----------
        symbols : list[Any]
            A list of symbols to add as internal positions.
        """
        self.word.extend(symbols)
        self.matching.extend(len(symbols))

    def add_calls(self, symbols: list[Any]) -> None:
        """
        Extend the nested word with a list of pending call
        positions labelled by `symbols`.

        Parameters
        ----------
        symbols : list[Any]
            A list of symbols to add as pending positions.
        """
        self.add_internals(symbols)
        for pos in range(1, len(symbols) + 1):
            self.matching.set_match(len(self.word) - pos, None)

    def add_returns(self, symbols: list[Any]) -> None:
        """
        Extend the nested word with a list of return positions labelled by `symbols`.

        Matches pending calls with the new return positions in order, starting with
        the most recent pending calls.

        Parameters
        ----------
        symbols : list[Any]
            A list of symbols to add as return positions.
        """
        self.add_internals(symbols)
        pending = sorted(self.matching.get_pending_calls(), reverse=True)
        for i in range(len(symbols)):
            call = pending[i] if i < len(pending) else None
            ret = len(self.word) - len(symbols) + i
            self.matching.set_match(call, ret)

    def add_internal(self, symbol: Any) -> None:
        """
        Extend the nested word with an internal position labelled `symbol`.

        Parameters
        ----------
        symbol : Any
            Symbol to add as an internal position.
        """
        self.add_internals([symbol])

    def add_call(self, symbol: Any) -> None:
        """
        Extend the nested word with a pending call position labelled `symbol`.

        Parameters
        ----------
        symbol : Any
            Symbol to add as a pending call position.
        """
        self.add_calls([symbol])

    def add_return(self, symbol: Any) -> None:
        """
        Extend the nested word with a return position labelled `symbol`.

        Parameters
        ----------
        symbol : Any
            Symbol to add as a return position.
        """
        self.add_returns([symbol])

    def __len__(self) -> int:
        """
        Return the length of the nested word.

        Returns
        -------
        int
            The length of the nested word.
        """
        return len(self.word)

    def __getitem__(
        self, key: int | slice,
    ) -> tuple[Any, tuple[int | None, int | None] | None] | 'NestedWord':
        """
        If key is int, returns the symbol and matching relation at the given index.
        If key is slice, returns the nested subword from start to stop position.

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
            return NestedWord(self.word[key], self.matching[key])
        elif isinstance(key, int):
            return self.word[key], self.matching.get_match(key)

    def __str__(self) -> str:
        """
        Return the string representation of the nested word.

        Returns
        -------
        str
            The string representation of the nested word.
        """
        return f"NestedWord({self.to_tagged()})"

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
            self.word == other.word
            and self.matching == other.matching
        )

    def __ne__(self, other: Any) -> bool:
        """
        Check if the nested word is not equal to another object.

        Parameters
        ----------
        other : Any
            The object to compare.

        Returns
        -------
        bool
            True if the nested word is not equal to the other object, False otherwise.
        """
        return not self == other

    def __iter__(self) -> Iterator[tuple[int, Any]]:
        """
        Iterate over the nested word, yielding (index, symbol) tuples.

        Yields
        ------
        tuple[int, Any]
            Tuples of (position index, symbol) for each position in the word.
        """
        for index, symbol in enumerate(self.word):
            yield index, symbol

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
        combined = self.to_tagged() + other.to_tagged()
        return NestedWord.from_tagged_sequence(combined)

    def __hash__(self) -> int:
        """
        Return hash of the nested word based on its tagged representation.

        Returns
        -------
        int
            Hash value of the nested word.
        """
        return hash(tuple(self.to_tagged()))
