"""Data structure related to a word."""

from typing import Any, Iterable, Iterator


class Word:
    """
    Immutable finite sequence of symbols.

    If no iterable is provided, the empty word is created.
    If an iterable is provided, its items form the symbols of the word.
    Words are used as input for FiniteAcceptors.
    """

    def __init__(self, sequence: Iterable[Any] | None = None):
        """
        Initialize a word from an optional sequence of symbols.

        Parameters
        ----------
        sequence : Iterable[Any] | None, optional
            Iterable whose items form the symbols of the word. If ``None``, the
            empty word is created.
        """
        data = tuple(sequence) if sequence is not None else ()
        self._data = data
        self._view = range(len(data))
        self._sequence = None

    @classmethod
    def _from_view(cls, data: tuple[Any, ...], view: range):
        w = cls.__new__(cls)
        w._data = data
        w._view = view
        w._sequence = None
        return w

    @property
    def sequence(self) -> tuple[Any, ...]:
        """Return the underlying tuple of symbols."""
        if self._sequence is None:
            self._sequence = tuple(self._data[i] for i in self._view)
        return self._sequence

    def get_symbol(self, pos: int) -> Any:
        """Return the raw symbol at position ``pos`` in O(1)."""
        return self._data[self._view[pos]]

    def __len__(self) -> int:
        """Return number of symbols in the word."""
        return len(self._view)

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the symbols of the word."""
        return iter(self._data[i] for i in self._view)

    def __eq__(self, other: Any) -> bool:
        """Check if the word is equal to another object."""
        if not isinstance(other, Word):
            return False
        return self.sequence == other.sequence

    def __hash__(self) -> int:
        """Return the hash of the word."""
        return hash(self.sequence)

    def __getitem__(self, key: int | slice) -> 'Word | Any':
        """Return the symbol at a position, or get a subword from a slice."""
        if isinstance(key, slice):
            return Word._from_view(self._data, self._view[key])
        elif isinstance(key, int):
            return self._data[self._view[key]]
        else:
            raise TypeError(
                f"Word indices must be integers or slices, not {type(key).__name__}"
            )

    def __repr__(self) -> str:
        """Return string representation of the word."""
        return f"Word({self.sequence})"

    def __add__(self, other: 'Word') -> 'Word':
        """
        Concatenate this word with another word.

        Parameters
        ----------
        other : Word
            The word to be concatenated to the right of this word.

        Returns
        -------
        Word
            A new word whose sequence is the concatenation of this word's
            sequence with ``other``'s sequence.
        """
        if not isinstance(other, Word):
            raise TypeError(
                f"Can only concatenate Word (not {type(other).__name__}) to Word"
            )
        if len(self) == 0:
            return other
        if len(other) == 0:
            return self
        return Word(self.sequence + other.sequence)

    def __mul__(self, n: int) -> 'Word':
        """
        Repeat this word a given number of times.

        Parameters
        ----------
        n : int
            The number of repetitions. Follows Python sequence repetition
            semantics for integers.

        Returns
        -------
        Word
            A new word whose sequence is this word's sequence repeated
            ``n`` times.
        """
        if not isinstance(n, int):
            raise TypeError(
                f"Word can only be multiplied by an int, got {type(n).__name__} instead"
            )
        return Word(self.sequence * n)

    def __rmul__(self, n: int) -> 'Word':
        """Support multiplication with the integer on the left: n * word."""
        return self.__mul__(n)
