"""Data structure related to a word."""

from typing import Any, Iterable


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
        self.sequence = tuple(sequence) if sequence is not None else ()

    def __len__(self):
        """Return number of symbols in the word."""
        return len(self.sequence)

    def __iter__(self):
        """Iterate over the symbols of the word."""
        return iter(self.sequence)

    def __eq__(self, other: Any) -> bool:
        """Check if the word is equal to another object."""
        if not isinstance(other, Word):
            return False
        return self.sequence == other.sequence

    def __hash__(self):
        """Return the hash of the word."""
        return hash(self.sequence)

    def __getitem__(self, key: int | slice) -> Any:
        """Return the symbol at a position, or get a subword from a slice."""
        if isinstance(key, slice):
            return Word(self.sequence[key])
        elif isinstance(key, int):
            return self.sequence[key]

    def __repr__(self) -> str:
        """Return string representation of the word."""
        return f"Word({self.sequence})"

    def __add__(self, other: 'Word') -> 'Word':
        """Return the concatenation of two words."""
        return Word(self.sequence + other.sequence)

    def __mul__(self, n: int) -> 'Word':
        """Return the concatenation of the word with n times itself."""
        if not isinstance(n, int):
            raise TypeError(
                f"Word can only be multiplied by an int, got {type(n).__name__} instead"
            )
        return Word(self.sequence * n)
