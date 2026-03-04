"""Implementation of mutable partially ordered sets."""

from collections import deque
from typing import Any, Iterator


class MutablePoset:
    """Represents a mutable partially ordered set."""

    def __init__(self, elements: set | None = None):
        if elements is None:
            elements = set()
        self._elements = set(elements)
        self._covering: dict[Any, set[Any]] = {}

    @property
    def elements(self) -> set[Any]:
        """The set of elements in the partially ordered set."""
        return self._elements

    @property
    def covering(self) -> dict[Any, set[Any]]:
        """Mapping from element to its direct predecessors."""
        return self._covering

    def __validate_element(self, element: Any) -> None:
        if element not in self._elements:
            raise ValueError(f"Element `{element}` is not in the poset.")

    def __validate_succeeding(self, a: Any, b: Any) -> None:
        if self.succeeds(a, b):
            raise ValueError(f"Element `{a}` already succeeds element `{b}`.")

    def __validate_covering(self, a: Any, b: Any) -> None:
        self.__validate_element(a)
        self.__validate_element(b)
        self.__validate_succeeding(a, b)
        self.__validate_succeeding(b, a)

    def add_element(self, element: Any) -> None:
        """
        Add an element to the partially ordered set.

        Parameters
        ----------
        element : Any
            The element to add to the poset.

        Raises
        ------
        ValueError
            Element is already part of the poset.
        """
        if element in self._elements:
            raise ValueError(f"Element `{element}` is already in the poset.")
        self._elements.add(element)

    def add_covering(self, a: Any, b: Any) -> None:
        """
        Add a covering relation between two elements.

        Parameters
        ----------
        a : Any
            The succeeding (larger) element.
        b : Any
            The preceding (smaller) element.

        Raises
        ------
        ValueError
            Elements are invalid or already precede/succeed each other.
        """
        self.__validate_covering(a, b)
        self._covering.setdefault(a, set()).add(b)

    def __bft(self, start: Any, end: Any | None = None) -> set[Any]:
        """
        Perform a breadth-first traversal starting from the given element.

        Parameters
        ----------
        start : Any
            The starting element for the traversal.
        end : Any | None
            Break and return when this element is reached, if provided.

        Returns
        -------
        set[Any]
            A set of elements reachable from the starting element.
        """
        visited = set()
        queue = deque([start])
        while queue:
            current = queue.popleft()
            if current not in visited:
                visited.add(current)
                queue.extend(self._covering.get(current, set()))
            if end is not None and current == end:
                break
        return visited

    def covers(self, a: Any, b: Any) -> bool:
        """
        Check if an element covers another element.

        Parameters
        ----------
        a : Any
            The covering (larger) element.
        b : Any
            The covered (smaller) element.

        Returns
        -------
        bool
            True if `a` directly covers `b` (i.e., `a > b` with no intermediate
            element), False otherwise.
        """
        return b in self._covering.get(a, set())

    def succeeds(self, a: Any, b: Any) -> bool:
        """
        Check if an element succeeds another element.

        Parameters
        ----------
        a : Any
            The succeeding (larger) element.
        b : Any
            The preceding (smaller) element.

        Returns
        -------
        bool
            True if `a` succeeds `b` (directly, transitively, or if `a == b`),
            False otherwise.
        """
        if a == b:
            return True
        return b in self.__bft(a, b)

    def precedes(self, a: Any, b: Any) -> bool:
        """
        Check if an element precedes another element.

        Parameters
        ----------
        a : Any
            The preceding (smaller) element.
        b : Any
            The succeeding (larger) element.

        Returns
        -------
        bool
            True if `a` precedes `b` (directly, transitively, or if `a == b`),
            False otherwise.
        """
        if a == b:
            return True
        return self.succeeds(b, a)

    @property
    def maximal(self) -> set[Any]:
        """
        Get all maximal elements of the poset.

        A maximal element is an element that has no succeeding elements.
        A poset can have more than one maximal element. If there is exactly one
        maximal element, it is called the greatest element of the poset.

        Returns
        -------
        set[Any]
            Set of maximal elements of the poset.
        """
        has_successor = set()
        for predecessors in self._covering.values():
            has_successor.update(predecessors)
        return self._elements - has_successor

    @property
    def greatest(self) -> Any | None:
        """
        Get the greatest element of the poset, if it exists.

        The greatest element of the poset is the unique maximal element.
        Returns None if there is not exactly one maximal element.

        Returns
        -------
        Any | None
            The greatest element of the poset.
        """
        maximals = self.maximal
        if len(maximals) == 1:
            return next(iter(maximals))
        return None

    def mc_subposet(self, element: Any) -> 'MutablePoset':
        """
        Get maximal connected subposet where `element` is the greatest element.

        Parameters
        ----------
        element : Any
            The greatest element of the requested subposet.

        Returns
        -------
        MutablePoset
            The maximal connected subposet where the element is the greatest.

        Raises
        ------
        ValueError
            The element is not part of the poset.
        """
        if element not in self.elements:
            raise ValueError(f"Element `{element}` is not in the poset.")
        elements = set(self.__bft(element))
        subposet = MutablePoset(elements)
        subposet._covering = {
            k: set(v) for k, v in self._covering.items() if k in subposet._elements
        }
        return subposet

    def __len__(self) -> int:
        """Get the number of elements in the partially ordered set."""
        return len(self._elements)

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the elements of the partially ordered set."""
        return iter(self._elements)

    def __eq__(self, other: Any) -> bool:
        """Check equality based on elements and covering relations."""
        if not isinstance(other, MutablePoset):
            return False
        return ((self._elements == other._elements) and
                (self._covering == other._covering))

    def __repr__(self) -> str:
        """Return a string representation showing the number of elements."""
        return f"MutablePoset({len(self)} elements)"
