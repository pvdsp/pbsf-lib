"""Nested word set model for hierarchical pattern matching."""

from pbsf.models.base import Model
from pbsf.models.pattern_graph import PatternGraph
from pbsf.models.pattern_tree import PatternTree
from pbsf.nodes import Node
from pbsf.utils.words.nested_word import NestedWord


class NestedWordSet(Model):
    """
    Model maintaining a set of nested words representing observed patterns.

    Nested words are formed by combining a fixed number (context size) of consecutive
    discretisation chains, creating structured representations of temporal sequences
    with hierarchical relationships.

    Parameters
    ----------
    params : dict | None, default=None
        Configuration dictionary. Optional keys:

        - context_size (int): Number of consecutive chains to combine into a nested
          word. Must be positive. Default is 2.
        - pattern_model (type): Pattern storage model class, either PatternGraph or
          PatternTree. Default is PatternGraph.
        - closest_match (bool): If True, use best match strategy for pattern matching.
          Default is True.

    Attributes
    ----------
    patterns : PatternGraph | PatternTree
        Pattern storage model that holds the discretisation nodes.
    nested_words : set[NestedWord]
        Set of nested words formed by combining recent discretisation chains.
    context_size : int
        Number of recent discretisation chains to combine into a nested word.
    pattern_model : type
        The pattern model class being used (PatternGraph or PatternTree).

    Raises
    ------
    ValueError
        If context_size is not positive.
    """

    def __init__(self, params: dict | None = None) -> None:
        self.params = params
        if self.params is None:
            self.params = {}
        context_size = self.params.get("context_size", 2)
        pattern_model = self.params.get("pattern_model", PatternGraph)
        if context_size <= 0:
            raise ValueError("Provided `context_size` must be positive.")
        self.pattern_model = pattern_model
        self.patterns = pattern_model({
            "closest_match": self.params.get("closest_match", True)
        })
        self.nested_words = set()
        self.context_size = context_size
        self._context_queue = []
        self._combined_cache = {}

    def _close_positions(self, nw: NestedWord, number: int) -> NestedWord:
        """
        Close a specified number of open positions in a nested word.

        Parameters
        ----------
        nw : NestedWord
            The nested word to modify.
        number : int
            Number of open positions to close.

        Returns
        -------
        NestedWord
            The modified nested word with positions closed.

        Raises
        ------
        ValueError
            If number is not positive or exceeds available open positions.
        """
        if number <= 0:
            raise ValueError("Cannot close negative number of open positions.")
        if number > len(pending := nw.matching.get_pending_calls()):
            raise ValueError("Number is greater than number of open positions.")
        for position in reversed(sorted(pending)[-number:]):
            nw.add_return(nw.word[position])
        return nw

    def _combine_nws(self, nw1: NestedWord, nw2: NestedWord) -> NestedWord:
        """
        Combine two nested words by matching pending calls and managing positions.

        Matches pending call positions between the two nested words. When symbols
        differ, closes mismatched positions and appends the remaining portion of nw2.
        Uses caching to avoid recomputing previously combined pairs.

        Parameters
        ----------
        nw1 : NestedWord
            First nested word.
        nw2 : NestedWord
            Second nested word to combine with the first.

        Returns
        -------
        NestedWord
            Combined nested word with properly managed hierarchical structure.
        """
        if len(nw1) == 0:
            return nw2
        if (nw1, nw2) in self._combined_cache:
            return self._combined_cache[(nw1, nw2)]
        # Positions and symbols of pending calls:
        p1 = sorted(nw1.matching.get_pending_calls())
        p2 = sorted(nw2.matching.get_pending_calls())
        s1 = [nw1.word[position] for position in p1]
        s2 = [nw2.word[position] for position in p2]
        nw = NestedWord() + nw1
        for depth, (c1, c2) in enumerate(zip(s1, s2)):
            if c1 != c2:
                self._close_positions(nw, len(s1) - depth)
                nw += nw2[p2[depth]:]
                break
        if nw.word[-1] != nw2.word[-1]:
            nw.add_internal(nw2.word[-1])
        self._combined_cache[(nw1, nw2)] = nw
        return nw

    def _combine_queue(self) -> NestedWord:
        """
        Combine all nested words in the context queue.

        Sequentially combines nested words from the context queue into a single
        nested word representing the combined hierarchical structure.

        Returns
        -------
        NestedWord
            Combined nested word from all queued elements.
        """
        result = NestedWord()
        for nw in self._context_queue:
            result = self._combine_nws(result, nw)
        return result

    def _chain_to_nw(self, chain: list[Node]) -> NestedWord:
        """
        Convert a discretisation chain to a nested word representation.

        Maps the chain to vertices in the pattern model and constructs a nested
        word with appropriate call and internal symbols.

        Parameters
        ----------
        chain : list[Node]
            Discretisation chain to convert.

        Returns
        -------
        NestedWord
            Nested word representation of the chain.

        Raises
        ------
        ValueError
            If an unsupported pattern model is used (not PatternGraph or PatternTree).
        """
        if self.pattern_model == PatternGraph:
            vertices, _ = self.patterns.chain_to_vertices(chain)
        elif self.pattern_model == PatternTree:
            vertices = self.patterns.chain_to_vertices(chain)
        else:
            raise ValueError(
                f"Unsupported pattern model:"
                f" {self.pattern_model}."
                f" Use PatternGraph or PatternTree."
            )
        nw = NestedWord()
        if len(vertices) > 1:
            nw.add_calls(vertices[:-1])
        nw.add_internal(vertices[-1])
        return nw

    def update(self, chain: list) -> list:
        """
        Update the model with a new discretisation chain.

        Adds the chain to the pattern model and context queue. If enough chains have
        been observed to fill the context window, forms a new nested word and adds it
        to the set.

        Parameters
        ----------
        chain : list
            Discretisation chain to add to the model.

        Returns
        -------
        list
            List containing the newly formed nested word if context is filled,
            otherwise an empty list.

        Raises
        ------
        ValueError
            If the chain is empty.
        """
        if len(chain) == 0:
            raise ValueError("Chain cannot be empty.")
        self.patterns.update(chain)
        nw = self._chain_to_nw(chain)
        if len(self._context_queue) >= self.context_size:
            self._context_queue.pop(0)
        self._context_queue.append(nw)
        if len(self._context_queue) == self.context_size:
            combined = self._combine_queue()
            self.nested_words.add(combined)
            return [combined]
        return []

    def learn(self, chains: list) -> list:
        """
        Learn patterns from a list of discretisation chains.

        Processes each chain sequentially, forming nested words as context windows
        are filled.

        Parameters
        ----------
        chains : list
            List of discretisation chains to learn from.

        Returns
        -------
        list
            List of all newly formed nested words during the learning process.
        """
        result = []
        for chain in chains:
            result.extend(self.update(chain))
        return result

    def contains(self, chains: list) -> bool:
        """
        Check if the model contains a nested word formed by the given chains.

        Combines the provided chains into a nested word and checks for membership
        in the learned set.

        Parameters
        ----------
        chains : list
            List of discretisation chains to combine and check.

        Returns
        -------
        bool
            True if the combined nested word exists in the model, False otherwise.

        Raises
        ------
        ValueError
            If chains is not a list of lists or if the number of chains doesn't
            match the context size.
        """
        if not all(isinstance(chain, list) for chain in chains):
            raise ValueError(
                f"NestedWordSet expects a list of"
                f" {self.context_size} discretisation chains."
            )
        if len(chains) != self.context_size:
            raise ValueError(f"Expected {self.context_size} chains, got {len(chains)}.")
        nws = [self._chain_to_nw(chain) for chain in chains]
        combined = NestedWord()
        for nw in nws:
            combined = self._combine_nws(combined, nw)
        return combined in self.nested_words

    def __repr__(self) -> str:
        """
        Return string representation of the NestedWordSet.

        Returns
        -------
        str
            String representation showing context size and number
            of learned nested words.
        """
        return (
            f"NestedWordSet(context_size={self.context_size},"
            f" nested_words={len(self.nested_words)})"
        )
