import unittest

from pbsf.utils.words import MatchingRelation, NestedWord, Word


class TestMatchingRelation(unittest.TestCase):
    def test_creation(self):
        # Empty matching relation
        self.assertEqual(len(MatchingRelation(0)), 0)
        # Matching relation of length 10
        self.assertEqual(len(MatchingRelation(10)), 10)
        # Negative length raises ValueError
        with self.assertRaises(ValueError):
            MatchingRelation(-1)
        # Crossing matches raise ValueError
        with self.assertRaises(ValueError):
            MatchingRelation(10, {(0, 6), (4, 9)})
        # Duplicate call position raises ValueError
        with self.assertRaises(ValueError):
            MatchingRelation(10, {(0, 6), (0, 7)})
        # Duplicate return position raises ValueError
        with self.assertRaises(ValueError):
            MatchingRelation(10, {(0, 6), (1, 6)})
        # Return before call raises ValueError
        with self.assertRaises(ValueError):
            MatchingRelation(10, {(9, 0)})
        # Same call and return position raises ValueError
        with self.assertRaises(ValueError):
            MatchingRelation(10, {(5, 5)})
        # Negative position raises ValueError
        with self.assertRaises(ValueError):
            MatchingRelation(10, {(-1, 5)})
        # Out-of-bounds position raises ValueError
        with self.assertRaises(ValueError):
            MatchingRelation(10, {(0, 10)})

    def test_length(self):
        for length in range(0, 26, 5):
            self.assertEqual(len(MatchingRelation(length)), length)

    def test_call(self):
        # Matched call
        mr = MatchingRelation(10, {(0, 9)})
        self.assertTrue(mr.is_call(0))
        for i in range(1, 10):
            self.assertFalse(mr.is_call(i))
        # Pending call
        mr = MatchingRelation(10, {(0, None)})
        self.assertTrue(mr.is_call(0))
        for i in range(1, 10):
            self.assertFalse(mr.is_call(i))
        # Pending return: no calls
        mr = MatchingRelation(10, {(None, 0)})
        for i in range(10):
            self.assertFalse(mr.is_call(i))

    def test_return(self):
        # Matched return
        mr = MatchingRelation(10, {(0, 9)})
        self.assertTrue(mr.is_return(9))
        for i in range(0, 9):
            self.assertFalse(mr.is_return(i))
        # Pending return
        mr = MatchingRelation(10, {(None, 9)})
        self.assertTrue(mr.is_return(9))
        for i in range(0, 9):
            self.assertFalse(mr.is_return(i))
        # Pending call: no returns
        mr = MatchingRelation(10, {(9, None)})
        for i in range(10):
            self.assertFalse(mr.is_return(i))

    def test_internal(self):
        # Non-call, non-return positions are internal
        mr = MatchingRelation(10, {(0, 4), (5, 9)})
        for i in (0, 4, 5, 9):
            self.assertFalse(mr.is_internal(i))
        for i in (1, 2, 3, 6, 7, 8):
            self.assertTrue(mr.is_internal(i))

    def test_pending(self):
        # Pending return and pending call
        mr = MatchingRelation(10, {(None, 4), (5, None)})
        self.assertTrue(mr.is_pending(4))
        self.assertTrue(mr.is_pending(5))
        # get_pending, get_pending_calls, get_pending_returns
        mr = MatchingRelation(10, {(None, 3), (4, 6), (7, None)})
        self.assertEqual(mr.get_pending(), {(None, 3), (7, None)})
        self.assertEqual(mr.get_pending_calls(), {7})
        self.assertEqual(mr.get_pending_returns(), {3})

    def test_match(self):
        mr = MatchingRelation(10, {(None, 3), (4, 6), (7, None)})
        # Internal position returns None
        self.assertIsNone(mr.get_match(0))
        # Pending return
        self.assertEqual(mr.get_match(3), (None, 3))
        # Matched call
        self.assertEqual(mr.get_match(4), (4, 6))
        # Matched return
        self.assertEqual(mr.get_match(6), (4, 6))
        # Pending call
        self.assertEqual(mr.get_match(7), (7, None))
        # All matches
        self.assertEqual(mr.get_matches(), {(None, 3), (4, 6), (7, None)})

    def test_equal(self):
        # Same length, no matches
        self.assertEqual(MatchingRelation(5), MatchingRelation(5))
        # Different length
        self.assertNotEqual(MatchingRelation(5), MatchingRelation(6))
        # Same length, same matches
        matches = {(None, 3), (4, 6), (7, None)}
        self.assertEqual(MatchingRelation(10, matches), MatchingRelation(10, matches))
        # Same length, different matches
        self.assertNotEqual(
            MatchingRelation(10, matches),
            MatchingRelation(10, {(4, 6)}),
        )
        # Different length, same matches (not valid at length 7, but length 10 vs 8)
        self.assertNotEqual(
            MatchingRelation(10, matches),
            MatchingRelation(12, matches)
        )

    def test_iter(self):
        # Length 0: yields nothing
        self.assertEqual(list(MatchingRelation(0)), [])
        # Length 5: yields 0..4
        self.assertEqual(list(MatchingRelation(5)), [0, 1, 2, 3, 4])

    def test_hash(self):
        # Equal matching relations have the same hash
        matches = {(None, 3), (4, 6), (7, None)}
        self.assertEqual(
            hash(MatchingRelation(10, matches)),
            hash(MatchingRelation(10, matches))
        )
        # Different length
        self.assertNotEqual(
            hash(MatchingRelation(5)),
            hash(MatchingRelation(6))
        )
        # Same length, different matches
        self.assertNotEqual(
            hash(MatchingRelation(10, matches)),
            hash(MatchingRelation(10, {(4, 6)})),
        )

    def test_indexing(self):
        mr = MatchingRelation(10, {(None, 3), (4, 6), (7, None)})
        # Integer indexing
        self.assertIsNone(mr[0])
        self.assertIsNone(mr[1])
        self.assertIsNone(mr[2])
        self.assertEqual(mr[3], (None, 3))
        self.assertEqual(mr[4], (4, 6))
        self.assertIsNone(mr[5])
        self.assertEqual(mr[6], (4, 6))
        self.assertEqual(mr[7], (7, None))
        self.assertIsNone(mr[8])
        self.assertIsNone(mr[9])
        # Slice [3:8]: positions 3,4,5,6,7 remapped to 0,1,2,3,4
        sub = mr[3:8]
        self.assertEqual(len(sub), 5)
        self.assertEqual(sub[0], (None, 0))  # pending return
        self.assertEqual(sub[1], (1, 3))     # matched call
        self.assertIsNone(sub[2])            # internal
        self.assertEqual(sub[3], (1, 3))     # matched return
        self.assertEqual(sub[4], (4, None))  # pending call
        # Slice [0:6]: return of call at 4 is outside slice, so 4 becomes pending call
        sub = mr[0:6]
        self.assertEqual(len(sub), 6)
        self.assertIsNone(sub[0])
        self.assertIsNone(sub[1])
        self.assertIsNone(sub[2])
        self.assertEqual(sub[3], (None, 3))  # pending return unchanged
        self.assertEqual(sub[4], (4, None))  # call becomes pending
        self.assertIsNone(sub[5])

class TestNestedWord(unittest.TestCase):
    def test_creation(self):
        # Empty NW
        nw = NestedWord()
        self.assertEqual(nw.word, Word())
        self.assertEqual(nw.matching, MatchingRelation(0))
        self.assertEqual(nw.tagged, ())
        self.assertEqual(len(nw), 0)
        self.assertEqual(str(nw), "NestedWord(())")
        # Singleton word
        nw = NestedWord(Word('a'), MatchingRelation(1))
        self.assertEqual(nw.word, Word('a'))
        self.assertEqual(nw.matching, MatchingRelation(1))
        self.assertEqual(nw.tagged, ('a',))
        self.assertEqual(len(nw), 1)
        self.assertEqual(str(nw), "NestedWord(('a',))")
        # Length mismatch raises ValueError
        with self.assertRaises(ValueError):
            NestedWord(Word('a'), MatchingRelation(2))

    def test_from_tagged(self):
        # Empty NW
        nw1 = NestedWord.from_tagged("")
        nw2 = NestedWord()
        self.assertEqual(nw1, nw2)
        # Singleton NW
        nw1 = NestedWord.from_tagged("a")
        nw2 = NestedWord(Word("a"), MatchingRelation(1))
        self.assertEqual(nw1, nw2)
        # Long NW
        nw1 = NestedWord.from_tagged("<abb><ab>")
        nw2 = NestedWord(Word("abbab"),
                         MatchingRelation(5, {(0, 2), (3, 4)}))
        self.assertEqual(nw1, nw2)
        # Call and return at same position raises ValueError
        with self.assertRaises(ValueError):
            NestedWord.from_tagged("<a>")

    def test_tagged(self):
        # Check creation of tagged word representation
        nw = NestedWord(Word("abbab"),
                        MatchingRelation(5, {(0, 2), (3, 4)}))
        self.assertEqual(nw.tagged, ('<', 'a', 'b', 'b', '>', '<', 'a', 'b', '>'))

    def test_repr(self):
        # Check representation of empty NW
        nw = NestedWord()
        self.assertEqual(repr(nw), "NestedWord(Word(()), MatchingRelation(set()))")
        # Check representation of NW
        nw = NestedWord.from_tagged("<abb><ab>")
        self.assertEqual(repr(nw),
                         "NestedWord(Word(('a', 'b', 'b', 'a', 'b')),"
                         " MatchingRelation({(0, 2), (3, 4)}))")

    def test_equal(self):
        nw1 = NestedWord.from_tagged("")
        nw2 = NestedWord()
        nw3 = NestedWord.from_tagged("<abb><ab>")
        # Empty NWs are equal
        self.assertTrue(nw1 == nw2)
        # Non-equal NWs are not equal
        self.assertTrue(nw1 != nw3)
        # NWs are only equal to NWs
        self.assertTrue(nw3 != "<abb><ab>")

    def test_length(self):
        # Empty NW
        self.assertEqual(len(NestedWord()), 0)
        # Singleton NW
        self.assertEqual(len(NestedWord(Word('a'), MatchingRelation(1))), 1)
        # Longer NW
        self.assertEqual(len(NestedWord.from_tagged("<abb><ab>")), 5)

    def test_indexing(self):
        # Well-matched slicing
        nw1 = NestedWord.from_tagged("<abb><ab>")
        nw2 = NestedWord.from_tagged("<abb>")
        self.assertEqual(nw1[0:3], nw2)
        # General slicing
        nw2 = NestedWord.from_tagged("b><a")
        self.assertEqual(nw1[2:4], nw2)
        # Indexing
        symbol = 'b'
        matching = (0, 2)
        self.assertEqual(nw1[2], (symbol, matching))

    def test_concatenation(self):
        nw1 = NestedWord()
        nw2 = NestedWord.from_tagged("<abb>")
        nw3 = NestedWord.from_tagged("<ab>")
        nw4 = NestedWord.from_tagged("aa>")
        nw5 = NestedWord.from_tagged("<bb")
        sum = NestedWord.from_tagged("<bb<abb><ab>aa>")
        self.assertEqual(nw1 + nw5 + nw2 + nw3 + nw4, sum)

    def test_iter(self):
        # Empty NW yields nothing
        self.assertEqual(list(NestedWord()), [])
        # Yields (index, symbol) pairs
        nw = NestedWord.from_tagged("<abb><ab>")
        self.assertEqual(
            list(nw),
            [(0, 'a'), (1, 'b'), (2, 'b'), (3, 'a'), (4, 'b')],
        )

    def test_hash(self):
        # Equal NWs have the same hash
        nw1 = NestedWord.from_tagged("<abb><ab>")
        nw2 = NestedWord(Word("abbab"), MatchingRelation(5, {(0, 2), (3, 4)}))
        self.assertEqual(hash(nw1), hash(nw2))
        # Different NWs have different hashes
        nw3 = NestedWord.from_tagged("<abb>")
        self.assertNotEqual(hash(nw1), hash(nw3))
