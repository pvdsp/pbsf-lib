import unittest

from pbsf.utils.words.word import Word


class TestWord(unittest.TestCase):
    def test_creation(self):
        # Empty word
        empty = Word(None)
        self.assertEqual(empty.sequence, ())
        # Word from string
        from_str = Word("word")
        self.assertEqual(from_str.sequence, ('w', 'o', 'r', 'd'))
        # Word from list
        from_list = Word(['w', 'o', 'r', 'd'])
        self.assertEqual(from_list.sequence, ('w', 'o', 'r', 'd'))
        # Word from range
        from_range = Word(range(5))
        self.assertEqual(from_range.sequence, (0, 1, 2, 3, 4))

    def test_length(self):
        # Empty word
        empty = Word(None)
        self.assertEqual(len(empty), 0)
        # Singleton word
        singleton = Word('w')
        self.assertEqual(len(singleton), 1)
        # Word from range
        twenty = Word(range(20))
        self.assertEqual(len(twenty), 20)

    def test_iter(self):
        # Empty word
        empty = Word(None)
        self.assertEqual(list(empty), [])
        # Word from string
        word = Word("word")
        self.assertEqual(list(word), ['w', 'o', 'r', 'd'])

    def test_equality(self):
        # Two empty words are equal
        w1 = Word(None)
        w2 = Word(None)
        self.assertEqual(w1, w2)
        # Two identical words are equal
        w3 = Word("word")
        w4 = Word("word")
        self.assertEqual(w3, w4)
        # Different words are not equal
        self.assertNotEqual(w3, w1)
        # Different length words are not equal
        w5 = Word("wordd")
        self.assertNotEqual(w5, w1)
        self.assertNotEqual(w5, w3)
        # Self-equality
        self.assertEqual(w5, w5)
        # Not equal to non-Word types
        self.assertNotEqual(w3, None)
        self.assertNotEqual(w3, 1)
        self.assertNotEqual(w3, True)

    def test_hashing(self):
        # Equal words have equal hashes
        w1 = Word("word")
        w2 = Word("word")
        self.assertEqual(hash(w1), hash(w2))

    def test_indexing(self):
        # Empty word: index raises IndexError
        empty = Word(None)
        with self.assertRaises(IndexError):
            empty[0]
        # Empty word: empty slice
        self.assertEqual(empty[0:0], Word(None))
        # Positive indexing
        word = Word("word")
        self.assertEqual(word[0], 'w')
        self.assertEqual(word[1], 'o')
        self.assertEqual(word[2], 'r')
        self.assertEqual(word[3], 'd')
        # Negative indexing
        self.assertEqual(word[-1], 'd')
        self.assertEqual(word[-2], 'r')
        # Slicing
        self.assertEqual(word[0:2], Word("wo"))
        self.assertEqual(word[1:3], Word("or"))
        self.assertEqual(word[2:4], Word("rd"))

    def test_representation(self):
        # Empty word
        empty = Word(None)
        self.assertEqual(repr(empty), "Word(())")
        # Word from string
        word = Word("word")
        self.assertEqual(repr(word), "Word(('w', 'o', 'r', 'd'))")

    def test_concatenation(self):
        # Empty + empty = empty
        empty = Word(None)
        self.assertEqual(empty + empty, Word(None))
        # Word + empty = word
        word = Word("word")
        self.assertEqual(word + empty, word)
        # Word + word
        self.assertEqual(word + word, Word("wordword"))
        # Chained concatenation
        self.assertEqual(word + empty + word, Word("wordword"))

    def test_multiplication(self):
        # Empty word multiplied
        empty = Word(None)
        self.assertEqual(empty * 1, Word(None))
        self.assertEqual(empty * 10, Word(None))
        # Word * 0 = empty
        word = Word("word")
        self.assertEqual(word * 0, Word(None))
        # Word * 1 = word
        self.assertEqual(word * 1, word)
        # Word * 3
        self.assertEqual(word * 3, Word("wordwordword"))
        # Negative multiplier = empty
        self.assertEqual(word * -1, Word(None))
