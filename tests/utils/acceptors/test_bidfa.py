import unittest

from pbsf.utils.acceptors import bidfa
from pbsf.utils.words import Word


class TestBiDFA(unittest.TestCase):
    def test_creation(self):
        # Create an empty biDFA, default name is None
        d = bidfa.biDFA()
        self.assertIsNone(d.name)
        # Create a named biDFA
        d = bidfa.biDFA(name="my_bidfa")
        self.assertEqual(d.name, "my_bidfa")
        # Check that there is one state
        self.assertEqual(len(d.states), 1)
        # Check that this is the initial state
        self.assertIn(d.initial, d.states.inverse)
        # Check that this state is a left state
        self.assertIn(d.initial, d.left)
        # Check that there are no right states
        self.assertEqual(len(d.right), 0)
        # Check that there are no final states
        self.assertEqual(len(d.final), 0)
        # Check that the alphabet is empty
        self.assertEqual(len(d.alphabet), 0)
        # Check that there are no transitions
        self.assertEqual(len(d.transitions), 0)
        self.assertEqual(d.size(), (1, 0))

    def test_from_description(self):
        # biDFA with no state identifier for initial
        with self.assertRaises(ValueError):
            bidfa.biDFA.from_description("empty\n    left 0\n    initial")

        # biDFA with left initial state
        d = bidfa.biDFA.from_description("empty\n    left 0\n    initial 0")
        self.assertEqual(d.name, "empty")
        self.assertEqual(d.size(), (1, 0))
        self.assertIn(d.initial, d.left)
        self.assertEqual(len(d.right), 0)
        self.assertNotIn(None, d.states)

        # biDFA with right initial state
        d = bidfa.biDFA.from_description("empty\n    right 0\n    initial 0")
        self.assertEqual(d.name, "empty")
        self.assertEqual(d.size(), (1, 0))
        self.assertIn(d.initial, d.right)
        self.assertEqual(len(d.left), 0)
        self.assertNotIn(None, d.states)

        # biDFA with left and right state, raises error
        with self.assertRaises(ValueError):
            bidfa.biDFA.from_description("empty\n    right 0\n    left 0")

        # biDFA with non-existent initial state should raise error
        with self.assertRaises(ValueError):
            bidfa.biDFA.from_description("empty\n    left 0\n    initial 1")

        # biDFA with non-existent final state should raise error
        with self.assertRaises(ValueError):
            bidfa.biDFA.from_description("empty\n    left 0\n    final 1")

        # biDFA with transition from or to non-existent state should raise error
        with self.assertRaises(ValueError):
            bidfa.biDFA.from_description("empty\n    left 0\n    right 1\n    0 2 a")
        with self.assertRaises(ValueError):
            bidfa.biDFA.from_description("empty\n    left 0\n    right 1\n    2 0 a")

        # biDFA with no initial state should raise error
        with self.assertRaises(ValueError):
            bidfa.biDFA.from_description("empty\n    left 0")

        # biDFA with >1 initial state should raise error
        with self.assertRaises(ValueError):
            bidfa.biDFA.from_description("empty\n    left 0 1\n    initial 0 1")

        # Duplicate initial lines should raise error
        with self.assertRaises(ValueError):
            bidfa.biDFA.from_description(
                "empty\n    left 0 1\n    initial 0\n    initial 1"
            )

        # Empty and whitespace-only lines should be skipped
        d = bidfa.biDFA.from_description(
            "a1\n"
            "\n"
            "    left 0\n"
            "    \n"
            "    right 1\n"
            "\n"
            "    initial 0\n"
            "    final 0\n"
            "    0 1 a\n"
            "    1 0 b"
        )
        self.assertEqual(d.name, "a1")
        self.assertEqual(d.size(), (2, 2))

        # Valid biDFA
        d = bidfa.biDFA.from_description(
            "a1\n"
            "    left 0\n"
            "    right 1\n"
            "    initial 0\n"
            "    final 0\n"
            "    0 1 a\n"
            "    1 0 b"
        )
        self.assertEqual(d.name, "a1")
        self.assertEqual(d.size(), (2, 2))
        self.assertNotIn(None, d.states)
        self.assertIn('a', d.alphabet)
        self.assertIn('b', d.alphabet)
        self.assertIn('0', d.states)
        self.assertIn('1', d.states)
        self.assertIn(d.states['0'], d.left)
        self.assertIn(d.states['1'], d.right)
        self.assertEqual(d.states['0'], d.initial)
        self.assertIn(d.initial, d.final)

        # Valid words
        self.assertTrue(d.accept(Word()))
        self.assertTrue(d.accept(Word("aabb")))
        self.assertTrue(d.accept(Word('a') * 10 + Word('b') * 10))

        # Invalid words
        self.assertFalse(d.accept(Word("a")))
        self.assertFalse(d.accept(Word("ba")))
        self.assertFalse(d.accept(Word("abb")))
        self.assertFalse(d.accept(Word("aabb") * 5))

        # Unrecognised line should raise error
        with self.assertRaises(ValueError):
            bidfa.biDFA.from_description(
                "a1\n"
                "    left 0\n"
                "    right 1\n"
                "    initial 0\n"
                "    final 0\n"
                "    0 1 a\n"
                "    1 0 a\n"
                "    0 0 b\n"
                "    1 1 b\n"
                "    invalid line"
            )

    def test_add_left(self):
        # Create an empty biDFA
        d = bidfa.biDFA()
        # Check that there is one state
        self.assertEqual(len(d.states), 1)
        # Check that this state is the initial state
        self.assertIn(d.initial, d.states.inverse)
        # Add a character 'a' as left state
        id_a = d.add_left('a')
        self.assertIn('a', d.states)
        self.assertEqual(d.states['a'], id_a)
        # Add an int(1) as left state
        id_1 = d.add_left(1)
        self.assertIn(1, d.states)
        self.assertEqual(d.states[1], id_1)
        # Add a string "state" as left state
        id_st = d.add_left("state")
        self.assertIn("state", d.states)
        self.assertEqual(d.states["state"], id_st)
        # Check that these three states are in biDFA
        self.assertEqual(len(d.states), 4)
        # Check that these three states are in self.left
        self.assertIn(id_a, d.left)
        self.assertIn(id_1, d.left)
        self.assertIn(id_st, d.left)
        # Check that these three states are not in self.right
        self.assertNotIn(id_a, d.right)
        self.assertNotIn(id_1, d.right)
        self.assertNotIn(id_st, d.right)
        # Attempt adding 1 as state again, catch ValueError
        with self.assertRaises(ValueError):
            d.add_left(1)
        # Add a state without associated object
        id_none = d.add_left(None)
        self.assertIn(id_none, d.states)
        self.assertIn(id_none, d.left)
        # Confirm that there are 5 left states and 0 right states
        self.assertEqual(len(d.left), 5)
        self.assertEqual(len(d.right), 0)

    def test_add_right(self):
        # Initial state is still a left state
        d = bidfa.biDFA()
        self.assertEqual(len(d.states), 1)
        self.assertIn(d.initial, d.states.inverse)
        # Add character 'a' as right state
        id_a = d.add_right('a')
        self.assertIn('a', d.states)
        self.assertEqual(d.states['a'], id_a)
        # Add int(1) as right state
        id_1 = d.add_right(1)
        self.assertIn(1, d.states)
        self.assertEqual(d.states[1], id_1)
        # Add string "state" as right state
        id_st = d.add_right("state")
        self.assertIn("state", d.states)
        self.assertEqual(d.states["state"], id_st)
        # Check that these three states are in biDFA
        self.assertEqual(len(d.states), 4)
        # Check that these three states are in self.right
        self.assertIn(id_a, d.right)
        self.assertIn(id_1, d.right)
        self.assertIn(id_st, d.right)
        # Check that these three states are not in self.left
        self.assertNotIn(id_a, d.left)
        self.assertNotIn(id_1, d.left)
        self.assertNotIn(id_st, d.left)
        # Attempt adding 1 as state again, catch ValueError
        with self.assertRaises(ValueError):
            d.add_right(1)
        # Add a state without associated object
        id_none = d.add_right(None)
        self.assertIn(id_none, d.states)
        self.assertIn(id_none, d.right)
        # Confirm that there are 1 left state (initial) and 4 right states
        self.assertEqual(len(d.left), 1)
        self.assertEqual(len(d.right), 4)

    def test_swap(self):
        # Create an empty biDFA
        d = bidfa.biDFA()
        # Check that there is one state
        self.assertEqual(len(d.states), 1)
        # Check that this state is the initial state
        self.assertIn(d.initial, d.states.inverse)
        # Check that this state is a left state
        self.assertIn(d.initial, d.left)
        # Check that this state is not a right state
        self.assertNotIn(d.initial, d.right)
        # Swap the initial state from left to right
        d.swap(d.initial)
        # Check that this state is not a left state
        self.assertNotIn(d.initial, d.left)
        # Check that this state is a right state
        self.assertIn(d.initial, d.right)
        # Swap the initial state from right to left
        d.swap(d.initial)
        # Check that this state is a left state
        self.assertIn(d.initial, d.left)
        # Check that this state is not a right state
        self.assertNotIn(d.initial, d.right)
        # Add a new left state
        id_left = d.add_left('L')
        # Add a new right state
        id_right = d.add_right('R')
        # Swap state 'L' from left to right
        d.swap(id_left)
        # Swap state 'R' from right to left
        d.swap(id_right)
        # Check that 'L' is now a right state
        self.assertNotIn(id_left, d.left)
        self.assertIn(id_left, d.right)
        # Check that 'R' is now a left state
        self.assertIn(id_right, d.left)
        self.assertNotIn(id_right, d.right)
        # Swap with invalid state, catch ValueError
        with self.assertRaises(ValueError):
            d.swap(999)

    def test_add_state(self):
        # add_state should delegate to add_left
        d = bidfa.biDFA()
        id_s = d.add_state('s')
        self.assertIn('s', d.states)
        self.assertIn(id_s, d.left)
        self.assertNotIn(id_s, d.right)

    def test_add_states(self):
        # add_states should produce n left states
        d = bidfa.biDFA()
        ids = d.add_states(['a', 'b', 'c'])
        self.assertEqual(len(ids), 3)
        for sid in ids:
            self.assertIn(sid, d.left)
            self.assertNotIn(sid, d.right)
        # 3 new + 1 initial = 4 left states
        self.assertEqual(len(d.left), 4)
        self.assertEqual(len(d.right), 0)

    def test_follow(self):
        # Create an empty biDFA
        d = bidfa.biDFA()
        # Add a left and right state
        q1 = d.add_right('q1')
        q2 = d.add_left('q2')
        # Add characters to alphabet
        a = d.add_symbol('a')
        b = d.add_symbol('b')
        # Add transitions
        d.set_transition(d.initial, q1, a)
        d.set_transition(q1, d.initial, b)
        d.set_transition(d.initial, q2, b)
        d.set_transition(q1, q2, a)
        d.set_transition(q2, q2, a)
        d.set_transition(q2, q2, b)
        # Check that follow([]) results in state 0
        self.assertEqual(d.follow(d.initial, Word()), {d.initial})
        # Check that follow("aaaabbbb") results in state 0
        self.assertEqual(d.follow(d.initial, Word("aaaabbbb")), {d.initial})
        # Check that follow("aaaabbb") results in state q1
        self.assertEqual(d.follow(d.initial, Word("aaaabbb")), {q1})
        # Check that follow("abababb") results in state q2
        self.assertEqual(d.follow(d.initial, Word("abababb")), {q2})

    def test_accept(self):
        # Repeat biDFA setup from test_follow
        d = bidfa.biDFA()
        q1 = d.add_right('q1')
        q2 = d.add_left('q2')
        a = d.add_symbol('a')
        b = d.add_symbol('b')
        d.set_transition(d.initial, q1, a)
        d.set_transition(q1, d.initial, b)
        d.set_transition(d.initial, q2, b)
        d.set_transition(q1, q2, a)
        d.set_transition(q2, q2, a)
        d.set_transition(q2, q2, b)
        d.final.add(d.initial)
        # Check if a^{n} b^{n} for n >= 0 is accepted
        for n in range(20):
            seq = Word('a') * n + Word('b') * n
            self.assertTrue(d.accept(seq))
        # Check that everything else is rejected
        self.assertFalse(d.accept(Word("a")))
        self.assertFalse(d.accept(Word("b")))
        self.assertFalse(d.accept(Word("aab")))
        self.assertFalse(d.accept(Word("abb")))
        self.assertFalse(d.accept(Word("ba")))
        self.assertFalse(d.accept(Word("baba")))
