import unittest

from pbsf.utils.acceptors import dfa
from pbsf.utils.words import Word


class TestDFA(unittest.TestCase):
    def test_creation(self):
        # Create an empty DFA, default name is None
        d = dfa.DFA()
        self.assertIsNone(d.name)
        # Create a named DFA
        d = dfa.DFA(name="my_dfa")
        self.assertEqual(d.name, "my_dfa")
        # Check that there is one state and that this is the initial state
        self.assertEqual(len(d.states), 1)
        self.assertIn(d.initial, d.states.inverse)
        # Check that there are no final states
        self.assertEqual(len(d.final), 0)
        # Check that the alphabet is empty
        self.assertEqual(len(d.alphabet), 0)
        # Check that there are no transitions
        self.assertEqual(len(d.transitions), 0)
        self.assertEqual(d.size(), (1, 0))

    def test_from_description(self):
        # Empty DFA with initial state
        d = dfa.DFA.from_description("empty\n    initial 0")
        self.assertEqual(d.name, "empty")
        self.assertEqual(d.size(), (1, 0))
        self.assertNotIn(None, d.states)

        # Empty DFA with no initial state should raise error
        with self.assertRaises(ValueError):
            dfa.DFA.from_description("empty\n    final 0")

        # Empty DFA from with >1 initial state should raise error
        with self.assertRaises(ValueError):
            dfa.DFA.from_description("empty\n    initial 0 1")

        # Valid DFA
        d = dfa.DFA.from_description(
            "a1\n"
            "    initial 0\n"
            "    final 0\n"
            "    0 1 a\n"
            "    1 0 a\n"
            "    0 0 b\n"
            "    1 1 b"
        )
        self.assertEqual(d.name, "a1")
        self.assertEqual(d.size(), (2, 4))
        self.assertNotIn(None, d.states)
        self.assertIn('a', d.alphabet)
        self.assertIn('b', d.alphabet)
        self.assertIn('0', d.states)
        self.assertIn('1', d.states)
        self.assertEqual(d.states['0'], d.initial)
        self.assertIn(d.initial, d.final)

        # Valid words
        self.assertTrue(d.accept(Word()))
        self.assertTrue(d.accept(Word("aa")))
        self.assertTrue(d.accept(Word("aa") * 3))
        self.assertTrue(d.accept(Word("bb")))
        self.assertTrue(d.accept(Word("baab")))
        self.assertTrue(d.accept(Word("abba")))
        self.assertTrue(d.accept(Word("aabb")))
        self.assertTrue(d.accept(Word("bbaa")))
        self.assertTrue(d.accept(Word("bbaabb")))

        # Invalid words
        self.assertFalse(d.accept(Word("a")))
        self.assertFalse(d.accept(Word("a") * 9))
        self.assertFalse(d.accept(Word("ba")))
        self.assertFalse(d.accept(Word("ab")))
        self.assertFalse(d.accept(Word("bab") * 5))
        self.assertFalse(d.accept(Word("abbb")))

        # Unrecognised line should raise error
        with self.assertRaises(ValueError):
            dfa.DFA.from_description(
                "a1\n"
                "    initial 0\n"
                "    final 0\n"
                "    0 1 a\n"
                "    1 0 a\n"
                "    0 0 b\n"
                "    1 1 b\n"
                "    invalid line"
            )

    def test_add_symbol(self):
        # Create an empty DFA
        d = dfa.DFA()
        # Check that the alphabet is empty
        self.assertEqual(len(d.alphabet), 0)
        # Add a character 'a' as symbol
        id_a = d.add_symbol('a')
        # Check that 'a' is in alphabet
        self.assertIn('a', d.alphabet)
        self.assertEqual(d.alphabet['a'], id_a)
        # Add a character int(1) as symbol
        id_1 = d.add_symbol(1)
        # Check that 'a' and 1 are in alphabet
        self.assertIn('a', d.alphabet)
        self.assertIn(1, d.alphabet)
        self.assertEqual(d.alphabet[1], id_1)
        # Add a string "symbol" as symbol
        id_sym = d.add_symbol("symbol")
        # Check that 'a', 1, and "symbol" in alphabet
        self.assertIn('a', d.alphabet)
        self.assertIn(1, d.alphabet)
        self.assertIn("symbol", d.alphabet)
        self.assertEqual(d.alphabet["symbol"], id_sym)
        self.assertEqual(len(d.alphabet), 3)
        # Attempt adding 1 as symbol again, catch ValueError
        with self.assertRaises(ValueError):
            d.add_symbol(1)
        # Add a symbol without associated object, catch id
        id_none = d.add_symbol(None)
        # Check that this id is in the alphabet
        self.assertIn(id_none, d.alphabet)
        self.assertEqual(d.alphabet[id_none], id_none)
        self.assertEqual(len(d.alphabet), 4)

    def test_add_state(self):
        # Create an empty DFA
        d = dfa.DFA()
        # Check that there is only 1 state
        self.assertEqual(len(d.states), 1)
        # Check that this state is the initial state
        self.assertIn(d.initial, d.states.inverse)
        # Add a character 'a' as state
        id_a = d.add_state('a')
        self.assertIn('a', d.states)
        self.assertEqual(d.states['a'], id_a)
        # Add an int(1) as state
        id_1 = d.add_state(1)
        self.assertIn(1, d.states)
        self.assertEqual(d.states[1], id_1)
        # Add a string "state" as state
        id_st = d.add_state("state")
        self.assertIn("state", d.states)
        self.assertEqual(d.states["state"], id_st)
        self.assertEqual(len(d.states), 4)
        # Attempt adding 1 as state again, catch ValueError
        with self.assertRaises(ValueError):
            d.add_state(1)
        # Add a state without associated object
        id_none = d.add_state(None)
        self.assertIn(id_none, d.states)
        self.assertEqual(d.states[id_none], id_none)
        self.assertEqual(len(d.states), 5)

    def test_add_symbols(self):
        import string
        # Create an empty DFA
        d = dfa.DFA()
        # Use dfa.add_symbols to add string.ascii_lowercase
        ids_lower = d.add_symbols(string.ascii_lowercase)
        # Check that the alphabet has size 26
        self.assertEqual(len(d.alphabet), 26)
        self.assertEqual(len(ids_lower), 26)
        # Use dfa.add_symbols to add string.ascii_uppercase
        ids_upper = d.add_symbols(string.ascii_uppercase)
        # Check that the alphabet has size 52
        self.assertEqual(len(d.alphabet), 52)
        self.assertEqual(len(ids_upper), 26)
        # Use dfa.add_symbols to add string.hexdigits, catch ValueError
        with self.assertRaises(ValueError):
            d.add_symbols(string.hexdigits)
        # Check that the alphabet still has size 52
        self.assertEqual(len(d.alphabet), 52)

    def test_add_states(self):
        import string
        # Create an empty DFA
        d = dfa.DFA()
        # Use dfa.add_states to add string.ascii_lowercase
        ids_lower = d.add_states(string.ascii_lowercase)
        # When adding 26 states, we should have 27 states (including initial)
        self.assertEqual(len(d.states), 27)
        self.assertEqual(len(ids_lower), 26)
        # Use dfa.add_states to add string.ascii_uppercase
        ids_upper = d.add_states(string.ascii_uppercase)
        self.assertEqual(len(d.states), 53)
        self.assertEqual(len(ids_upper), 26)
        # Use dfa.add_states to add string.hexdigits, catch ValueError
        with self.assertRaises(ValueError):
            d.add_states(string.hexdigits)
        # Check that the states still has size 53
        self.assertEqual(len(d.states), 53)

    def test_transition(self):
        # Create an empty DFA
        d = dfa.DFA()
        # Check that there are no transitions
        self.assertEqual(d.size(), (1, 0))
        # Add one state, keep identifier
        state = d.add_state('q1')
        # Add one symbol, keep identifier
        sym1 = d.add_symbol('a')
        # Set a transition from 0 to state with symbol
        d.set_transition(0, state, sym1)
        # Set a transition from state to 0 with symbol
        d.set_transition(state, 0, sym1)
        # Set a transition from 0 to state with symbol, catch ValueError
        with self.assertRaises(ValueError):
            d.set_transition(0, state, sym1)
        # Add another symbol, keep identifier
        sym2 = d.add_symbol('b')
        # Set a transition from 0 to 0 with new symbol
        d.set_transition(0, 0, sym2)
        # Set a transition from 0 to state with new symbol, catch ValueError
        with self.assertRaises(ValueError):
            d.set_transition(0, state, sym2)
        # Confirm that there are three transitions in the DFA
        self.assertEqual(d.size(), (2, 3))
        # Create transition with a symbol as s1, catch ValueError
        with self.assertRaises(ValueError):
            d.set_transition(sym1, 0, sym1)
        # Create transition with a symbol as s2, catch ValueError
        with self.assertRaises(ValueError):
            d.set_transition(0, sym1, sym1)
        # Create transition with a state as symbol, catch ValueError
        with self.assertRaises(ValueError):
            d.set_transition(0, state, state)
        # Confirm that there are still three transitions in the DFA
        self.assertEqual(d.size(), (2, 3))

    def test_step(self):
        # Create an empty DFA
        d = dfa.DFA()
        # Add two states and two symbols
        s1, s2 = d.add_states(['q1', 'q2'])
        a, b = d.add_symbols(['a', 'b'])
        # Add transitions 0 -(a)-> s1 -(b)-> s2 -(a)-> 0
        d.set_transition(0, s1, a)
        d.set_transition(s1, s2, b)
        d.set_transition(s2, 0, a)
        # Confirm step(0, a) = {s1}
        self.assertEqual(d.step(0, a), {s1})
        # Confirm step(s1, b) = {s2}
        self.assertEqual(d.step(s1, b), {s2})
        # Confirm step(s2, a) = {0}
        self.assertEqual(d.step(s2, a), {0})
        # Confirm step(0, b) = {}
        self.assertEqual(d.step(0, b), set())
        # Confirm step(s1, a) = {}
        self.assertEqual(d.step(s1, a), set())
        # Confirm step(s2, b) = {}
        self.assertEqual(d.step(s2, b), set())
        # Add transitions 0 -(b)-> 0, s1 -(a)-> s1, s2 -(b)-> s2
        d.set_transition(0, 0, b)
        d.set_transition(s1, s1, a)
        d.set_transition(s2, s2, b)
        # Confirm step(0, b) = {0}
        self.assertEqual(d.step(0, b), {0})
        # Confirm step(s1, a) = {s1}
        self.assertEqual(d.step(s1, a), {s1})
        # Confirm step(s2, b) = {s2}
        self.assertEqual(d.step(s2, b), {s2})

    def test_follow(self):
        # Create an empty DFA
        d = dfa.DFA()
        # Add two states and two symbols
        s1, s2 = d.add_states(['q1', 'q2'])
        a, b = d.add_symbols(['a', 'b'])
        # Add transitions 0 -(a)-> s1 -(b)-> s2 -(a)-> 0
        d.set_transition(0, s1, a)
        d.set_transition(s1, s2, b)
        d.set_transition(s2, 0, a)
        # Test that "aba" -> {0}
        self.assertEqual(d.follow(0, Word("aba")), {0})
        # Test that "aba" * 10 -> {0}
        self.assertEqual(d.follow(0, Word("aba") * 10), {0})
        # Test that "bab" -> {}
        self.assertEqual(d.follow(0, Word("bab")), set())
        # Test that "abb" -> {}
        self.assertEqual(d.follow(0, Word("abb")), set())
        # Test with invalid state, catch ValueError
        with self.assertRaises(ValueError):
            d.follow(999, Word("abba"))
        # Test with invalid symbol, catch ValueError
        with self.assertRaises(ValueError):
            d.follow(0, Word(('a', 'b', 999)))
        # Empty sequence from state 0 returns {0}
        self.assertEqual(d.follow(0, Word()), {0})
        # Empty sequence from state s1 returns {s1}
        self.assertEqual(d.follow(s1, Word()), {s1})

    def test_size(self):
        # Create an empty DFA
        d = dfa.DFA()
        self.assertEqual(d.size(), (1, 0))
        # Add two states and two symbols
        s1, s2 = d.add_states(['q1', 'q2'])
        a, b = d.add_symbols(['a', 'b'])
        self.assertEqual(d.size(), (3, 0))
        # Add transitions 0 -(a)-> s1 -(b)-> s2 -(a)-> 0
        d.set_transition(0, s1, a)
        d.set_transition(s1, s2, b)
        d.set_transition(s2, 0, a)
        # Confirm size (3, 3)
        self.assertEqual(d.size(), (3, 3))

    def test_accept(self):
        # Create an empty DFA
        d = dfa.DFA()
        # Add two states and two symbols
        s1, s2 = d.add_states(['q1', 'q2'])
        a, b = d.add_symbols(['a', 'b'])
        # Add transitions 0 -(a)-> s1 -(b)-> s2 -(a)-> 0
        d.set_transition(0, s1, a)
        d.set_transition(s1, s2, b)
        d.set_transition(s2, 0, a)
        # Add state 0 as accepting state
        d.final.add(0)
        # Confirm acceptance of empty word
        self.assertTrue(d.accept(Word()))
        # Confirm acceptance of "aba"
        self.assertTrue(d.accept(Word("aba")))
        # Confirm acceptance of (a, b, a) * 10
        self.assertTrue(d.accept(Word("aba") * 10))
        # Confirm rejection of other sequences
        self.assertFalse(d.accept(Word("a")))
        self.assertFalse(d.accept(Word("b")))
        self.assertFalse(d.accept(Word("ab")))
        self.assertFalse(d.accept(Word("abb")))
        self.assertFalse(d.accept(Word("bab")))
        # Confirm rejection with invalid symbols
        self.assertFalse(d.accept(Word((999,))))

    def test_type_errors(self):
        # Build a small DFA to test against
        d = dfa.DFA()
        s1 = d.add_state('q1')
        a = d.add_symbol('a')
        d.set_transition(0, s1, a)
        # Step with non-int state
        with self.assertRaises(TypeError):
            d.step("q1", a)
        # Step with non-int symbol
        with self.assertRaises(TypeError):
            d.step(0, "a")
        # Follow with non-int state
        with self.assertRaises(TypeError):
            d.follow("q1", Word('a'))
        # Follow with non-word
        with self.assertRaises(TypeError):
            d.follow(0, a)
        # Set transition with non-int s1
        with self.assertRaises(TypeError):
            d.set_transition("q1", s1, a)
        # Set transition with non-int s2
        with self.assertRaises(TypeError):
            d.set_transition(0, "q1", a)
        # Set transition with non-int symbol
        with self.assertRaises(TypeError):
            d.set_transition(0, s1, "a")
        # Accept with non-word
        with self.assertRaises(TypeError):
            d.accept(("a",))
