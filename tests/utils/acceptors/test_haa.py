import unittest

from pbsf.utils.acceptors import DFA, HAA, biDFA
from pbsf.utils.sets import MutablePoset
from pbsf.utils.words import NestedWord, Word


def even_odd() -> tuple[MutablePoset, tuple[DFA, DFA, DFA]]:
    main = DFA.from_description(
        """
        main
            initial 0
            final 0
            0 1 s
            1 2 s
            2 3 s
            3 0 s
        """
    )
    even = DFA.from_description(
        """
        even
            initial 0
            final 2
            0 1 s
            1 2 s
            2 1 s
        """
    )
    odd = DFA.from_description(
        """
        odd
            initial 0
            final 3
            0 1 s
            1 2 s
            2 3 s
            3 2 s
        """
    )
    poset = MutablePoset({main, even, odd})
    poset.add_covering(main, even)
    poset.add_covering(main, odd)
    return poset, (main, even, odd)


def make_haa() -> tuple[HAA, DFA, DFA, DFA]:
    """Build the even-odd HAA with both mapping conditions registered."""
    acceptors, (main, even, odd) = even_odd()
    haa = HAA(name="even-odd", acceptors=acceptors)
    haa.add_mapping(
        acceptors=(main, even),
        states=({main.states['0']},),
        symbols={'s'},
    )
    haa.add_mapping(
        acceptors=(main, odd),
        states=({main.states['2']},),
        symbols={'s'},
    )
    return haa, main, even, odd


def simplest_recursive_haa() -> HAA:
    """HAA that accepts any NestedWord with symbols 's' up to certain depth."""
    dfa = DFA.from_description(
        """
        any
            initial 0
            final 0
            0 0 s
        """
    )
    haa = HAA(name="simplest",
              acceptors=MutablePoset({dfa}))

    for k in range(2, 5):
        haa.add_mapping(
            acceptors=(dfa,) * k,
            states=({dfa.states['0']},) * (k - 1),
            symbols={'s'},
        )
    return haa


def palindromic_haa() -> HAA:
    bidfa = biDFA.from_description(
        """
        palindromic
            left 0 3
            right 1 2
            initial 0
            final 0 1 2
            0 1 p
            1 0 p
            0 2 q
            2 0 q
            1 3 q
            2 3 p
            3 3 p
            3 3 q
        """
    )
    haa = HAA(name="palindromic-haa",
              acceptors=MutablePoset({bidfa}))

    final_ids = {bidfa.states['0'], bidfa.states['1'], bidfa.states['2']}
    for k in range(2, 5):
        haa.add_mapping(
            acceptors=(bidfa,) * k,
            states=(final_ids,) * (k - 1),
            symbols={'p', 'q'},
        )
    return haa


EVEN_ODD_DESCRIPTION = """
    even-odd

    main dfa:
        initial 0
        final 0
        0 1 s
        1 2 s
        2 3 s
        3 0 s
    even dfa:
        initial 0
        final 2
        0 1 s
        1 2 s
        2 1 s
    odd dfa:
        initial 0
        final 3
        0 1 s
        1 2 s
        2 3 s
        3 2 s

    main > even
    main > odd

    main even (0) (s)
    main odd (2) (s)
"""


class TestHAAFromDescription(unittest.TestCase):
    def test_name_and_acceptors(self):
        haa = HAA.from_description(EVEN_ODD_DESCRIPTION)
        self.assertEqual(haa.name, "even-odd")
        self.assertEqual(len(haa.acceptors), 3)
        names = {a.name for a in haa.acceptors}
        self.assertEqual(names, {"main", "even", "odd"})

    def test_covering_relations(self):
        haa = HAA.from_description(EVEN_ODD_DESCRIPTION)
        main = next(a for a in haa.acceptors if a.name == "main")
        even = next(a for a in haa.acceptors if a.name == "even")
        odd = next(a for a in haa.acceptors if a.name == "odd")
        self.assertEqual(haa.acceptors.greatest, main)
        self.assertTrue(haa.acceptors.succeeds(main, even))
        self.assertTrue(haa.acceptors.succeeds(main, odd))

    def test_mappings(self):
        haa = HAA.from_description(EVEN_ODD_DESCRIPTION)
        main = next(a for a in haa.acceptors if a.name == "main")
        mappings = haa.find_mappings((main,))
        self.assertEqual(len(mappings), 2)

    def test_accept_reject(self):
        haa = HAA.from_description(EVEN_ODD_DESCRIPTION)
        self.assertTrue(haa.accept(NestedWord.from_tagged("<ss><sss>")))
        self.assertFalse(haa.accept(NestedWord.from_tagged("<sss>")))

    def test_no_name(self):
        description = """
            main dfa:
                initial 0
                final 0
                0 0 s
        """
        haa = HAA.from_description(description)
        self.assertIsNone(haa.name)
        self.assertEqual(len(haa.acceptors), 1)

    def test_bidfa_type(self):
        haa = HAA.from_description("""
            palindromic-haa

            palindromic bidfa:
                left 0 3
                right 1 2
                initial 0
                final 0 1 2
                0 1 p
                1 0 p
                0 2 q
                2 0 q
                1 3 q
                2 3 p
                3 3 p
                3 3 q
        """)
        self.assertEqual(haa.name, "palindromic-haa")
        self.assertEqual(len(haa.acceptors), 1)

    def test_custom_type(self):
        class CustomDFA(DFA):
            pass

        haa = HAA.from_description(
            """
            custom dfa:
                initial 0
                final 0
                0 0 s
            """,
            types={'dfa': CustomDFA},
        )
        self.assertIsInstance(next(iter(haa.acceptors)), CustomDFA)

    def test_error_duplicate_acceptor(self):
        with self.assertRaises(ValueError):
            HAA.from_description("""
                main dfa:
                    initial 0
                    final 0
                    0 0 s
                main dfa:
                    initial 0
                    final 0
                    0 0 s
            """)

    def test_error_unknown_type(self):
        with self.assertRaises(ValueError):
            HAA.from_description("""
                main nfa:
                    initial 0
                    final 0
            """)

    def test_error_covering_acceptor_not_found(self):
        with self.assertRaises(ValueError):
            HAA.from_description("""
                main dfa:
                    initial 0
                    final 0
                    0 0 s

                main > ghost
            """)

    def test_error_mapping_acceptor_not_found(self):
        with self.assertRaises(ValueError):
            HAA.from_description("""
                main dfa:
                    initial 0
                    final 0
                    0 0 s

                main ghost (0) (s)
            """)

    def test_error_mapping_too_few_acceptors(self):
        with self.assertRaises(ValueError):
            HAA.from_description("""
                main dfa:
                    initial 0
                    final 0
                    0 0 s

                main (0) (s)
            """)

    def test_error_mapping_mismatched_groups(self):
        with self.assertRaises(ValueError):
            HAA.from_description("""
                main dfa:
                    initial 0
                    final 0
                    0 0 s
                sub dfa:
                    initial 0
                    final 0
                    0 0 s

                main > sub

                main sub (0) (0) (s)
            """)

    def test_error_mapping_state_not_found(self):
        with self.assertRaises(ValueError):
            HAA.from_description("""
                main dfa:
                    initial 0
                    final 0
                    0 0 s
                sub dfa:
                    initial 0
                    final 0
                    0 0 s

                main > sub

                main sub (99) (s)
            """)

    def test_error_duplicate_name(self):
        with self.assertRaises(ValueError):
            HAA.from_description("""
                name-one
                name-two
            """)

    def test_error_malformed_covering(self):
        with self.assertRaises(ValueError):
            HAA.from_description("""
                main dfa:
                    initial 0
                    final 0
                    0 0 s

                main > sub > extra
            """)

    def test_error_covering_left_not_found(self):
        with self.assertRaises(ValueError):
            HAA.from_description("""
                main dfa:
                    initial 0
                    final 0
                    0 0 s

                ghost > main
            """)


class TestHAA(unittest.TestCase):
    def test_creation(self):
        haa = HAA()
        self.assertIsNone(haa.name)
        self.assertEqual(len(haa.acceptors), 0)

        acceptors, (main, even, odd) = even_odd()
        haa = HAA(name="even-odd", acceptors=acceptors)
        self.assertEqual(haa.name, "even-odd")
        self.assertEqual(len(haa.acceptors), 3)
        self.assertIn(main, haa.acceptors)
        self.assertIn(even, haa.acceptors)
        self.assertIn(odd, haa.acceptors)

    def test_mappings(self):
        acceptors, (main, even, odd) = even_odd()
        haa = HAA(name="even-odd", acceptors=acceptors)

        # No mappings yet
        self.assertEqual(haa.find_mappings((main,)), set())

        haa.add_mapping(
            acceptors=(main, even),
            states=({main.states['0']},),
            symbols={'s'},
        )
        haa.add_mapping(
            acceptors=(main, odd),
            states=({main.states['2']},),
            symbols={'s'},
        )

        # Two conditions registered under prefix (main,)
        mappings = haa.find_mappings(acceptors=(main,))
        self.assertEqual(len(mappings), 2)

        # Prefix starting with a non-greatest acceptor must raise ValueError
        with self.assertRaises(ValueError):
            haa.find_mappings((even,))

    def test_add_mapping_invalid_length(self):
        acceptors, (main, even, odd) = even_odd()
        haa = HAA(name="even-odd", acceptors=acceptors)
        # states must have length k = len(acceptors) - 1 = 1, not 0
        with self.assertRaises(ValueError):
            haa.add_mapping(
                acceptors=(main, even),
                states=(),
                symbols={'s'},
            )

    def test_add_mapping_invalid_state(self):
        acceptors, (main, even, odd) = even_odd()
        haa = HAA(name="even-odd", acceptors=acceptors)
        # State ID 999 does not exist in main
        with self.assertRaises(ValueError):
            haa.add_mapping(
                acceptors=(main, even),
                states=({999},),
                symbols={'s'},
            )

    def test_accept(self):
        haa, main, even, odd = make_haa()
        word = NestedWord.from_tagged("<ss><sss>")
        self.assertTrue(haa.accept(word))

    def test_reject(self):
        haa, main, even, odd = make_haa()
        word = NestedWord.from_tagged("<sss>")
        self.assertFalse(haa.accept(word))
        word = NestedWord.from_tagged("<ss>")
        self.assertFalse(haa.accept(word))

    def test_reject_no_mapping_for_call(self):
        haa, main, even, odd = make_haa()
        word = NestedWord.from_tagged("s<ss>")
        self.assertFalse(haa.accept(word))

    def test_reject_pending_call(self):
        haa, main, even, odd = make_haa()
        word = NestedWord.from_tagged("<s")
        self.assertFalse(haa.accept(word))

    def test_reject_pending_return(self):
        haa, main, even, odd = make_haa()
        word = NestedWord.from_tagged("s>")
        self.assertFalse(haa.accept(word))

    def test_follow_type_error(self):
        haa, main, even, odd = make_haa()
        with self.assertRaises(TypeError):
            haa.follow(main.initial, Word(['s']))

    def test_simplest_recursive(self):
        haa = simplest_recursive_haa()
        self.assertTrue(haa.accept(NestedWord.from_tagged("ssss")))
        self.assertTrue(haa.accept(NestedWord.from_tagged("<ssss>")))
        self.assertTrue(haa.accept(NestedWord.from_tagged("<s<sssss>s>")))
        self.assertTrue(haa.accept(NestedWord.from_tagged("<s<s<ss><sss>ss>s>")))
        self.assertTrue(haa.accept(NestedWord.from_tagged("<s<s<ssss><sss><ss>ss>s>")))

    def test_palindromic(self):
        haa = palindromic_haa()
        self.assertTrue(haa.accept(NestedWord.from_tagged("")))
        self.assertTrue(haa.accept(NestedWord.from_tagged("ppp")))
        self.assertTrue(haa.accept(NestedWord.from_tagged("qqq")))
        self.assertTrue(haa.accept(NestedWord.from_tagged("pqqp")))
        self.assertTrue(haa.accept(NestedWord.from_tagged("pq<pp>qp")))
        self.assertTrue(haa.accept(NestedWord.from_tagged("pqp<pqp<pqppqp>pqp>pqp")))

    def test_add_mapping_invalid_ordering(self):
        # Chain (main, odd, even): main > odd is valid, but odd and even are
        # incomparable, so the chain is not a valid descending chain.
        acceptors, (main, even, odd) = even_odd()
        haa = HAA(name="even-odd", acceptors=acceptors)
        with self.assertRaises(ValueError):
            haa.add_mapping(
                acceptors=(main, odd, even),
                states=({main.states['0']}, {odd.states['0']}),
                symbols={'s'},
            )

    def test_alphabet(self):
        haa, main, even, odd = make_haa()
        self.assertEqual(haa.alphabet, {'s'})

    def test_nondeterministic_mapping(self):
        # Two conditions that both match main in state 0 on symbol 's'
        acceptors, (main, even, odd) = even_odd()
        haa = HAA(name="nondeterministic", acceptors=acceptors)
        haa.add_mapping(
            acceptors=(main, even),
            states=({main.states['0'], main.states['1']},),
            symbols={'s'},
        )
        haa.add_mapping(
            acceptors=(main, odd),
            states=({main.states['0'], main.states['2']},),
            symbols={'s'},
        )
        with self.assertRaises(ValueError):
            haa.accept(NestedWord.from_tagged("<ss>"))

    def test_reject_call_without_mapping(self):
        # HAA with no mapping conditions: nested structure always rejects
        dfa = DFA.from_description("""
            any
                initial 0
                final 0
                0 0 s
        """)
        haa = HAA(acceptors=MutablePoset({dfa}))
        self.assertFalse(haa.accept(NestedWord.from_tagged("<ss>")))

    def test_reject_no_outer_transition_for_call_return(self):
        # outer only processes 'a'; inner only processes 'b'.
        # Mapping fires on 'b', inner accepts, but outer has no 'b' transitions
        # so _step_call_return returns None and the word is rejected.
        outer = DFA.from_description("""
            outer
                initial 0
                final 0
                0 0 a
        """)
        inner = DFA.from_description("""
            inner
                initial 0
                final 0
                0 0 b
        """)
        poset = MutablePoset({outer, inner})
        poset.add_covering(outer, inner)
        haa = HAA(acceptors=poset)
        haa.add_mapping(
            acceptors=(outer, inner),
            states=({outer.states['0']},),
            symbols={'b'},
        )
        self.assertFalse(haa.accept(NestedWord.from_tagged("<bb>")))

    def test_reject_unknown_internal_symbol(self):
        # 'x' is not in main's alphabet, so step returns no states.
        haa, main, even, odd = make_haa()
        self.assertFalse(haa.accept(NestedWord.from_tagged("x")))

    def test_size(self):
        haa, main, even, odd = make_haa()
        states, transitions = haa.size()
        # main: 4 states / 4 transitions; even: 3/3; odd: 4/4
        self.assertEqual(states, 11)
        self.assertEqual(transitions, 11)

    def test_step_not_implemented(self):
        haa, main, even, odd = make_haa()
        with self.assertRaises(NotImplementedError):
            haa.step(main.initial, NestedWord.from_tagged("sss"))

    def test_add_acceptor_duplicate(self):
        haa, main, even, odd = make_haa()
        with self.assertRaises(ValueError):
            haa.add_acceptor(main)

    def test_accept_no_greatest_raises(self):
        # With two incomparable acceptors there is no unique greatest element;
        # greatest returns None and accept raises AttributeError.
        a = DFA.from_description("""
            a
                initial 0
                final 0
                0 0 s
        """)
        b = DFA.from_description("""
            b
                initial 0
                final 0
                0 0 s
        """)
        haa = HAA(acceptors=MutablePoset({a, b}))
        with self.assertRaises(AttributeError):
            haa.accept(NestedWord.from_tagged("s"))


if __name__ == '__main__':
    unittest.main()
