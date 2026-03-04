import unittest

from pbsf.utils.sets import MutablePoset


class TestMutablePoset(unittest.TestCase):
    def test_creation(self):
        # Create an empty MutablePoset, check no elements or coverings
        empty = MutablePoset()
        self.assertEqual(empty.elements, set())
        self.assertEqual(empty.covering, {})
        # Create a MutablePoset with one element, check no coverings
        one = MutablePoset({1})
        self.assertEqual(one.elements, {1})
        self.assertEqual(one.covering, {})
        # Create a MutablePoset with five elements, check no coverings
        five = MutablePoset({1, 2, 3, 4, 5})
        self.assertEqual(five.elements, {1, 2, 3, 4, 5})
        self.assertEqual(five.covering, {})
        # Mutating the original set should not affect the poset
        original = {1, 2, 3}
        poset = MutablePoset(original)
        original.add(99)
        self.assertNotIn(99, poset.elements)

    def test_add_element(self):
        # Create an empty MutablePoset
        poset = MutablePoset()
        # Add an integer, check it is in elements
        poset.add_element(1)
        self.assertIn(1, poset.elements)
        # Add a character, check it is in elements
        poset.add_element('a')
        self.assertIn('a', poset.elements)
        # Add a string, check it is in elements
        poset.add_element("hello")
        self.assertIn("hello", poset.elements)
        # Attempt to add same string again, catch ValueError
        with self.assertRaises(ValueError):
            poset.add_element("hello")

    def test_covering(self):
        # Create an empty MutablePoset, add elements 1, 2, 3
        poset = MutablePoset()
        for i in (1, 2, 3):
            poset.add_element(i)
        self.assertEqual(poset.covering, {})
        # Add coverings 1 > 2 and 2 > 3
        poset.add_covering(1, 2)
        poset.add_covering(2, 3)
        # Confirm direct coverings exist, transitive and reverse do not
        self.assertTrue(poset.covers(1, 2))
        self.assertTrue(poset.covers(2, 3))
        self.assertFalse(poset.covers(1, 3))
        self.assertFalse(poset.covers(2, 1))
        self.assertFalse(poset.covers(3, 2))
        self.assertFalse(poset.covers(3, 1))
        # Element not in poset raises ValueError
        with self.assertRaises(ValueError):
            poset.add_covering(1, 99)
        with self.assertRaises(ValueError):
            poset.add_covering(99, 1)
        # Relation already exists (1 > 3 transitively via 2)
        with self.assertRaises(ValueError):
            poset.add_covering(1, 3)
        # Duplicate covering (1 > 2 already exists)
        with self.assertRaises(ValueError):
            poset.add_covering(1, 2)

    def test_precedes(self):
        # Create an empty MutablePoset, add elements 1, 2, 3, 4
        poset = MutablePoset()
        for i in (1, 2, 3, 4):
            poset.add_element(i)
        self.assertEqual(poset.covering, {})
        # Add coverings 1 > 2 > 3 and 2 > 4
        poset.add_covering(1, 2)
        poset.add_covering(2, 3)
        poset.add_covering(2, 4)
        # Confirm that 3 and 4 precede 2 and 1, and 2 precedes 1
        self.assertTrue(poset.precedes(3, 2))
        self.assertTrue(poset.precedes(3, 1))
        self.assertTrue(poset.precedes(4, 2))
        self.assertTrue(poset.precedes(4, 1))
        self.assertTrue(poset.precedes(2, 1))
        # Confirm that 1 precedes nothing
        for x in (1, 2, 3, 4):
            self.assertFalse(poset.precedes(1, x))

    def test_succeeds(self):
        # Create an empty MutablePoset, add elements 1, 2, 3, 4
        poset = MutablePoset()
        for i in (1, 2, 3, 4):
            poset.add_element(i)
        self.assertEqual(poset.covering, {})
        # Add coverings 1 > 2 > 3 and 2 > 4
        poset.add_covering(1, 2)
        poset.add_covering(2, 3)
        poset.add_covering(2, 4)
        # Confirm 1 succeeds 2, 3, and 4
        self.assertTrue(poset.succeeds(1, 2))
        self.assertTrue(poset.succeeds(1, 3))
        self.assertTrue(poset.succeeds(1, 4))
        # Confirm 2 succeeds 3 and 4
        self.assertTrue(poset.succeeds(2, 3))
        self.assertTrue(poset.succeeds(2, 4))
        # Confirm 3 and 4 succeed nothing
        for x in (1, 2, 3, 4):
            self.assertFalse(poset.succeeds(3, x))
            self.assertFalse(poset.succeeds(4, x))

    def test_greatest(self):
        # Poset without coverings
        poset = MutablePoset({1, 2, 3})
        # All elements are maximal elements (as no element succeeds them)
        self.assertEqual(poset.maximal, poset.elements)
        # Add covering relations
        poset.add_covering(1, 2)
        poset.add_covering(2, 3)
        # Only maximal element (and thus greatest) is 1
        self.assertEqual(poset.maximal, {1})
        self.assertEqual(poset.greatest, 1)
        # Add a new element 4 > 1
        poset.add_element(4)
        poset.add_covering(4, 1)
        # Only maximal element (and thus greatest) is 4
        self.assertEqual(poset.maximal, {4})
        self.assertEqual(poset.greatest, 4)
        # Two maximal elements, no greatest element
        poset.add_element(5)
        poset.add_covering(5, 1)
        self.assertEqual(poset.maximal, {4, 5})
        self.assertIsNone(poset.greatest)

    def test_mc_subposet(self):
        # Create poset with coverings 1 > 2 > 3 > 4 and 5 > 2
        poset = MutablePoset({1, 2, 3, 4, 5})
        poset.add_covering(1, 2)
        poset.add_covering(2, 3)
        poset.add_covering(3, 4)
        poset.add_covering(5, 2)
        # Confirm subposet contains all elements reachable from element
        self.assertEqual(poset.mc_subposet(1).elements, {1, 2, 3, 4})
        self.assertEqual(poset.mc_subposet(5).elements, {5, 2, 3, 4})
        self.assertEqual(poset.mc_subposet(2).elements, {2, 3, 4})
        self.assertEqual(poset.mc_subposet(3).elements, {3, 4})
        self.assertEqual(poset.mc_subposet(4).elements, {4})
        # Confirm subposet from non-existant element raises error
        with self.assertRaises(ValueError):
            poset.mc_subposet("not in poset")

    def test_length(self):
        # Empty MutablePoset has length 0
        self.assertEqual(len(MutablePoset()), 0)
        # MutablePoset with n elements has length n
        for n in range(1, 11):
            self.assertEqual(len(MutablePoset(set(range(n)))), n)

    def test_iter(self):
        # Iterating over the poset yields all elements
        elements = {1, 2, 3, 4, 5}
        poset = MutablePoset(elements)
        self.assertEqual(set(poset), elements)

    def test_equality(self):
        # Two posets with same elements and coverings are equal
        p1 = MutablePoset({1, 2, 3})
        p1.add_covering(1, 2)
        p1.add_covering(2, 3)
        p2 = MutablePoset({1, 2, 3})
        p2.add_covering(1, 2)
        p2.add_covering(2, 3)
        # Poset with different coverings is not equal
        p3 = MutablePoset({1, 2, 3})
        p3.add_covering(1, 2)
        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, p3)
        # Poset with different elements is not equal
        self.assertNotEqual(p1, MutablePoset({1, 2}))
        # Non-MutablePoset is not equal
        self.assertNotEqual(p1, "not a poset")

    def test_repr(self):
        # Confirm repr shows element count
        self.assertEqual(repr(MutablePoset()), "MutablePoset(0 elements)")
        self.assertEqual(repr(MutablePoset({1})), "MutablePoset(1 elements)")
        self.assertEqual(repr(MutablePoset({1, 2, 3})), "MutablePoset(3 elements)")
