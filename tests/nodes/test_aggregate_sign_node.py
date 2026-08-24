import unittest

import numpy as np

from pbsf.nodes import AggregateSignNode, SlopeSignNode


class TestAggregateSignNode(unittest.TestCase):
    def test_creation(self):
        """Test the creation of an AggregateSignNode instance."""
        # Test the creation of a regular AggregateSignNode instance:
        node = AggregateSignNode({
            "depth": 0,
            "paa": np.array([1, 2, 3]),
            "breakpoints": []
        })
        self.assertEqual(node.depth, 0)
        self.assertTrue(np.array_equal(node.paa, np.array([1, 2, 3])))

        # Test creation with properties that should be ignored:
        node = AggregateSignNode({
            "depth": 0,
            "paa": np.array([1, 2, 3]),
            "breakpoints": [],
            "random_property": np.array([0, 0, 0])
        })
        self.assertEqual(node.depth, 0)
        self.assertTrue(np.array_equal(node.paa, np.array([1, 2, 3])))
        self.assertFalse(hasattr(node, "random_property"))

        # Test creation with properties of the wrong type:
        with self.assertRaises(ValueError):
            AggregateSignNode({
                "depth": "abc",
                "paa": np.array([1, 2, 3]),
                "breakpoints": []
            })
        with self.assertRaises(ValueError):
            AggregateSignNode({
                "depth": 0,
                "paa": None,
                "breakpoints": []
            })

        # Test creation with a missing property:
        with self.assertRaises(ValueError):
            AggregateSignNode({
                "depth": 0,
                "paa": np.array([1, 2, 3])
            })

    def test_equality(self):
        """Test the equivalence of AggregateSignNode instances."""
        # Test the equivalence of two identical AggregateSignNode instances:
        n1 = AggregateSignNode({
            "depth": 0,
            "paa": np.array([1, 2, 3]),
            "breakpoints": []
        })
        n2 = AggregateSignNode({
            "depth": 0,
            "paa": np.array([1, 2, 3]),
            "breakpoints": []
        })
        self.assertEqual(n1, n2)
        self.assertEqual(hash(n1), hash(n2))

        # Test equivalence with same mean signs:
        n1 = AggregateSignNode({
            "depth": 0,
            "paa": np.array([10, -2, 3]),
            "breakpoints": []
        })
        n2 = AggregateSignNode({
            "depth": 0,
            "paa": np.array([2, -3, 5]),
            "breakpoints": []
        })
        self.assertEqual(n1, n2)
        self.assertEqual(hash(n1), hash(n2))

        # Test equivalence with zero means, which count as positive:
        n1 = AggregateSignNode({
            "depth": 0,
            "paa": np.array([0, -2, 3]),
            "breakpoints": []
        })
        n2 = AggregateSignNode({
            "depth": 0,
            "paa": np.array([2, -3, 5]),
            "breakpoints": []
        })
        self.assertEqual(n1, n2)

        # Test equivalence with different mean signs:
        n1 = AggregateSignNode({
            "depth": 0,
            "paa": np.array([1, 2, 3]),
            "breakpoints": []
        })
        n2 = AggregateSignNode({
            "depth": 0,
            "paa": np.array([1, 2, -3]),
            "breakpoints": []
        })
        self.assertNotEqual(n1, n2)

        # Test equivalence with different depths:
        n1 = AggregateSignNode({
            "depth": 0,
            "paa": np.array([1, 2, 3]),
            "breakpoints": []
        })
        n2 = AggregateSignNode({
            "depth": 1,
            "paa": np.array([1, 2, 3]),
            "breakpoints": []
        })
        self.assertNotEqual(n1, n2)

        # Test equivalence with a node of a different type:
        n2 = SlopeSignNode({
            "depth": 0,
            "slopes": np.array([1, 2, 3]),
            "intercepts": np.array([1, 2, 3]),
            "breakpoints": []
        })
        self.assertNotEqual(n1, n2)

    def test_distance(self):
        n1 = AggregateSignNode({
            "depth": 0,
            "paa": np.array([1, -2, 3]),
            "breakpoints": []
        })

        n2 = AggregateSignNode({
            "depth": 0,
            "paa": np.array([-1, 2, -3]),
            "breakpoints": []
        })

        self.assertEqual(n1.distance(n1), 0.0)
        self.assertEqual(n2.distance(n2), 0.0)
        self.assertEqual(n1.distance(n2), 1.0)
        self.assertEqual(n2.distance(n1), 1.0)

        n2.paa = np.array([1, 2, -3])
        self.assertEqual(n1.distance(n2), 2/3)
        n2.paa = np.array([1, 2, 3])
        self.assertEqual(n1.distance(n2), 1/3)
        n2.paa = np.array([1, -2, 3])
        self.assertEqual(n1.distance(n2), 0.0)

    def test_comparability(self):
        n1 = AggregateSignNode({
            "depth": 0,
            "paa": np.array([1, -2, 3]),
            "breakpoints": []
        })

        # Test the distance to a node of a different type:
        n2 = SlopeSignNode({
            "depth": 0,
            "slopes": np.array([1, -2, 3]),
            "intercepts": np.array([1, 2, 3]),
            "breakpoints": []
        })
        with self.assertRaises(ValueError):
            n1.distance(n2)

        # Test the distance to a node of a different depth:
        n2 = AggregateSignNode({
            "depth": 1,
            "paa": np.array([1, -2, 3]),
            "breakpoints": []
        })
        with self.assertRaises(ValueError):
            n1.distance(n2)
