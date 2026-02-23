import unittest

import numpy as np

from pbsf.nodes import SlopeSignNode


class TestSlopeSignNode(unittest.TestCase):
    def test_creation(self):
        """Test the creation of a SlopeSignNode instance."""
        # Test the creation of a regular SlopeSignNode instance:
        node = SlopeSignNode({
            "depth": 0,
            "slopes": np.array([1, 2, 3]),
            "intercepts": np.array([1, 2, 3]),
            "breakpoints": []
        })
        self.assertEqual(node.depth, 0)
        self.assertTrue(np.array_equal(node.slopes, np.array([1, 2, 3])))
        self.assertTrue(np.array_equal(node.intercepts, np.array([1, 2, 3])))

        # Test creation with properties that should be ignored:
        node = SlopeSignNode({
            "depth": 0,
            "slopes": np.array([1, 2, 3]),
            "intercepts": np.array([1, 2, 3]),
            "breakpoints": [],
            "random_property": np.array([0, 0, 0])
        })
        self.assertEqual(node.depth, 0)
        self.assertTrue(np.array_equal(node.slopes, np.array([1, 2, 3])))
        self.assertTrue(np.array_equal(node.intercepts, np.array([1, 2, 3])))
        self.assertFalse(hasattr(node, "random_property"))

        # Test creation with properties of the wrong type:
        with self.assertRaises(ValueError):
            SlopeSignNode({
                "depth": "abc",
                "slopes": True,
                "intercepts": np.array([1, 2, 3]),
                "breakpoints": []
            })
        with self.assertRaises(ValueError):
            SlopeSignNode({
                "depth": 0,
                "slopes": None,
                "intercepts": np.array([1, 2, 3]),
                "breakpoints": []
            })

    def test_equality(self):
        """Test the equivalence of SlopeSignNode instances."""
        # Test the equivalence of two identical SlopeSignNode instances:
        n1 = SlopeSignNode({
            "depth": 0,
            "slopes": np.array([1, 2, 3]),
            "intercepts": np.array([1, 2, 3]),
            "breakpoints": []
        })
        n2 = SlopeSignNode({
            "depth": 0,
            "slopes": np.array([1, 2, 3]),
            "intercepts": np.array([1, 2, 3]),
            "breakpoints": []
        })
        self.assertEqual(n1, n2)

        # Test equivalence with same slope signs:
        n1 = SlopeSignNode({
            "depth": 0,
            "slopes": np.array([10, -2, 3]),
            "intercepts": np.array([1, 2, 3]),
            "breakpoints": []
        })
        n2 = SlopeSignNode({
            "depth": 0,
            "slopes": np.array([2, -3, 5]),
            "intercepts": np.array([1, 2, 3]),
            "breakpoints": []
        })
        self.assertEqual(n1, n2)

        # Test equivalence with different slope signs:
        n1 = SlopeSignNode({
            "depth": 0,
            "slopes": np.array([1, 2, 3]),
            "intercepts": np.array([1, 2, 3]),
            "breakpoints": []
        })
        n2 = SlopeSignNode({
            "depth": 1,
            "slopes": np.array([1, 2, 3]),
            "intercepts": np.array([1, 2, 3]),
            "breakpoints": []
        })
        self.assertNotEqual(n1, n2)

        # Test equivalence with different depths:
        n1 = SlopeSignNode({
            "depth": 0,
            "slopes": np.array([1, 2, 3]),
            "intercepts": np.array([1, 2, 3]),
            "breakpoints": []
        })
        n2 = SlopeSignNode({
            "depth": 1,
            "slopes": np.array([1, 2, 3]),
            "intercepts": np.array([1, 2, 3]),
            "breakpoints": []
        })
        self.assertNotEqual(n1, n2)

    def test_distance(self):
        n1 = SlopeSignNode({
            "depth": 0,
            "slopes": np.array([1, -2, 3]),
            "intercepts": np.array([1, 2, 3]),
            "breakpoints": []
        })

        n2 = SlopeSignNode({
            "depth": 0,
            "slopes": np.array([-1, 2, -3]),
            "intercepts": np.array([1, 2, 3]),
            "breakpoints": []
        })

        self.assertEqual(n1.distance(n1), 0.0)
        self.assertEqual(n2.distance(n2), 0.0)
        self.assertEqual(n1.distance(n2), 1.0)
        self.assertEqual(n2.distance(n1), 1.0)

        n2.slopes = np.array([1, 2, -3])
        self.assertEqual(n1.distance(n2), 2/3)
        n2.slopes = np.array([1, 2, 3])
        self.assertEqual(n1.distance(n2), 1/3)
        n2.slopes = np.array([1, -2, 3])
        self.assertEqual(n1.distance(n2), 0.0)
