import unittest

import numpy as np

from pbsf.nodes import SumNode


class TestSumNode(unittest.TestCase):
    def test_distance(self):
        """Test the distance calculation between SumNode instances."""
        n1 = SumNode({
            "depth": 0,
            "sums": np.array([100, 200, 300]),
            "distance_threshold": lambda depth: 10.0
        })

        n2 = SumNode({
            "depth": 0,
            "sums": np.array([115, 195, 300]),
            "distance_threshold": lambda depth: 10.0
        })

        self.assertEqual(n1.distance(n1), 0.0)
        self.assertEqual(n2.distance(n2), 0.0)
        self.assertAlmostEqual(n1.distance(n2), 20/3)
        self.assertAlmostEqual(n2.distance(n1), 20/3)

        n2.sums = np.array([300, 195, 115])
        self.assertEqual(n1.distance(n2), 130.0)
        self.assertEqual(n1.distance(n2), 130.0)

    def test_equality(self):
        """Test the equivalence of SumNode instances."""
        n1 = SumNode({
            "depth": 0,
            "sums": np.array([100, 200, 300]),
            "distance_threshold": lambda depth: 10.0
        })

        n2 = SumNode({
            "depth": 0,
            "sums": np.array([115, 195, 300]),
            "distance_threshold": lambda depth: 10.0
        })

        self.assertEqual(n1, n2)
        self.assertEqual(n2, n1)

        threshold = 0.1
        n1.distance_threshold = threshold
        n2.distance_threshold = threshold

        self.assertNotEqual(n1, n2)
        self.assertNotEqual(n2, n1)
