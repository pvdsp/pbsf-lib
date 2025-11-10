import unittest

import numpy as np

from pbsf.nodes import StructuralProminenceNode


class TestStructuralProminenceNode(unittest.TestCase):
    def test_creation(self):
        """Test the creation of a StructuralProminenceNode"""
        node = StructuralProminenceNode({
            "depth": 0,
            "std": 0.5,
            "slopes": np.array([1, 2, 3]),
            "intercepts": np.array([0, 0, 0]),
            "breakpoints": [],
            "structural_threshold": lambda depth: 0.1,
            "prominence_threshold": lambda depth: 0.2
        })

        self.assertEqual(node.depth, 0)
        self.assertEqual(node.std, 0.5)
        self.assertTrue(np.array_equal(node.slopes, np.array([1, 2, 3])))
        self.assertTrue(np.array_equal(node.intercepts, np.array([0, 0, 0])))
        self.assertEqual(node.structural_threshold, 0.1)
        self.assertEqual(node.prominence_threshold, 0.2)

    def test_equality(self):
        """"""
        n1 = StructuralProminenceNode({
            "depth": 0,
            "std": 0.5,
            "slopes": np.array([1, 2, 3]),
            "intercepts": np.array([0, 0, 0]),
            "breakpoints": [],
            "structural_threshold": lambda depth: 0.1,
            "prominence_threshold": lambda depth: 0.2
        })
        n2 = StructuralProminenceNode({
            "depth": 0,
            "std": 0.5,
            "slopes": np.array([1, 2, 3]),
            "intercepts": np.array([0, 0, 0]),
            "breakpoints": [],
            "structural_threshold": lambda depth: 0.1,
            "prominence_threshold": lambda depth: 0.2
        })
        self.assertEqual(n1, n2)

        n1 = StructuralProminenceNode({
            "depth": 0,
            "std": 0.123,
            "slopes": np.array([0.1, 0.2, 0.6]),
            "intercepts": np.array([0.2, 0.4, 0.6]),
            "breakpoints": [],
            "structural_threshold": lambda depth: 0.1,
            "prominence_threshold": lambda depth: 0.2
        })
        n2 = StructuralProminenceNode({
            "depth": 0,
            "std": 0.130,
            "slopes": np.array([0.4, 0.2, 0.1]),
            "intercepts": np.array([0.6, 0.4, 0.2]),
            "breakpoints": [],
            "structural_threshold": lambda depth: 0.1,
            "prominence_threshold": lambda depth: 0.2
        })

        structural_actual = n1.structural_distance(n2)
        structural_expected = ((n1.slopes[0] - n2.slopes[0] + n1.intercepts[0] - n2.intercepts[0]) +
                               (n1.slopes[1] - n2.slopes[1] + n1.intercepts[1] - n2.intercepts[1]) +
                               (n1.slopes[2] - n2.slopes[2] + n1.intercepts[2] - n2.intercepts[2])) / 3
        prominence_actual = n1.prominence_distance(n2)
        prominence_expected = max(n1.std, n2.std) / min(n1.std, n2.std) - 1.0

        self.assertAlmostEqual(structural_actual, structural_expected)
        self.assertAlmostEqual(prominence_actual, prominence_expected)
        self.assertEqual(n1, n2)

    def test_distance(self):
        n1 = StructuralProminenceNode({
            "depth": 0,
            "std": 0.5,
            "slopes": np.array([1/2, 1/3, 1/4]),
            "intercepts": np.array([1, 2, 3]),
            "breakpoints": [],
            "structural_threshold": lambda depth: 0.1,
            "prominence_threshold": lambda depth: 0.2
        })

        n2 = StructuralProminenceNode({
            "depth": 0,
            "std": 0.5,
            "slopes": np.array([1/2, 2/3, 1/4]),
            "intercepts": np.array([1, 2, 3]),
            "breakpoints": [],
            "structural_threshold": lambda depth: 0.1,
            "prominence_threshold": lambda depth: 0.2
        })

        self.assertAlmostEqual(n1.distance(n1), 0.0)
        self.assertAlmostEqual(n2.distance(n2), 0.0)
        self.assertAlmostEqual(n1.distance(n2), 1/9)
        self.assertAlmostEqual(n2.distance(n1), 1/9)

        n2.intercepts = np.array([3, 2, 1])
        self.assertAlmostEqual(n2.distance(n2), 0.0)
        self.assertAlmostEqual(n1.distance(n2), 1/9)

        n2.intercepts = np.array([0, 0, 0])
        self.assertAlmostEqual(n1.distance(n2), 17/9)
        self.assertAlmostEqual(n2.distance(n1), 17/9)
