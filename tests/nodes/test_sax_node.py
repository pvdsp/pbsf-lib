import unittest

import numpy as np

from pbsf.nodes import SAXNode


class TestSAXNode(unittest.TestCase):
    def test_alphabet_bounds(self):
        # test if dist throws error on -1 or alphabet_size
        node = SAXNode({
            "depth": 0,
            "segment_length": 10,
            "nr_of_frames": 3,
            "alphabet_size": 4,
            "breakpoints": [],
            "sax": np.array([0, 1, 2]),
            "cut_points": np.array([-0.67448975,  0.        ,  0.67448975]),
            "distance_threshold": lambda depth: 0.5
        })

        for symbol in (-1, node.alphabet_size):
            with self.assertRaises(ValueError):
                node._dist(0, symbol)
            with self.assertRaises(ValueError):
                node._dist(symbol, 0)

        # the first and last symbols of the alphabet are themselves valid
        self.assertAlmostEqual(node._dist(0, 0), 0.0)
        self.assertAlmostEqual(node._dist(0, node.alphabet_size - 1), 1.35,
                               places=2)

    def test_distance(self):
        n1 = SAXNode({
            "depth": 0,
            "segment_length": 10,
            "nr_of_frames": 3,
            "alphabet_size": 4,
            "breakpoints": [],
            "sax": np.array([0, 1, 2]),
            "cut_points": np.array([-0.67448975,  0.        ,  0.67448975]),
            "distance_threshold": lambda depth: 0.5
        })

        n2 = SAXNode({
            "depth": 0,
            "segment_length": 10,
            "nr_of_frames": 3,
            "alphabet_size": 4,
            "breakpoints": [],
            "sax": np.array([1, 2, 3]),
            "cut_points": np.array([-0.67448975,  0.        ,  0.67448975]),
            "distance_threshold": lambda depth: 0.5
        })

        self.assertEqual(n1.distance(n1), 0.0)
        self.assertEqual(n2.distance(n2), 0.0)

        self.assertAlmostEqual(n1._dist(0, 0), 0.0)
        self.assertAlmostEqual(n1._dist(0, 1), 0.0)
        self.assertAlmostEqual(n1._dist(0, 2), 0.67, places=2)

        self.assertEqual(n1.distance(n2), 0.0)
        self.assertEqual(n2.distance(n1), 0.0)

        n2.sax = np.array([2, 1, 2])
        expected = np.sqrt(n1.n / n1.frames) * n1._dist(0, 2)
        self.assertAlmostEqual(n1.distance(n2), expected)
        n2.sax = np.array([2, 2, 3])
        self.assertEqual(n1.distance(n2), np.sqrt(n1.n / n1.frames) * n1._dist(0, 2))
        n2.sax = np.array([1, 1, 1])
        self.assertAlmostEqual(n1.distance(n2), 0.0)
        n2.sax = np.array([0, 0, 0])
        self.assertEqual(n1.distance(n2), np.sqrt(n1.n / n1.frames) * (n1._dist(0, 0) +
                                                                       n1._dist(1, 0) +
                                                                       n1._dist(2, 0)))
