import math
import unittest

import numpy as np

from pbsf.nodes import PAANode


class TestPAANode(unittest.TestCase):
    def test_distance(self):
        n1 = PAANode({
            "depth": 0,
            "segment_length": 10,
            "nr_of_frames": 5,
            "breakpoints": [(0, 2), (2, 4), (4, 6)],
            "paa": np.array([1, 2, 3]),
            "distance_threshold": lambda depth: 0.5
        })

        n2 = PAANode({
            "depth": 0,
            "segment_length": 10,
            "nr_of_frames": 5,
            "breakpoints": [(0, 2), (2, 4), (4, 6)],
            "paa": np.array([1, 2, 4]),
            "distance_threshold": lambda depth: 0.5
        })

        self.assertAlmostEqual(n1.distance(n1), 0.0)
        self.assertAlmostEqual(n2.distance(n2), 0.0)
        self.assertAlmostEqual(n1.distance(n2), np.sqrt(2))

        n2.paa = np.array([10, 5, 9])
        self.assertAlmostEqual(n1.distance(n2),
                               np.sqrt(2) * (math.sqrt((sum((n1.paa - n2.paa) ** 2)))))
