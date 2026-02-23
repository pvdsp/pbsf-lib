import unittest

import numpy as np

from pbsf.algorithms import hpm
from pbsf.nodes import StructuralProminenceNode


class TestHPM(unittest.TestCase):
    def test_hpm(self):
        train = np.sin(np.arange(0, 10, 0.1))
        test = np.sin(np.arange(0, 10, 0.1))
        test[10:15] = np.random.random(5)

        params = {
            'max_depth': lambda _: 2,
            'pieces': lambda d: 2 ** d,
            'node_type': StructuralProminenceNode,
            'segmenter_params': {
                'window_size': 5,
            },
            'node_params': {
                'structural_threshold': lambda _: 0.1,
                'prominence_threshold': lambda _: 0.1,
            }
        }
        scores = hpm(train, test, params)
        self.assertEqual(len(scores), len(test))


if __name__ == '__main__':
    unittest.main()
