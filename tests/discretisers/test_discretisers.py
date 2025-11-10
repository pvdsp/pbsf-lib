import unittest
import numpy as np
from pbsf.nodes import SlopeSignNode, StructuralProminenceNode
from pbsf.discretisers import PiecewiseLinear


class TestPiecewiseLinear(unittest.TestCase):
    def test_creation(self):
        discretiser = PiecewiseLinear({
            "max_depth": lambda _: 1,
            "frames": lambda depth: 1,
            "node_type": SlopeSignNode,
            "node_params": {}
        })
        self.assertEqual(discretiser.max_depth(np.array([1, 2, 3])), 1)
        self.assertEqual(discretiser.frames(0), 1)
        self.assertEqual(discretiser.node_type, SlopeSignNode)
        self.assertEqual(discretiser.node_params, {})

        discretiser = PiecewiseLinear({
            "max_depth": lambda _: 2,
            "frames": lambda depth: 2,
            "node_type": SlopeSignNode,
            "node_params": {
                "threshold": 0.5
            }
        })
        self.assertEqual(discretiser.max_depth(np.array([1, 2, 3])), 2)
        self.assertEqual(discretiser.frames(0), 2)
        self.assertEqual(discretiser.node_type, SlopeSignNode)
        self.assertEqual(discretiser.node_params, {"threshold": 0.5})

    def test_discretise(self):
        discretiser = PiecewiseLinear({
            "max_depth": lambda _: 1,
            "frames": lambda _: 8,
            "node_type": SlopeSignNode,
            "node_params": {}
        })
        segment = np.sin(np.linspace(0, 2 * np.pi, 200))
        nodes = discretiser.discretise(segment)
        slopes = nodes[0].slopes
        signs = np.array([1, 1, -1, -1, -1, -1, 1, 1])
        self.assertTrue(np.all((slopes >= 0) == (signs >= 0)))

        discretiser = PiecewiseLinear({
            "max_depth": lambda _: 2,
            "frames": lambda depth: 2 ** depth,
            "node_type": StructuralProminenceNode,
            "node_params": {
                "structural_threshold": lambda _: 0.1,
                "prominence_threshold": lambda _: 0.1
            }
        })
        data = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            discretiser.discretise(data)
        segment = np.arange(100)
        nodes = discretiser.discretise(segment)
        self.assertEqual(len(nodes), discretiser.max_depth(segment))
        for idx, node in enumerate(nodes):
            self.assertEqual(node.depth, idx)
            self.assertEqual(node.std, np.std(segment))
            self.assertEqual(len(node.slopes), 2 ** idx)

