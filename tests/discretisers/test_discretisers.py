import unittest
import warnings

import numpy as np

from pbsf.chains import Chain
from pbsf.discretisers import PiecewiseLinear, SymbolicAggregate
from pbsf.nodes import SAXNode, SlopeSignNode, StructuralProminenceNode


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
        chain = discretiser.discretise(segment)
        self.assertIsInstance(chain, Chain)
        slopes = chain[0].slopes
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
        chain = discretiser.discretise(segment)
        self.assertIsInstance(chain, Chain)
        self.assertEqual(len(chain), discretiser.max_depth(segment))
        self.assertEqual(chain.length, discretiser.max_depth(segment))
        for idx, node in enumerate(chain):
            self.assertEqual(node.depth, idx)
            self.assertEqual(node.std, np.std(segment))
            self.assertEqual(len(node.slopes), 2 ** idx)


class TestSymbolicAggregate(unittest.TestCase):
    def setUp(self):
        self.sax = SymbolicAggregate({
            "max_depth": lambda _: 1,
            "frames": lambda _: 4,
            "alphabet_size": 4,
            "node_type": SAXNode,
            "node_params": {
                "distance_threshold": lambda _: 0.1,
            },
        })

    def test_unnormalised_segment_warns(self):
        """Test that a UserWarning is raised for unnormalised input."""
        segment = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        with self.assertWarns(UserWarning):
            self.sax.discretise(segment)

    def test_normalised_segment_no_warning(self):
        """Test that no warning is raised for z-normalised input."""
        segment = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        segment = (segment - np.mean(segment)) / np.std(segment)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.sax.discretise(segment)

