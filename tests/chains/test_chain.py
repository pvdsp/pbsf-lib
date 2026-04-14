import unittest

import numpy as np

from pbsf.chains import Chain
from pbsf.nodes import SumNode


def _make_sum_node(depth, sums):
    return SumNode({
        "depth": depth,
        "sums": np.array(sums),
        "distance_threshold": lambda _: 0.5,
    })


class TestChain(unittest.TestCase):
    def test_creation(self):
        nodes = [_make_sum_node(0, [10.0]), _make_sum_node(1, [5.0, 5.0])]
        chain = Chain(nodes)
        self.assertEqual(chain.length, 2)
        self.assertEqual(len(chain), 2)
        self.assertIsInstance(chain.nodes, tuple)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            Chain([])

    def test_non_node_raises(self):
        with self.assertRaises(ValueError):
            Chain(["not", "nodes"])

    def test_mixed_types_raises(self):
        from pbsf.nodes import SlopeSignNode
        n1 = _make_sum_node(0, [10.0])
        n2 = SlopeSignNode({
            "depth": 0,
            "slopes": np.array([1.0]),
            "intercepts": np.array([0.0]),
            "breakpoints": [(0, 1)],
        })
        with self.assertRaises(ValueError):
            Chain([n1, n2])

    def test_iteration(self):
        nodes = [_make_sum_node(0, [10.0]), _make_sum_node(1, [5.0, 5.0])]
        chain = Chain(nodes)
        result = list(chain)
        self.assertEqual(len(result), 2)

    def test_indexing(self):
        nodes = [_make_sum_node(0, [10.0]), _make_sum_node(1, [5.0, 5.0])]
        chain = Chain(nodes)
        self.assertIsInstance(chain[0], SumNode)
        self.assertIsInstance(chain[1], SumNode)

    def test_repr(self):
        chain = Chain([_make_sum_node(0, [10.0])])
        self.assertIn("SumNode", repr(chain))
        self.assertIn("length=1", repr(chain))

    def test_distance_same_chain(self):
        nodes = [_make_sum_node(0, [10.0]), _make_sum_node(1, [5.0, 5.0])]
        chain = Chain(nodes)
        self.assertAlmostEqual(chain.distance(chain), 0.0)

    def test_distance_different_chains(self):
        c1 = Chain([_make_sum_node(0, [10.0]), _make_sum_node(1, [5.0, 5.0])])
        c2 = Chain([_make_sum_node(0, [20.0]), _make_sum_node(1, [10.0, 10.0])])
        d = c1.distance(c2)
        self.assertGreater(d, 0.0)

    def test_distance_length_mismatch_raises(self):
        c1 = Chain([_make_sum_node(0, [10.0])])
        c2 = Chain([_make_sum_node(0, [10.0]), _make_sum_node(1, [5.0, 5.0])])
        with self.assertRaises(ValueError):
            c1.distance(c2)

    def test_distance_custom_fn(self):
        custom = lambda a, b: 42.0
        c1 = Chain([_make_sum_node(0, [10.0])], distance_fn=custom)
        c2 = Chain([_make_sum_node(0, [20.0])], distance_fn=custom)
        self.assertEqual(c1.distance(c2), 42.0)

    def test_distance_different_fn_raises(self):
        c1 = Chain([_make_sum_node(0, [10.0])], distance_fn=lambda a, b: 1.0)
        c2 = Chain([_make_sum_node(0, [10.0])], distance_fn=lambda a, b: 1.0)
        with self.assertRaises(ValueError):
            c1.distance(c2)

    def test_mean_distance(self):
        c1 = Chain(
            [_make_sum_node(0, [10.0]), _make_sum_node(1, [5.0, 5.0])],
            distance_fn=Chain._mean_distance,
        )
        c2 = Chain(
            [_make_sum_node(0, [10.0]), _make_sum_node(1, [5.0, 5.0])],
            distance_fn=Chain._mean_distance,
        )
        self.assertAlmostEqual(c1.distance(c2), 0.0)

    def test_not_hashable(self):
        c = Chain([_make_sum_node(0, [10.0])])
        with self.assertRaises(TypeError):
            hash(c)

    def test_weighted_distance_coarse_matters_more(self):
        # Two chains that differ only at depth 0 (coarse)
        c1 = Chain([_make_sum_node(0, [0.0]), _make_sum_node(1, [0.0, 0.0])])
        c2 = Chain([_make_sum_node(0, [10.0]), _make_sum_node(1, [0.0, 0.0])])
        d_coarse = c1.distance(c2)

        # Two chains that differ only at depth 1 (fine)
        c3 = Chain([_make_sum_node(0, [0.0]), _make_sum_node(1, [0.0, 0.0])])
        c4 = Chain([_make_sum_node(0, [0.0]), _make_sum_node(1, [10.0, 0.0])])
        d_fine = c3.distance(c4)

        self.assertGreater(d_coarse, d_fine)

    def test_equality(self):
        c1 = Chain([_make_sum_node(0, [10.0])])
        c2 = Chain([_make_sum_node(0, [10.0])])
        c3 = Chain([_make_sum_node(0, [20.0])])
        self.assertEqual(c1, c2)
        self.assertNotEqual(c1, c3)
