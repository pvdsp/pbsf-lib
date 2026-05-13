import unittest

import numpy as np

from pbsf.algorithms import matrix_profile
from pbsf.algorithms.matrix_profile import nn_approximate
from pbsf.models import PatternTree
from pbsf.nodes import PLANode


class TestMatrixProfile(unittest.TestCase):
    """Tests for the matrix_profile algorithm."""

    def setUp(self):
        self.train = np.sin(np.arange(0, 10, 0.1))
        self.test = np.sin(np.arange(0, 10, 0.1))
        self.params = {
            "segmenter_params": {
                "window_size": 10,
            },
            "discretiser_params": {
                "max_depth": lambda _: 2,
                "frames": lambda d: 2 ** d,
                "node_type": PLANode,
            },
            "node_params": {
                "distance_threshold": lambda _: 5.0,
            },
        }

    def test_output_length_matches_test(self):
        scores = matrix_profile(self.train, self.test, self.params)
        self.assertEqual(len(scores), len(self.test))

    def test_output_is_ndarray(self):
        scores = matrix_profile(self.train, self.test, self.params)
        self.assertIsInstance(scores, np.ndarray)

    def test_scores_non_negative(self):
        scores = matrix_profile(self.train, self.test, self.params)
        self.assertTrue(np.all(scores >= 0))

    def test_missing_window_size_uses_default(self):
        train = np.sin(np.arange(0, 100, 0.1))
        test = np.sin(np.arange(0, 100, 0.1))
        params = {**self.params, "segmenter_params": {}}
        scores = matrix_profile(train, test, params)
        self.assertEqual(len(scores), len(test))

    def test_filter_max_overlap_returns_tuple(self):
        params = {**self.params, "filter_max_overlap": True}
        result = matrix_profile(self.train, self.test, params)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_filter_max_overlap_consistent_lengths(self):
        params = {**self.params, "filter_max_overlap": True}
        indices, scores = matrix_profile(self.train, self.test, params)
        self.assertEqual(len(indices), len(scores))

    def test_filter_max_overlap_indices_within_bounds(self):
        params = {**self.params, "filter_max_overlap": True}
        indices, _ = matrix_profile(self.train, self.test, params)
        if len(indices) > 0:
            self.assertTrue(np.all(indices >= 0))
            self.assertTrue(np.all(indices < len(self.test)))


class TestNNApproximate(unittest.TestCase):
    """Tests for the nn_approximate helper."""

    def setUp(self):
        self.train = np.sin(np.arange(0, 10, 0.1))
        self.params = {
            "segmenter_params": {
                "window_size": 10,
            },
            "discretiser_params": {
                "max_depth": lambda _: 2,
                "frames": lambda d: 2 ** d,
                "node_type": PLANode,
            },
            "node_params": {
                "distance_threshold": lambda _: 5.0,
            },
        }

    def _build_model_and_chains(self, data):
        from pbsf.discretisers import PiecewiseLinear
        from pbsf.segmenters import SlidingWindow

        segmenter = SlidingWindow(params=self.params["segmenter_params"])
        discretiser = PiecewiseLinear(params={
            **self.params["discretiser_params"],
            "node_params": self.params["node_params"],
        })
        chains = [discretiser.discretise(s) for s in segmenter.segment(data)]

        model = PatternTree(params={})
        model.learn(chains)
        return model, chains

    def test_returns_float(self):
        model, chains = self._build_model_and_chains(self.train)
        dist = nn_approximate(model, chains[0])
        self.assertIsInstance(dist, float)

    def test_returns_non_negative(self):
        model, chains = self._build_model_and_chains(self.train)
        dist = nn_approximate(model, chains[0])
        self.assertGreaterEqual(dist, 0.0)

    def test_drop_ratio_negative_raises(self):
        model, chains = self._build_model_and_chains(self.train)
        with self.assertRaises(ValueError):
            nn_approximate(model, chains[0], drop_ratio=-0.1)

    def test_drop_ratio_one_raises(self):
        model, chains = self._build_model_and_chains(self.train)
        with self.assertRaises(ValueError):
            nn_approximate(model, chains[0], drop_ratio=1.0)

    def test_drop_ratio_above_one_raises(self):
        model, chains = self._build_model_and_chains(self.train)
        with self.assertRaises(ValueError):
            nn_approximate(model, chains[0], drop_ratio=1.5)

    def test_drop_ratio_zero_accepted(self):
        model, chains = self._build_model_and_chains(self.train)
        dist = nn_approximate(model, chains[0], drop_ratio=0.0)
        self.assertIsInstance(dist, float)

    def test_empty_model_returns_inf(self):
        model = PatternTree(params={})
        model.learn([])
        # Build a chain from some data to query against the empty model
        _, chains = self._build_model_and_chains(self.train)
        dist = nn_approximate(model, chains[0])
        self.assertEqual(dist, float("inf"))


if __name__ == "__main__":
    unittest.main()
