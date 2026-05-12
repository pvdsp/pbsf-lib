import unittest

import numpy as np

from pbsf.segmenters import SlidingWindow


class TestSlidingWindow(unittest.TestCase):
    def setUp(self):
        """Set up common test data."""
        self.simple_data = np.array([1, 2, 3, 4, 5])
        self.periodic_data = np.sin(np.linspace(0, 8 * np.pi, 400))
        # Set seed for reproducible random tests
        np.random.seed(42)
        self.random_data = np.random.normal(0, 1, 400)

    def test_creation(self):
        """Test the creation of a SlidingWindow instance."""
        segmenter = SlidingWindow({'window_size': 3})
        self.assertEqual(segmenter.window_size, 3)
        self.assertEqual(segmenter.step_size, 1)
        self.assertFalse(segmenter.autocorrelation)
        self.assertFalse(segmenter.differentiation)

        # Test with a custom step size
        segmenter = SlidingWindow({'window_size': 3, 'step_size': 2})
        self.assertEqual(segmenter.window_size, 3)
        self.assertEqual(segmenter.step_size, 2)
        self.assertFalse(segmenter.autocorrelation)
        self.assertFalse(segmenter.differentiation)

        # Test with autocorrelation and differentiation enabled
        segmenter = SlidingWindow({'window_size': 3, 'step_size': 2,
                                   'autocorrelation': True,
                                   'differentiation': True})
        self.assertEqual(segmenter.window_size, None)
        self.assertEqual(segmenter.window_fallback, 3)
        self.assertEqual(segmenter.step_size, 2)
        self.assertTrue(segmenter.autocorrelation)
        self.assertTrue(segmenter.differentiation)

        # Test invalid parameters
        with self.assertRaises(ValueError):
            SlidingWindow({'window_size': 0})
        with self.assertRaises(ValueError):
            SlidingWindow({'window_size': -1})
        with self.assertRaises(ValueError):
            SlidingWindow({'autocorrelation': True})
        with self.assertRaises(ValueError):
            segmenter = SlidingWindow({'window_size': 3, 'step_size': 0})
        with self.assertRaises(ValueError):
            segmenter = SlidingWindow({'window_size': 3, 'step_size': -1})
        with self.assertRaises(ValueError):
            segmenter = SlidingWindow({'window_size': 3, 'step_size': -5})
        with self.assertRaises(AttributeError):
            SlidingWindow(None)
        with self.assertRaises(ValueError):
            SlidingWindow({})

    def test_segment(self):
        """Test the segment method of a SlidingWindow instance."""
        segmenter = SlidingWindow({'window_size': 3, 'z_normalisation': False})
        segments = segmenter.segment(self.simple_data)
        expected = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        np.testing.assert_array_equal(segments, expected)
        segmenter = SlidingWindow({
            'window_size': 3,
            'step_size': 2,
            'z_normalisation': False,
        })
        segments = segmenter.segment(self.simple_data)
        expected = np.array([[1, 2, 3], [3, 4, 5]])
        np.testing.assert_array_equal(segments, expected)
        with self.assertRaises(ValueError):
            segmenter.segment(np.array([1, 2]))
        with self.assertRaises(ValueError):
            segmenter.segment(np.array([1]))
        with self.assertRaises(ValueError):
            segmenter.segment(np.array([]))
        with self.assertRaises(ValueError):
            segmenter.segment(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_differentiation(self):
        """Test the differentiation option of SlidingWindow."""
        data = np.array([1, 2, 4, 7, 11])
        segmenter = SlidingWindow({
            'window_size': 3,
            'differentiation': True,
            'z_normalisation': False,
        })
        segments = segmenter.segment(data)
        expected = np.array([[1, 2, 3], [2, 3, 4]])
        np.testing.assert_array_equal(segments, expected)

    def test_autocorrelation(self):
        """Test the autocorrelation option of SlidingWindow."""
        # Test with periodic data (sine wave with 8 periods over 400 points)
        segmenter = SlidingWindow({'window_size': 20, 'autocorrelation': True})
        segmenter.segment(self.periodic_data)
        self.assertEqual(segmenter.window_size, 101)

        # Test with random data (no periodicity, seed=42 for reproducibility)
        # Expected: autocorrelation < 0.5 threshold, falls back to window_size=20
        segmenter = SlidingWindow({'window_size': 20, 'autocorrelation': True})
        segmenter.segment(self.random_data)
        self.assertEqual(segmenter.window_size, 20)

    def test_autocorrelation_with_differentiation(self):
        """Test combining autocorrelation with differentiation."""
        # Test that both options can be used together
        segmenter = SlidingWindow({
            'window_size': 20,
            'autocorrelation': True,
            'differentiation': True
        })
        # Differentiation reduces data length by 1
        segments = segmenter.segment(self.periodic_data)
        # Verify that window_size was set via autocorrelation
        self.assertIsNotNone(segmenter.window_size)
        # Verify segments were created
        self.assertGreater(len(segments), 0)

    def test_z_normalisation(self):
        """Test that z-score normalisation is applied to each segment."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        segmenter = SlidingWindow({'window_size': 3})
        segments = segmenter.segment(data)

        # Each segment should have mean ≈ 0 and std ≈ 1
        for segment in segments:
            np.testing.assert_almost_equal(np.mean(segment), 0.0)
            np.testing.assert_almost_equal(np.std(segment), 1.0)

        # Segments with identical relative shape should be equal
        # [1, 2, 3] and [4, 5, 6] both normalise to the same values
        np.testing.assert_array_almost_equal(segments[0], segments[-1])

        # Constant segment should become all zeros
        constant_data = np.array([5.0, 5.0, 5.0, 5.0])
        segmenter = SlidingWindow({'window_size': 3})
        segments = segmenter.segment(constant_data)
        for segment in segments:
            np.testing.assert_array_equal(segment, np.zeros(3))

    def test_edge_cases(self):
        """Test edge cases for sliding window segmentation."""
        # Test data length exactly equal to window size (should produce 1 segment)
        segmenter = SlidingWindow({'window_size': 5, 'z_normalisation': False})
        segments = segmenter.segment(self.simple_data)
        self.assertEqual(segments.shape, (1, 5))
        np.testing.assert_array_equal(segments[0], self.simple_data)

        # Test window size of 1 (should produce n segments of length 1)
        segmenter = SlidingWindow({'window_size': 1, 'z_normalisation': False})
        segments = segmenter.segment(self.simple_data)
        expected = np.array([[1], [2], [3], [4], [5]])
        np.testing.assert_array_equal(segments, expected)

        # Test large step size (larger than window)
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        segmenter = SlidingWindow({
            'window_size': 3,
            'step_size': 5,
            'z_normalisation': False,
        })
        segments = segmenter.segment(data)
        expected = np.array([[1, 2, 3], [6, 7, 8]])
        np.testing.assert_array_equal(segments, expected)

        # Test step size equal to window size (non-overlapping windows)
        segmenter = SlidingWindow({
            'window_size': 2,
            'step_size': 2,
            'z_normalisation': False,
        })
        segments = segmenter.segment(np.array([1, 2, 3, 4, 5, 6]))
        expected = np.array([[1, 2], [3, 4], [5, 6]])
        np.testing.assert_array_equal(segments, expected)
