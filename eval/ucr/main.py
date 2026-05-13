"""Benchmark configuration for UCR anomaly detection evaluation."""

import numpy as np
from benchmark import evaluate_configurations

from pbsf.algorithms import matrix_profile
from pbsf.discretisers import PiecewiseLinear
from pbsf.nodes import PLANode
from pbsf.segmenters import SlidingWindow

algorithms = [
    {
        "function": matrix_profile,
        "name": "MatrixProfile",
        "select_anomaly": lambda s: np.argmax(s),
        "segmenter": SlidingWindow,
        "segmenter_params": {
            "window_size": 200,
            "step_size": 5,
            "autocorrelation": True,
        },
        "discretiser": PiecewiseLinear,
        "discretiser_params": {
            "node_type": PLANode,
        },
        "node_params": {
            "distance_threshold": lambda depth: 5.0,
        },
    },
]

if __name__ == "__main__":
    evaluate_configurations(algorithms)
