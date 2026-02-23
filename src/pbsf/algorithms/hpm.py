import numpy as np

from pbsf.discretisers import PiecewiseLinear
from pbsf.models import PatternTree
from pbsf.nodes import StructuralProminenceNode
from pbsf.segmenters import SlidingWindow


def hpm(train: np.ndarray, test: np.ndarray, parameters: dict) -> np.ndarray:
    """
    Implementation of the Hierarchical Pattern Matching Anomaly Detection algorithm.

    Parameters
    ----------
    train : np.ndarray
        Training data.
    test : np.ndarray
        Test data.
    parameters : dict
        Dictionary containing the parameters for the algorithm.
        Expected keys:

        - segmenter (type): Segmenter class to use. Default is SlidingWindow.
        - segmenter_params (dict): Parameters for the segmenter. Default is {'window_size': 200}.
        - discretiser (type): Discretiser class to use. Default is PiecewiseLinear.
        - discretiser_params (dict): Parameters for the discretiser. Expected keys:

          - max_depth (callable): Maximum depth of the tree given the data.
          - frames (callable): Number of segments given the depth.
          - node_type (type): The type of node to use for the tree.

        - model (type): Model class to use. Default is PatternTree.
        - model_params (dict): Parameters for the model (e.g., context_size for NestedWordSet).
        - node_params (dict): Extra parameters to pass to the node constructor,
          e.g., functions for the structural and prominence thresholds.

    Returns
    -------
    np.ndarray
        Anomaly scores for each point in the test data. Values closer to 1 indicate
        the pattern was seen during training (normal), values closer to 0 indicate
        anomalies.
    """
    segmenter = parameters.get("segmenter") or SlidingWindow
    discretiser = parameters.get("discretiser") or PiecewiseLinear
    model = parameters.get("model") or PatternTree
    model_params = parameters.get("model_params")

    model = model(params=model_params)
    segmenter = segmenter(params=parameters.get("segmenter_params") or {'window_size': 200})

    # Set up standard HPM StructuralProminenceNode parameters if missing:
    node_params = parameters.get("node_params") or {}
    node_params["structural_threshold"] = node_params.get("structural_threshold") or (lambda depth: 0.5)
    node_params["prominence_threshold"] = node_params.get("prominence_threshold") or (lambda depth: 0.5)

    # Set up standard HPM discretiser parameters if missing:
    discretiser_params = parameters.get("discretiser_params") or {}
    discretiser_params["max_depth"] = discretiser_params.get("max_depth") or (lambda d: int(np.floor(np.log(len(d)))))
    discretiser_params["frames"] = discretiser_params.get("frames") or (lambda depth: 2 ** depth)
    discretiser_params["node_type"] = discretiser_params.get("node_type") or StructuralProminenceNode
    discretiser_params["node_params"] = node_params

    discretiser = discretiser(
        params=discretiser_params,
    )

    training_chains = [discretiser.discretise(segment) for segment in segmenter.segment(train)]
    testing_chains = [discretiser.discretise(segment) for segment in segmenter.segment(test)]
    model.learn(training_chains)

    counts = np.zeros(len(test))
    scores = np.zeros(len(test))

    if model_params is not None and (context_size := model_params.get("context_size")) is not None:
        for i in range(0, len(testing_chains) - context_size + 1):
            contains_pattern = model.contains(testing_chains[i:i + context_size])
            # Context spans multiple segments
            start_point = i * segmenter.step_size
            end_point = min((i + context_size - 1) * segmenter.step_size + segmenter.window_size, len(test))
            for point in range(start_point, end_point):
                counts[point] += 1
                scores[point] += float(contains_pattern)
    else:
        for i, chain in enumerate(testing_chains):
            contains_pattern = model.contains(chain)
            start_point = i * segmenter.step_size
            end_point = min(start_point + segmenter.window_size, len(test))
            for point in range(start_point, end_point):
                counts[point] += 1
                scores[point] += float(contains_pattern)

    # Avoid division by zero: set score to 0 where counts is 0
    return np.divide(scores, counts, out=np.zeros_like(scores), where=counts != 0)
