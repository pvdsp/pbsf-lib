"""Matrix Profile of test sequences to training sequences."""

import math

import numpy as np

from pbsf.chains.base import Chain
from pbsf.discretisers import PiecewiseLinear
from pbsf.models import PatternTree
from pbsf.models.base import Model
from pbsf.nodes import PLANode
from pbsf.segmenters import SlidingWindow


def nn_approximate(model: Model, chain: Chain, drop_ratio: float = 0.6):
    """
    Find the approximate distance of a chain to its nearest neighbour in the model.

    Iterates through levels of the chain, pruning candidates at each level
    by keeping only the closest fraction (determined by drop_ratio). At the
    final level, returns the mean distance to the surviving candidates.

    Parameters
    ----------
    model : Model
        A trained model supporting get_level, get_node, and get_related.
    chain : Chain
        A discretised chain to find the nearest neighbour for.
    drop_ratio : float
        Fraction of candidates to discard at each level (default 0.6).

    Returns
    -------
    float
        The approximate nearest-neighbour distance, or math.inf if no
        candidates are found.
    """
    if not 0 <= drop_ratio < 1:
        msg = f"drop_ratio must be in [0, 1), got {drop_ratio}"
        raise ValueError(msg)

    candidates = model.get_level(0)
    if not candidates:
        return math.inf

    for level, node in enumerate(chain):
        distances = [
            (candidate, node.distance(model.get_node(candidate)))
            for candidate in candidates
        ]

        # Prune candidates by discarding candidates with highest distance
        distances.sort(key=lambda x: x[1])
        keep = max(1, int(len(distances) * (1 - drop_ratio)))
        distances = distances[:keep]

        # Proceed to finer granularity with remaining candidates
        if level < chain.length - 1:
            candidates = set()
            for candidate, _ in distances:
                candidates |= model.get_related(candidate, level + 1)
            if not candidates:
                return math.inf
        else:
            return sum(d for _, d in distances) / len(distances)
    return math.inf


def matrix_profile(train: np.ndarray, test: np.ndarray, parameters: dict):
    """
    Compute approximate nearest-neighbour distance from test to training subsequences.

    Segments and discretises both sequences, builds a model from the training
    chains, then scores each test chain against the model. Overlapping window
    scores are averaged per point.

    Parameters
    ----------
    train : np.ndarray
        Training time series.
    test : np.ndarray
        Test time series.
    parameters : dict
        Algorithm configuration. Supported keys:

        - ``segmenter`` : Segmenter class (default ``SlidingWindow``).
        - ``segmenter_params`` : dict passed to the segmenter
          (default ``{'window_size': 200}``).
        - ``discretiser`` : Discretiser class (default ``PiecewiseLinear``).
        - ``discretiser_params`` : dict passed to the discretiser.
        - ``node_params`` : dict passed to the node type via the discretiser.
        - ``model`` : Model class (default ``PatternTree``).
        - ``model_params`` : dict passed to the model.
        - ``drop_ratio`` : float, fraction of candidates to discard per level
          in ``nn_approximate`` (default 0.6).
        - ``filter_max_overlap`` : bool, if True return only the points with
          maximum window overlap (default False).

    Returns
    -------
    np.ndarray
        Per-point averaged nearest-neighbour distances. If
        ``filter_max_overlap`` is True, returns a tuple
        ``(indices, scores)`` instead.
    """
    segmenter_type = parameters.get("segmenter") or SlidingWindow
    discretiser_type = parameters.get("discretiser") or PiecewiseLinear
    model_type = parameters.get("model") or PatternTree

    segmenter_params = dict(parameters.get("segmenter_params", {}))
    segmenter_params.setdefault("window_size", 200)

    node_params = dict(parameters.get("node_params", {}))
    if node_params.get("distance_threshold") is None:
        node_params["distance_threshold"] = lambda depth: 5.0

    discretiser_params = dict(parameters.get("discretiser_params", {}))
    if discretiser_params.get("max_depth") is None:
        discretiser_params["max_depth"] = lambda d: int(np.floor(np.log(len(d))))
    if discretiser_params.get("frames") is None:
        discretiser_params["frames"] = lambda depth: 2 ** depth
    if discretiser_params.get("node_type") is None:
        discretiser_params["node_type"] = PLANode
    discretiser_params["node_params"] = node_params
    model_params = parameters.get("model_params")
    if model_params is None:
        model_params = {}
    drop_ratio = parameters.get("drop_ratio", 0.6)

    segmenter = segmenter_type(params=segmenter_params)
    discretiser = discretiser_type(params=discretiser_params)
    model = model_type(params=model_params)

    training_chains = [discretiser.discretise(s) for s in segmenter.segment(train)]
    testing_chains = [discretiser.discretise(s) for s in segmenter.segment(test)]

    model.learn(training_chains)

    counts = np.zeros(len(test))
    scores = np.zeros(len(test))

    for i, chain in enumerate(testing_chains):
        dist = nn_approximate(model, chain, drop_ratio=drop_ratio)
        start_point = i * segmenter.step_size
        end_point = min(
            start_point + segmenter.window_size, len(test)
        )
        for point in range(start_point, end_point):
            counts[point] += 1
            scores[point] += dist

    filter_max_overlap = parameters.get("filter_max_overlap", False)
    if filter_max_overlap:
        max_count = counts.max()
        if max_count == 0:
            return np.array([], dtype=int), np.array([])
        mask = counts == max_count
        x = np.where(mask)[0]
        return x, scores[mask] / max_count

    return np.divide(
        scores, counts,
        out=np.zeros_like(scores),
        where=counts != 0,
    )
