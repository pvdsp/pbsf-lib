import os
import time
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from pbsf.algorithms import hpm
from pbsf.discretisers import PiecewiseLinear
from pbsf.models import PatternGraph
from pbsf.nodes import StructuralProminenceNode
from pbsf.segmenters import SlidingWindow


def get_ucr(path: str):
    """
    Generator that returns UCR datasets from the specified path, sorted by file size.

    Parameters
    ----------
    path : str
        Path to the directory containing UCR dataset files.

    Yields
    ------
    identifier : str
        Identifier of the specific UCR time series.
    train : np.ndarray
        Training data.
    test : np.ndarray
        Testing data.
    anomaly_range : tuple[int, int]
        Start and end indices of the ground-truth anomaly in the test data.
    """
    files = sorted(
        os.listdir(path),
        key=lambda x: os.path.getsize(os.path.join(path, x)),
    )
    for file in files:
        if file.endswith(".txt"):
            # Extract information on UCR time series from filename:
            split_filename = file.split("_")  # Split filename by underscores
            identifier = split_filename[0]  # Identifier of the UCR time series
            training = int(split_filename[-3])  # Training size of the UCR time series
            # Start/end of ground-truth anomaly in test data
            start_anomaly = int(split_filename[-2])
            end_anomaly = int(split_filename[-1].split(".")[0])

            # Load the data, split into training and testing sets:
            data = np.loadtxt(os.path.join(path, file))
            train = data[:training]
            test = data[training:]
            yield identifier, train, test, (start_anomaly, end_anomaly)


def evaluate_configurations(
    algorithms: list[dict[str, Any]],
    data_path: str = "data",
    results_path: str = "results/results.csv",
    visualise: bool = True,
    save_scores: bool = False,
    scores_dir: str = "results",
) -> None:
    """
    Evaluate the performance of different algorithms on the UCR dataset.

    Parameters
    ----------
    algorithms : list[dict[str, Any]]
        An ordered list of dictionaries containing algorithms and their parameters.
    data_path : str, optional
        Path to the directory containing UCR dataset files. Default is "data".
    results_path : str, optional
        The path to the results CSV file. Default is "results/results.csv".
    visualise : bool, optional
        Whether to visualise the results. Default is True.
    save_scores : bool, optional
        Whether to save the scores arrays as .npy files. Default is False.
    scores_dir : str, optional
        Directory to save the scores arrays. Default is "results".
    """
    # Create necessary directories
    results_dir = os.path.dirname(results_path) or "."
    os.makedirs(results_dir, exist_ok=True)

    if visualise:
        os.makedirs(results_dir, exist_ok=True)

    if save_scores:
        scores_subdir = os.path.join(scores_dir, "scores")
        os.makedirs(scores_subdir, exist_ok=True)

    # Write header to results file
    with open(results_path, 'w') as f:
        f.write("id")
        for params in algorithms:
            name = params.get("name") or params.get("function").__name__
            f.write(f",{name}")
        f.write("\n")

    # Iterate through UCR datasets, apply algorithms, and write results:
    with open(results_path, 'a') as f:
        for identifier, train, test, (start_anomaly, end_anomaly) in get_ucr(data_path):
            start_time = time.time()
            print(f"Evaluating UCR {identifier}...\t" \
                  f"(training: [0:{len(train)}], " \
                  f"testing: [{len(train)}:{len(train) + len(test)}], " \
                  f"anomaly: [{start_anomaly}, {end_anomaly}])", end="")
            results = []
            for params in algorithms:
                name = params.get("name") or params.get("function").__name__
                title = f'{name} on UCR {identifier}'
                segmenter_params = params.get("segmenter_params") or {}
                window_size = segmenter_params.get("window_size")

                if window_size is None:
                    segmenter = params.get("segmenter")()
                    segmenter.segment(data=train)
                    window_size = segmenter.window_size

                # Apply the algorithm and find the minimum score
                func = params["function"]
                scores = func(train, test, params)
                if len(scores) < window_size * 2:
                    print(
                        f"\nWarning: Scores length"
                        f" {len(scores)} is less than two"
                        f" times window size"
                        f" {window_size}. Skipping."
                    )
                    results.append(False)
                    continue

                # Save scores array if requested
                if save_scores:
                    scores_filename = f"{identifier}-{name}-scores.npy"
                    scores_path = os.path.join(scores_dir, "scores", scores_filename)
                    np.save(scores_path, scores)
                min_idx = np.argmin(scores[window_size:-window_size]) + window_size

                # Check overlap with ground-truth anomaly
                anomaly_length = end_anomaly - start_anomaly + 1
                # Acceptable range: anomaly_length or 100
                margin = max(anomaly_length, 100)
                min_anomaly = start_anomaly - margin
                max_anomaly = end_anomaly + margin
                if min_anomaly < (min_idx + len(train)) < max_anomaly:
                    title += "\nCorrect prediction ✓"
                    results.append(True)
                else:
                    title += "\nIncorrect prediction ✗"
                    results.append(False)

                # Visualise the results if requested
                if visualise:
                    # Create figure with two subplots
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
                    train_x = np.arange(len(train))
                    test_x = np.arange(len(train), len(train) + len(test))

                    # First subplot: Training and testing data
                    ax1.plot(
                        train_x, train, '--',
                        color='gray', alpha=0.7,
                        label='Training data',
                    )
                    ax1.plot(test_x, test, color='#223F7A', label='Testing data')
                    ax1.axvspan(
                        start_anomaly, end_anomaly,
                        color='crimson', alpha=0.4,
                        label='Ground truth anomaly',
                    )
                    ax1.axvline(
                        x=min_anomaly, color='crimson',
                        linestyle=':', alpha=0.25,
                    )
                    ax1.axvline(
                        x=max_anomaly, color='crimson',
                        linestyle=':', alpha=0.25,
                    )
                    ax1.set_title(title)
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)

                    # Second subplot: Anomaly scores aligned with test data
                    ax2.plot(test_x, scores, color='crimson', label='Scores')
                    ax2.plot(
                        min_idx + len(train),
                        scores[min_idx], 'X',
                        markerfacecolor='crimson',
                        markersize=5, markeredgewidth=1,
                        markeredgecolor='black',
                        label='Center of suspected anomaly',
                    )
                    ax2.axvspan(start_anomaly, end_anomaly, color='crimson', alpha=0.4)
                    ax2.axvline(
                        x=min_anomaly, color='crimson',
                        linestyle=':', alpha=0.25,
                    )
                    ax2.axvline(
                        x=max_anomaly, color='crimson',
                        linestyle=':', alpha=0.25,
                    )
                    ax2.set_ylabel('Anomaly Score')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)

                    plt.tight_layout()
                    plot_filename = os.path.join(
                        results_dir, f"{identifier}-{name}.png"
                    )
                    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                    plt.close()

            end_time = time.time()
            evaluation_time = end_time - start_time
            print(f"\t({evaluation_time:.2f}s)")

            f.write(f"{identifier}")
            for result in results:
                f.write(f",{result}")
            f.write("\n")
            f.flush()


if __name__ == '__main__':
    algorithms = [
        {
            "function": hpm,
            "name": "HPM_PatternGraph_auto_diff",
            "segmenter": SlidingWindow,
            "segmenter_params": {
                "window_size": 200,
                "autocorrelation": True,
                "differentiation": True
            },
            "model": PatternGraph,
            "model_params": {},
            "discretiser": PiecewiseLinear,
            "discretiser_params": {
                "node_type": StructuralProminenceNode,
            },
            "node_params": {
                "structural_threshold": lambda depth: 0.25,
                "prominence_threshold": lambda depth: 0.25
            }
        },
        {
            "function": hpm,
            "name": "HPM_PatternGraph_auto",
            "segmenter": SlidingWindow,
            "segmenter_params": {
                "window_size": 200,
                "autocorrelation": True,
                "differentiation": False
            },
            "model": PatternGraph,
            "model_params": {},
            "discretiser": PiecewiseLinear,
            "discretiser_params": {
                "node_type": StructuralProminenceNode,
            },
            "node_params": {
                "structural_threshold": lambda depth: 0.25,
                "prominence_threshold": lambda depth: 0.25
            }
        }
    ]

    evaluate_configurations(algorithms)
