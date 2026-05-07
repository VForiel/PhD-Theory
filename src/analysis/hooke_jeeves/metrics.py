import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy as copy
import tqdm
import yaml

# Metrics ---------------------------------------------------------------------

def bound_outputs(outputs):
    bounded_outputs = np.asarray(outputs, dtype=float).copy()
    bounded_outputs[bounded_outputs <= 1] = 1
    return bounded_outputs

def average_depth(outputs):
    bounded_outputs = bound_outputs(outputs)
    return np.sum(bounded_outputs[1:]) / bounded_outputs[0]

def max_depth(outputs):
    bounded_outputs = bound_outputs(outputs)
    return np.max(bounded_outputs[1:]) / bounded_outputs[0]

def average_flux(outputs):
    bounded_outputs = bound_outputs(outputs)
    return np.sum(bounded_outputs[1:])

def max_flux(outputs):
    bounded_outputs = bound_outputs(outputs)
    return np.max(bounded_outputs[1:])

# Data generation -------------------------------------------------------------

def _run_single_calibration(task):

    ctx, metric_name, sample_index, metric_index, seed, metrics = task
    task_metric_index = metric_index
    metric_function = metrics[metric_name]
    working_ctx = copy(ctx)

    rng = np.random.default_rng(seed)
    working_ctx.chip.σ = np.abs(rng.normal(0.0, 1.0, len(working_ctx.chip.σ))) * working_ctx.interferometer.λ

    history = working_ctx.chip.calibrate(plot=False, β=0.6, hooke_jeeves_metric=metric_function)

    depths_history = np.asarray(history["depths"], dtype=float)
    metric_history = np.asarray(history["metric"], dtype=float).ravel()

    # Evaluate all the metrics on the final state
    final_metrics = np.empty(len(metrics), dtype=float)
    for eval_metric_index, (metric_name, metric_function) in enumerate(metrics.items()):
        outs = working_ctx.observe()
        final_metrics[eval_metric_index] = metric_function(outs)

    return {
        "metric_index": task_metric_index,
        "sample_index": sample_index,
        "history_length": metric_history.size,
        "depths_history": depths_history,
        "metric_history": metric_history,
        "final_metrics": final_metrics,
    }

def generate_data(ctx, metrics, samples=1000):

    # Check if data already exists in cache
    if os.path.exists("data.npz") and os.path.exists("data.yml"):
        with open("data.yml", "r") as f:
            metadata = yaml.safe_load(f)
        
        if metadata.get("samples") >= samples and metadata.get("metrics") == list(metrics.keys()):
            print("Loading data from cache...")
            data = np.load("data.npz")
            return data

    nb_metrics = len(metrics)
    nb_runs = samples * nb_metrics
    
    worker_count = min(os.cpu_count() or 1, nb_runs)
    base_seed = np.random.SeedSequence().entropy

    tasks = [
        (ctx, metric_name, sample_index, metric_index, int(base_seed + sample_index * nb_metrics + metric_index), metrics)
        for sample_index in range(samples)
        for metric_index, metric_name in enumerate(metrics)
    ]

    executor = ProcessPoolExecutor(max_workers=worker_count)
    task_iterator = executor.map(_run_single_calibration, tasks)

    # Declare storage variables
    depths_histories = None
    metrics_histories = None
    final_metrics = None

    progress_label = f"Hooke and Jeeves metric runs ({samples} samples x {nb_metrics} metrics)"
    for result in tqdm.tqdm(task_iterator, total=nb_runs, desc=progress_label):
        metric_index = result["metric_index"]
        sample_index = result["sample_index"]
        history_length = result["history_length"]
        depths_history = result["depths_history"]
        metric_history = result["metric_history"]
        final_metrics_vector = result["final_metrics"]

        if depths_histories is None:
            depths_histories = np.empty((nb_metrics, nb_runs, *depths_history.shape), dtype=float)
            metrics_histories = np.empty((nb_metrics, nb_runs, *metric_history.shape), dtype=float)
            final_metrics = np.empty((nb_metrics, nb_runs, *final_metrics_vector.shape), dtype=float)

        depths_histories[metric_index, sample_index, :] = depths_history
        metrics_histories[metric_index, sample_index, :] = metric_history
        final_metrics[metric_index, sample_index, :] = final_metrics_vector

    executor.shutdown()

    # Save data in a cache file
    np.savez_compressed(
        "data.npz",
        depths_histories=depths_histories,
        metrics_histories=metrics_histories,
        final_metrics=final_metrics,
        metric_names=list(metrics.keys()),
    )

    # Save metadata in a yml file
    metadata = {
        "samples": samples,
        "metrics": list(metrics.keys()),
    }
    with open("data.yml", "w") as f:
        yaml.dump(metadata, f)

    return {
        "depths_histories": depths_histories,
        "metrics_histories": metrics_histories,
        "final_metrics": final_metrics,
        "metric_names": list(metrics.keys()),
    }

# Plots -----------------------------------------------------------------------

def plot_final_metric_distributions(data, bins=60, density=False, figsize=None):
    """Plot final metric distributions after calibration with each metric.

    Parameters
    ----------
    data : dict or numpy.lib.npyio.NpzFile
        Output returned by :func:`generate_data`.
        Must contain ``final_metrics`` with shape
        ``(n_calibration_metrics, n_samples, n_evaluated_metrics)``.
        If present, ``metric_names`` is used for legend and subplot titles.
    bins : int, default=60
        Number of histogram bins.
    density : bool, default=True
        If True, draw normalized distributions.
    figsize : tuple[float, float] or None, default=None
        Matplotlib figure size. If None, an automatic size is used.

    Returns
    -------
    tuple[matplotlib.figure.Figure, numpy.ndarray, pandas.DataFrame]
        The figure, the flattened array of axes, and a DataFrame containing
        distribution summary statistics for each plotted distribution.
    """
    final_metrics = np.asarray(data["final_metrics"], dtype=float)

    if final_metrics.ndim != 3:
        raise ValueError(
            "Expected data['final_metrics'] to have shape "
            "(n_calibration_metrics, n_samples, n_evaluated_metrics)."
        )

    n_calibration_metrics, n_samples_axis, n_evaluated_metrics = final_metrics.shape

    # Try to recover metric names from the data payload.
    metric_names = data["metric_names"]

    # If cache metadata is available, keep only the expected sample range.
    sample_count = n_samples_axis
    if os.path.exists("data.yml"):
        with open("data.yml", "r") as metadata_file:
            metadata = yaml.safe_load(metadata_file) or {}
        metadata_samples = metadata.get("samples")
        if isinstance(metadata_samples, int) and metadata_samples > 0:
            sample_count = min(sample_count, metadata_samples)

    final_metrics = final_metrics[:, :sample_count, :]

    if figsize is None:
        figsize = (6.0 * n_evaluated_metrics, 4.2)

    fig, axes = plt.subplots(1, n_evaluated_metrics, figsize=figsize, squeeze=False)
    axes = axes.ravel()
    distribution_stats = []

    for evaluated_metric_index, ax in enumerate(axes):
        for calibration_metric_index in range(n_calibration_metrics):
            values = final_metrics[calibration_metric_index, :, evaluated_metric_index]
            finite_values = values[np.isfinite(values)]
            calibration_name = metric_names[calibration_metric_index]
            evaluated_name = metric_names[evaluated_metric_index]

            distribution_stats.append(
                {
                    "calibration_metric": calibration_name,
                    "evaluated_metric": evaluated_name,
                    "mean": float(np.mean(finite_values)) if finite_values.size else np.nan,
                    "std": float(np.std(finite_values, ddof=0)) if finite_values.size else np.nan,
                    "count": int(finite_values.size),
                }
            )

            if finite_values.size == 0:
                continue

            ax.hist(
                finite_values,
                bins=bins,
                histtype="step",
                density=density,
                linewidth=1.6,
                label=f"Calibrated with {calibration_name}",
            )

        ax.set_title(f"Final {evaluated_name}")
        ax.set_xlabel(evaluated_name)
        ax.set_ylabel("Density" if density else "Count")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    fig.tight_layout()
    stats_df = pd.DataFrame(distribution_stats)
    return fig, axes, stats_df


def plot_final_null_depth_distributions(data, bins=60, density=False, figsize=None):
    """Plot distributions of final null depths per output, for each calibration metric.

    One subplot is created per interferometric output. For each subplot, the
    distributions obtained with the different calibration metrics are overlaid.

    Parameters
    ----------
    data : dict or numpy.lib.npyio.NpzFile
        Output returned by :func:`generate_data`.
        Must contain ``depths_histories`` with shape
        ``(n_calibration_metrics, n_samples, n_steps, n_outputs)``.
        If present, ``metric_names`` is used for legend and subplot titles.
    bins : int, default=60
        Number of histogram bins.
    density : bool, default=False
        If True, draw normalized distributions.
    figsize : tuple[float, float] or None, default=None
        Matplotlib figure size. If None, an automatic size is used.

    Returns
    -------
    tuple[matplotlib.figure.Figure, numpy.ndarray, pandas.DataFrame]
        The figure, the flattened array of axes, and a DataFrame containing
        distribution summary statistics for each plotted distribution.
    """
    depths_histories = np.asarray(data["depths_histories"], dtype=float)

    if depths_histories.ndim != 4:
        raise ValueError(
            "Expected data['depths_histories'] to have shape "
            "(n_calibration_metrics, n_samples, n_steps, n_outputs)."
        )

    n_calibration_metrics, n_samples_axis, n_steps, n_outputs = depths_histories.shape

    # Try to recover metric names from the data payload.
    metric_names = data["metric_names"]

    # If cache metadata is available, keep only the expected sample range.
    sample_count = n_samples_axis
    if os.path.exists("data.yml"):
        with open("data.yml", "r") as metadata_file:
            metadata = yaml.safe_load(metadata_file) or {}
        metadata_samples = metadata.get("samples")
        if isinstance(metadata_samples, int) and metadata_samples > 0:
            sample_count = min(sample_count, metadata_samples)

    # Extract the final depths (last calibration step) for each run.
    # Shape: (n_calibration_metrics, n_samples, n_outputs)
    final_depths = depths_histories[:, :sample_count, -1, :]

    if figsize is None:
        figsize = (5.0 * n_outputs, 4.2)

    fig, axes = plt.subplots(1, n_outputs, figsize=figsize, squeeze=False)
    axes = axes.ravel()
    distribution_stats = []

    for output_index, ax in enumerate(axes):
        for calibration_metric_index in range(n_calibration_metrics):
            values = final_depths[calibration_metric_index, :, output_index]
            finite_values = values[np.isfinite(values)]
            calibration_name = metric_names[calibration_metric_index]

            distribution_stats.append(
                {
                    "calibration_metric": calibration_name,
                    "output_index": output_index,
                    "mean": float(np.mean(finite_values)) if finite_values.size else np.nan,
                    "std": float(np.std(finite_values, ddof=0)) if finite_values.size else np.nan,
                    "count": int(finite_values.size),
                }
            )

            if finite_values.size == 0:
                continue

            ax.hist(
                finite_values,
                bins=bins,
                histtype="step",
                density=density,
                linewidth=1.6,
                label=f"Calibrated with {calibration_name}",
            )

        ax.set_title(f"Output {output_index}")
        ax.set_xlabel("Null depth")
        ax.set_ylabel("Density" if density else "Count")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    fig.tight_layout()
    stats_df = pd.DataFrame(distribution_stats)
    return fig, axes, stats_df

