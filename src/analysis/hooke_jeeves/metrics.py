"""Hooke and Jeeves analysis helpers.

This module contains the reusable implementation used by the notebook.
The notebook should stay as a scientific companion that explains the
problem, runs the code, and discusses the results.
"""

from __future__ import annotations

from copy import deepcopy as copy
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phise
import pltedit as plte
import tqdm


CACHE_VERSION = 1
CACHE_FILENAME = "hooke_jeeves_metrics_cache.npz"
HISTOGRAM_FIGURE_FILENAME = "hooke_jeeves_null_depth_histograms.plt"
EVOLUTION_FIGURE_FILENAME = "hooke_jeeves_metric_evolution.plt"


def _prepare_outputs(outputs):
    """Return a safe copy of the outputs with the log guard applied."""
    prepared_outputs = np.asarray(outputs, dtype=float).copy()
    prepared_outputs[prepared_outputs <= 1] = 1
    return prepared_outputs


def m1(outputs):
    """Compute the sum of non-bright outputs relative to the bright output."""
    prepared_outputs = _prepare_outputs(outputs)
    return np.sum(prepared_outputs[1:]) / prepared_outputs[0]


def m2(outputs):
    """Compute the brightest non-bright output relative to the bright output."""
    prepared_outputs = _prepare_outputs(outputs)
    return np.max(prepared_outputs[1:]) / prepared_outputs[0]


def m3(outputs):
    """Compute the sum of the non-bright outputs."""
    prepared_outputs = _prepare_outputs(outputs)
    return np.sum(prepared_outputs[1:])


def m4(outputs):
    """Compute the maximum of the non-bright outputs."""
    prepared_outputs = _prepare_outputs(outputs)
    return np.max(prepared_outputs[1:])


def m5(outputs, n=1):
    """Return the nth output relative to the bright output."""
    prepared_outputs = _prepare_outputs(outputs)
    return prepared_outputs[n] / prepared_outputs[0]


def m6(outputs, n=1):
    """Return the nth output without normalisation."""
    prepared_outputs = _prepare_outputs(outputs)
    return prepared_outputs[n]


metrics = {
    "sum(N) / B": m1,
    "max(N) / B": m2,
    "sum(N)": m3,
    "max(N)": m4,
    "O3/B": lambda outputs: m5(outputs, n=3),
    "O3": lambda outputs: m6(outputs, n=3),
}

comparison_metric_names = list(metrics.keys())[:4]


def _analysis_directory():
    """Return the directory that contains this analysis module."""
    return Path(__file__).resolve().parent


def _cache_path():
    """Return the cache path stored next to the analysis module."""
    return _analysis_directory() / CACHE_FILENAME


def _figure_path(filename):
    """Return the path used to persist a figure alongside the cache."""
    return _analysis_directory() / filename


def _load_cached_metrics(expected_sample_count=None):
    """Load the cached calibration data if it matches the configuration."""
    cache_path = _cache_path()
    if not cache_path.exists():
        return None

    with np.load(cache_path) as archive:
        if int(archive["cache_version"]) != CACHE_VERSION:
            return None

        cached_metric_names = archive["metric_names"].tolist()
        if cached_metric_names != list(metrics.keys()):
            return None

        if expected_sample_count is not None and int(archive["sample_count"]) != expected_sample_count:
            return None

        return {
            "cache_path": cache_path,
            "cache_version": int(archive["cache_version"]),
            "metric_names": archive["metric_names"],
            "output_labels": archive["output_labels"],
            "sample_count": int(archive["sample_count"]),
            "window_size": int(archive["window_size"]),
            "mean_depths": archive["mean_depths"],
            "final_metrics": archive["final_metrics"],
            "metric_histories": archive["metric_histories"],
            "metric_history_lengths": archive["metric_history_lengths"],
        }


def generate_metrics_cache(ctx: phise.Context, N=100, force=False):
    """Generate or refresh the calibration cache used by the analysis figures."""
    if not force:
        cached_data = _load_cached_metrics(expected_sample_count=N)
        if cached_data is not None:
            return cached_data

    metric_names = list(metrics.keys())
    n_metrics = len(metric_names)
    sample_count = int(N)
    mean_depths = None
    final_metrics = None
    raw_metric_histories = [[None for _ in range(sample_count)] for _ in range(n_metrics)]
    metric_history_lengths = np.zeros((n_metrics, sample_count), dtype=int)
    output_labels = None
    window_size = 100

    for sample_index in tqdm.tqdm(range(sample_count), desc="Hooke and Jeeves runs"):
        for metric_index, (metric_name, metric_function) in enumerate(metrics.items()):
            working_ctx = copy(ctx)
            working_ctx.chip.σ = np.abs(np.random.normal(0, 1, len(working_ctx.chip.σ))) * working_ctx.interferometer.λ

            history = working_ctx.chip.calibrate(plot=False, β=0.9, hooke_jeeves_metric=metric_function)
            depths_history = np.asarray(history["depths"], dtype=float)
            metric_history = np.asarray(history["metric"], dtype=float).ravel()
            current_window_size = min(window_size, depths_history.shape[0])
            mean_depth = np.mean(depths_history[-current_window_size:, :], axis=0)

            if mean_depths is None:
                n_outputs = mean_depth.shape[0]
                mean_depths = np.full((n_metrics, sample_count, n_outputs), np.nan, dtype=float)
                final_metrics = np.full((n_metrics, sample_count), np.nan, dtype=float)
                output_labels = np.array([f"Output {output_index + 1}" for output_index in range(n_outputs)], dtype="U")

            mean_depths[metric_index, sample_index, :] = mean_depth
            metric_history_lengths[metric_index, sample_index] = metric_history.size
            raw_metric_histories[metric_index][sample_index] = metric_history
            valid_metric_history = metric_history[np.isfinite(metric_history)]
            if valid_metric_history.size > 0:
                final_metrics[metric_index, sample_index] = valid_metric_history[-1]

    max_history_length = int(np.max(metric_history_lengths))
    metric_histories = np.full((n_metrics, sample_count, max_history_length), np.nan, dtype=float)
    for metric_index in range(n_metrics):
        for sample_index in range(sample_count):
            history = raw_metric_histories[metric_index][sample_index]
            history_length = history.size
            metric_histories[metric_index, sample_index, :history_length] = history

    cache_path = _cache_path()
    np.savez_compressed(
        cache_path,
        cache_version=np.array(CACHE_VERSION),
        metric_names=np.array(metric_names, dtype="U"),
        output_labels=output_labels,
        sample_count=np.array(sample_count),
        window_size=np.array(window_size),
        mean_depths=mean_depths,
        final_metrics=final_metrics,
        metric_histories=metric_histories,
        metric_history_lengths=metric_history_lengths,
    )

    return {
        "cache_path": cache_path,
        "cache_version": CACHE_VERSION,
        "metric_names": np.array(metric_names, dtype="U"),
        "output_labels": output_labels,
        "sample_count": sample_count,
        "window_size": window_size,
        "mean_depths": mean_depths,
        "final_metrics": final_metrics,
        "metric_histories": metric_histories,
        "metric_history_lengths": metric_history_lengths,
    }


def get_metrics_cache(ctx: phise.Context | None = None, N=100, force=False):
    """Return cached calibration data, generating it when necessary."""
    cached_data = _load_cached_metrics(expected_sample_count=N)
    if cached_data is not None and not force:
        return cached_data

    if ctx is None:
        raise ValueError("A context is required when the cache does not exist yet.")

    return generate_metrics_cache(ctx, N=N, force=force)


def build_summary_table(cache):
    """Build the metric comparison table from the cached null-depth values."""
    mean_depths = cache["mean_depths"]
    metric_names = cache["metric_names"].tolist()

    data_means = np.mean(mean_depths, axis=1)
    data_stds = np.std(mean_depths, axis=1)
    data_mins = np.min(mean_depths, axis=1)
    data_maxs = np.max(mean_depths, axis=1)

    table_data = []
    for metric_index, metric_name in enumerate(metric_names):
        row = [metric_name]
        row.extend(
            [
                f"{data_means[metric_index, output_index]:.2e} ± {data_stds[metric_index, output_index]:.2e} | {data_mins[metric_index, output_index]:.2e} - {data_maxs[metric_index, output_index]:.2e}"
                for output_index in range(mean_depths.shape[2])
            ]
        )
        row.append(
            f"{np.mean(data_means[metric_index]):.2e} ± {np.mean(data_stds[metric_index]):.2e} | {np.min(data_mins[metric_index]):.2e} - {np.max(data_maxs[metric_index]):.2e}"
        )
        table_data.append(row)

    columns = ["Metric", *cache["output_labels"].tolist(), "Total"]
    return pd.DataFrame(table_data, columns=columns)


def _plot_kde_density_curve(axis, values, label, color, grid_size=256, alpha=0.35, linewidth=2.0):
    """Plot a continuous density estimate on a logarithmic x-axis.

    The density is estimated in log10 space so the curve remains smooth and
    visually comparable to a violin plot, while still being displayed on the
    original positive metric axis.
    """
    density_values = np.asarray(values, dtype=float)
    density_values = density_values[np.isfinite(density_values) & (density_values > 0)]
    if density_values.size == 0:
        return None

    if density_values.size == 1:
        x_values = density_values
        y_values = np.array([1.0], dtype=float)
        axis.plot(x_values, y_values, color=color, linewidth=linewidth, label=label)
        return y_values, x_values

    log_values = np.log10(density_values)
    lower = np.min(log_values)
    upper = np.max(log_values)
    if np.isclose(lower, upper):
        lower -= 0.5
        upper += 0.5

    log_grid = np.linspace(lower, upper, grid_size)
    grid_values = np.power(10.0, log_grid)

    density_x = None
    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(log_values)
        density_log = kde(log_grid)
        density_x = density_log / (grid_values * np.log(10.0))
    except Exception:
        try:
            from scipy.ndimage import gaussian_filter1d

            fallback_bins = max(8, min(64, density_values.size * 4))
            histogram, edges = np.histogram(log_values, bins=fallback_bins, density=True)
            smoothed_histogram = gaussian_filter1d(histogram.astype(float), sigma=1.25, mode="nearest")
            centers = 0.5 * (edges[:-1] + edges[1:])
            density_log = np.interp(log_grid, centers, smoothed_histogram, left=0.0, right=0.0)
            density_x = density_log / (grid_values * np.log(10.0))
        except Exception:
            x_values = np.sort(density_values)
            y_values = np.linspace(0.0, 1.0, x_values.size, dtype=float)
            axis.plot(x_values, y_values, color=color, linewidth=linewidth, label=label)
            return y_values, x_values

    axis.fill_between(grid_values, density_x, alpha=alpha, color=color)
    axis.plot(grid_values, density_x, color=color, linewidth=linewidth, label=label)
    return density_x, grid_values


def _log_histogram_bins(values, bin_count):
    """Return logarithmically spaced bins for positive data."""
    histogram_values = np.asarray(values, dtype=float)
    histogram_values = histogram_values[np.isfinite(histogram_values) & (histogram_values > 0)]
    if histogram_values.size == 0:
        return np.array([1.0, 10.0])

    lower = np.min(histogram_values)
    upper = np.max(histogram_values)
    if np.isclose(lower, upper):
        lower *= 0.9
        upper *= 1.1
        if lower <= 0:
            lower = upper / 10.0

    return np.logspace(np.log10(lower), np.log10(upper), max(2, bin_count))


def _plot_log_histogram(axis, values, label, color, bin_count, alpha=0.5, linewidth=1.2):
    """Plot a classical histogram with bins spaced in log space."""
    histogram_values = np.asarray(values, dtype=float)
    histogram_values = histogram_values[np.isfinite(histogram_values) & (histogram_values > 0)]
    if histogram_values.size == 0:
        return None

    bins = _log_histogram_bins(histogram_values, bin_count)
    axis.hist(
        histogram_values,
        bins=bins,
        histtype="stepfilled",
        alpha=alpha,
        color=color,
        edgecolor="white",
        linewidth=linewidth,
        label=label,
    )
    return bins


def _plot_distribution_summary(axis, values, color="C0"):
    """Annotate a distribution with median, mean, and percentile markers."""
    summary_values = np.asarray(values, dtype=float)
    summary_values = summary_values[np.isfinite(summary_values) & (summary_values > 0)]
    if summary_values.size == 0:
        return

    median_value = np.median(summary_values)
    p5_value = np.percentile(summary_values, 5)
    p95_value = np.percentile(summary_values, 95)
    mean_value = np.mean(summary_values)
    std_value = np.std(summary_values)
    lower_bound = max(np.min(summary_values) * 0.9, mean_value - std_value)
    upper_bound = max(lower_bound * 1.001, mean_value + std_value)

    axis.axvspan(lower_bound, upper_bound, color=color, alpha=0.12, label="mean ± std")
    axis.axvline(median_value, color="C3", linewidth=2.0, label="median")
    axis.axvline(p5_value, color="C1", linestyle="--", linewidth=1.0, label="5th percentile")
    axis.axvline(p95_value, color="C1", linestyle="--", linewidth=1.0, label="95th percentile")
    axis.axvline(mean_value, color="black", linewidth=1.5, label="mean")
    axis.text(
        0.02,
        0.98,
        f"mean = {mean_value:.2e}\nstd = {std_value:.2e}",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize="small",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
    )


def plot_null_depth_histograms(ctx: phise.Context | None = None, N=100, cache=None, force=False, save=True):
    """Plot the null-depth distributions for the comparable metrics and each output."""
    if cache is None:
        cache = get_metrics_cache(ctx, N=N, force=force)

    mean_depths = cache["mean_depths"]
    metric_names = cache["metric_names"].tolist()
    output_labels = cache["output_labels"].tolist()
    sample_count = cache["sample_count"]

    comparable_metric_indices = [metric_names.index(metric_name) for metric_name in comparison_metric_names if metric_name in metric_names]

    fig, axes = plt.subplots(mean_depths.shape[2], 2, figsize=(14, 4.5 * mean_depths.shape[2]), constrained_layout=True)
    axes = np.atleast_2d(axes)
    bins = max(5, int(np.sqrt(sample_count)))

    for output_index, (hist_axis, density_axis) in enumerate(axes):
        for metric_index in comparable_metric_indices:
            metric_name = metric_names[metric_index]
            _plot_log_histogram(
                hist_axis,
                mean_depths[metric_index, :, output_index],
                label=metric_name,
                color=f"C{metric_index}",
                bin_count=bins,
            )
            _plot_kde_density_curve(
                density_axis,
                mean_depths[metric_index, :, output_index],
                label=metric_name,
                color=f"C{metric_index}",
            )
        hist_axis.set_title(f"Histogram for {output_labels[output_index]}")
        hist_axis.set_xlabel("Mean null depth")
        hist_axis.set_ylabel("Count")
        hist_axis.set_xscale("log")
        hist_axis.legend(fontsize="small")
        hist_axis.grid(alpha=0.2, which="both")

        density_axis.set_title(f"Density for {output_labels[output_index]}")
        density_axis.set_xlabel("Mean null depth")
        density_axis.set_ylabel("Density")
        density_axis.set_xscale("log")
        density_axis.legend(fontsize="small")
        density_axis.grid(alpha=0.2, which="both")

    fig.suptitle("Mean null depth histogram and density by metric")

    if save:
        plte.save(fig, _figure_path(HISTOGRAM_FIGURE_FILENAME))

    plt.show()
    return fig


def plot_metric_evolution(ctx: phise.Context | None = None, N=100, cache=None, force=False, save=True):
    """Plot the metric evolution and the final metric distribution for each metric."""
    if cache is None:
        cache = get_metrics_cache(ctx, N=N, force=force)

    metric_names = cache["metric_names"].tolist()
    metric_histories = cache["metric_histories"]
    final_metrics = cache["final_metrics"]

    n_metrics, _, max_history_length = metric_histories.shape
    fig, axes = plt.subplots(n_metrics, 3, figsize=(24, 4 * n_metrics), constrained_layout=True)
    if n_metrics == 1:
        axes = np.array([axes])

    iteration_axis = np.arange(1, max_history_length + 1)

    for metric_index, metric_name in enumerate(metric_names):
        evolution_axis, histogram_axis, density_axis = axes[metric_index]
        metric_data = metric_histories[metric_index]

        with np.errstate(all="ignore"):
            median_history = np.nanmedian(metric_data, axis=0)
            min_history = np.nanmin(metric_data, axis=0)
            max_history = np.nanmax(metric_data, axis=0)
            p5_history = np.nanpercentile(metric_data, 5, axis=0)
            p95_history = np.nanpercentile(metric_data, 95, axis=0)

        valid_iterations = np.isfinite(median_history)
        iteration_values = iteration_axis[valid_iterations]

        evolution_axis.fill_between(iteration_values, min_history[valid_iterations], max_history[valid_iterations], alpha=0.15, color="C0", label="min-max")
        evolution_axis.fill_between(iteration_values, p5_history[valid_iterations], p95_history[valid_iterations], alpha=0.20, color="C1", label="5th-95th percentile")
        evolution_axis.plot(iteration_values, median_history[valid_iterations], color="C3", linewidth=2.0, label="median")
        evolution_axis.plot(iteration_values, p5_history[valid_iterations], color="C1", linestyle="--", linewidth=1.0, alpha=0.9, label="5th percentile")
        evolution_axis.plot(iteration_values, p95_history[valid_iterations], color="C1", linestyle="--", linewidth=1.0, alpha=0.9, label="95th percentile")
        evolution_axis.set_title(f"Evolution of {metric_name}")
        evolution_axis.set_xlabel("Iteration")
        evolution_axis.set_ylabel("Metric value")
        evolution_axis.set_yscale("log")
        evolution_axis.grid(alpha=0.2)
        evolution_axis.legend(fontsize="small")

        final_values = final_metrics[metric_index]
        final_values = final_values[np.isfinite(final_values) & (final_values > 0)]
        bins = max(5, int(np.sqrt(final_values.size)))

        _plot_log_histogram(
            histogram_axis,
            final_values,
            label="final histogram",
            color="C0",
            bin_count=bins,
            alpha=0.65,
        )
        _plot_distribution_summary(histogram_axis, final_values, color="C1")
        histogram_axis.set_title(f"Final histogram of {metric_name}")
        histogram_axis.set_xlabel("Final metric value")
        histogram_axis.set_ylabel("Count")
        histogram_axis.set_xscale("log")
        histogram_axis.grid(alpha=0.2, which="both")
        histogram_axis.legend(fontsize="small")

        _plot_kde_density_curve(
            density_axis,
            final_values,
            label="final density",
            color="C0",
            alpha=0.25,
        )
        _plot_distribution_summary(density_axis, final_values, color="C1")
        density_axis.set_title(f"Final density of {metric_name}")
        density_axis.set_xlabel("Final metric value")
        density_axis.set_ylabel("Density")
        density_axis.set_xscale("log")
        density_axis.grid(alpha=0.2, which="both")
        density_axis.legend(fontsize="small")

    fig.suptitle("Metric evolution and final distributions")

    if save:
        plte.save(fig, _figure_path(EVOLUTION_FIGURE_FILENAME))

    plt.show()
    return fig


def run_once(ctx: phise.Context, name=None):
    """Run one calibration per metric and display the convergence summary."""
    ref_ctx = ctx

    if name is not None:
        considered_metrics = {name: metrics[name]}
    else:
        considered_metrics = metrics

    for metric_name, metric in considered_metrics.items():
        working_ctx = copy(ref_ctx)
        working_ctx.chip.σ = np.abs(np.random.normal(0, 1, len(working_ctx.chip.σ))) * working_ctx.interferometer.λ / 10

        print(f"Calibrating with metric: {metric_name}")
        history = working_ctx.chip.calibrate(plot=True, β=0.8, hooke_jeeves_metric=metric)
        depths_history = history["depths"]
        for output_index in range(depths_history.shape[1]):
            print(f"Mean on the last 100 null depth values for the output {output_index}: {np.mean(depths_history[-100:, output_index]):.2e}")
        print(f"Total mean: {np.mean(depths_history[-100:, :]):.2e}")
        working_ctx.chip.laugiergram(show=True)
        plt.show()


def compare(ctx: phise.Context, N=100, demo=False, plot=True):
    """Compare the metrics across multiple calibration runs."""
    ref_ctx = ctx

    data = np.zeros((len(metrics), N, 3))
    metric_data = np.zeros((len(metrics), N))

    for sample_index in tqdm.tqdm(range(N)):
        for metric_name, metric in metrics.items():
            working_ctx = copy(ref_ctx)
            working_ctx.chip.σ = np.abs(np.random.normal(0, 1, len(working_ctx.chip.σ))) * working_ctx.interferometer.λ / 10

            history = working_ctx.chip.calibrate(plot=False, β=0.9, hooke_jeeves_metric=metric)
            depths_history = history["depths"]
            mean_depth = np.mean(depths_history[-100:, :], axis=0)
            metric_index = list(metrics.keys()).index(metric_name)
            data[metric_index, sample_index] = mean_depth
            metric_data[metric_index, sample_index] = history["metric"][-1]

    data_means = np.mean(data, axis=1)
    data_stds = np.std(data, axis=1)
    data_mins = np.min(data, axis=1)
    data_maxs = np.max(data, axis=1)

    table_data = []
    for metric_index, metric_name in enumerate(metrics.keys()):
        row = [metric_name]
        row.extend([f"{data_means[metric_index, output_index]:.2e} ± {data_stds[metric_index, output_index]:.2e} | {data_mins[metric_index, output_index]:.2e} - {data_maxs[metric_index, output_index]:.2e}" for output_index in range(data.shape[2])])
        row.append(f"{np.mean(data_means[metric_index]):.2e} ± {np.mean(data_stds[metric_index]):.2e} | {np.min(data_mins[metric_index]):.2e} - {np.max(data_maxs[metric_index]):.2e}")
        table_data.append(row)
    columns = ["Metric", "Output 1", "Output 2", "Output 3", "Total"]
    df = pd.DataFrame(table_data, columns=columns)

    fig = None
    fig2 = None
    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        for output_index in range(data.shape[2]):
            for metric_index, metric_name in enumerate(list(metrics.keys())[:4]):
                ax[output_index].hist(data[metric_index, :, output_index], bins=int(np.sqrt(N)), alpha=0.5, label=metric_name)
            ax[output_index].set_title(f"Null depth distrib. for output {output_index + 1}")
            ax[output_index].set_xlabel("Mean null depth")
            ax[output_index].set_xscale("log")
            ax[output_index].set_ylabel("Occurrences")
            ax[output_index].legend()
        plt.show()

        fig2, ax2 = plt.subplots(1, len(metrics), figsize=(len(metrics) * 5, 5))
        for metric_index, metric_name in enumerate(metrics.keys()):
            ax2[metric_index].hist(metric_data[metric_index], bins=int(np.sqrt(N)), alpha=0.5)
            ax2[metric_index].set_title(f"Final metric value distrib. for {metric_name}")
            ax2[metric_index].set_xlabel("Final metric value")
            ax2[metric_index].set_xscale("log")
            ax2[metric_index].set_ylabel("Occurrences")

    return df, fig, fig2