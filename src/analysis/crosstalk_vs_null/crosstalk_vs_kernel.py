"""Bootstrap map of attainable kernel null versus crosstalk and cophasing error."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FuncFormatter
from tqdm.auto import tqdm

try:
    from .utils import make_phob_context, random_energy_conserving_matrix, validate_energy_conservation
except ImportError:
    from utils import make_phob_context, random_energy_conserving_matrix, validate_energy_conservation


@dataclass
class KernelGridStatistics:
    """Bootstrap statistics for each crosstalk and cophasing point."""

    mean: np.ndarray
    standard_deviation: np.ndarray
    relative_error: np.ndarray


def _metrics_from_outputs(outputs: np.ndarray) -> tuple[float, float]:
    """Return the classical null depth and kernel from raw output fluxes."""
    outputs = np.asarray(outputs, dtype=float)
    null_depth = float(outputs[3] / outputs[0]) if outputs[0] > 0 else np.nan
    kernel = float(abs(outputs[1] - outputs[2]))
    return null_depth, kernel


def _mean_metrics(context, phase_rms_nm: float, observations: int, rng: np.random.Generator) -> tuple[float, float]:
    """Average classical null depth and kernel over input-phase observations."""
    upstream_pistons = rng.normal(0.0, phase_rms_nm, size=(observations, 4)) * u.nm
    null_depths = np.empty(observations, dtype=float)
    kernels = np.empty(observations, dtype=float)
    for observation_index, piston_sample in enumerate(upstream_pistons):
        outputs = context.observe(upstream_pistons=piston_sample)
        null_depths[observation_index], kernels[observation_index] = _metrics_from_outputs(outputs)
    return float(np.nanmean(null_depths)), float(np.nanmean(kernels))


def _simulate_bootstrap_sample(
    crosstalk_value: float,
    phase_rms_value: float,
    observations: int,
    sample_seed: int,
) -> tuple[float, float]:
    """Simulate one independent crosstalk realization."""
    rng = np.random.default_rng(sample_seed)
    cin = random_energy_conserving_matrix(crosstalk_value, rng)
    cout = random_energy_conserving_matrix(crosstalk_value, rng)
    if not validate_energy_conservation(cin) or not validate_energy_conservation(cout):
        raise RuntimeError("Generated crosstalk matrix is not energy conserving")
    context = make_phob_context(cin, cout)
    return _mean_metrics(context, phase_rms_value, observations, rng)


def _simulate_grid_point(
    phase_index: int,
    crosstalk_index: int,
    crosstalk_value: float,
    phase_rms_value: float,
    bootstrap_samples: int,
    observations: int,
    sample_seeds: tuple[int, ...],
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Compute both metrics for one grid point in one process."""
    null_depths = np.empty(bootstrap_samples, dtype=float)
    kernels = np.empty(bootstrap_samples, dtype=float)
    for sample_index, sample_seed in enumerate(sample_seeds):
        null_depths[sample_index], kernels[sample_index] = _simulate_bootstrap_sample(
            crosstalk_value,
            phase_rms_value,
            observations,
            sample_seed,
        )
    return phase_index, crosstalk_index, null_depths, kernels


def _summarize_samples(samples: np.ndarray) -> KernelGridStatistics:
    """Summarize bootstrap samples along the last axis."""
    mean = np.nanmean(samples, axis=2)
    standard_deviation = np.nanstd(samples, axis=2, ddof=1 if samples.shape[2] > 1 else 0)
    relative_error = np.divide(
        standard_deviation,
        mean,
        out=np.full_like(standard_deviation, np.nan),
        where=mean > 0,
    )
    return KernelGridStatistics(mean, standard_deviation, relative_error)


def simulate_kernel_grid(
    crosstalk_levels: Iterable[float],
    phase_rms_levels_nm: Iterable[float],
    bootstrap_samples: int = 100,
    observations: int = 100,
    seed: int | None = 0,
    show_progress: bool = True,
    max_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, KernelGridStatistics]:
    """Simulate the pre-calibration kernel over a two-dimensional parameter grid.

    The chip has ideal injected phases and zero static OPD error. For every
    bootstrap matrix pair and every grid point, ``observations`` independent
    input piston vectors are drawn from a zero-mean normal distribution. Their
    RMS parameter is the standard deviation of each input piston in nanometres.
    The reported kernel is the mean of ``abs(Dark 1 - Dark 2)`` over those
    observations, followed by a bootstrap mean over crosstalk realizations.
    """
    crosstalk = np.asarray(tuple(crosstalk_levels), dtype=float)
    phase_rms = np.asarray(tuple(phase_rms_levels_nm), dtype=float)
    if crosstalk.ndim != 1 or crosstalk.size == 0 or np.any(crosstalk <= 0):
        raise ValueError("crosstalk_levels must be a non-empty sequence of positive values")
    if phase_rms.ndim != 1 or phase_rms.size == 0 or np.any(phase_rms < 0):
        raise ValueError("phase_rms_levels_nm must be a non-empty sequence of non-negative values")
    if bootstrap_samples < 1 or observations < 1:
        raise ValueError("bootstrap_samples and observations must be positive")

    _, _, kernel_statistics, _ = _simulate_metric_grids(
        crosstalk, phase_rms, bootstrap_samples, observations, seed, show_progress, max_workers
    )
    return crosstalk, phase_rms, kernel_statistics


def _simulate_metric_grids(
    crosstalk: np.ndarray,
    phase_rms: np.ndarray,
    bootstrap_samples: int,
    observations: int,
    seed: int | None,
    show_progress: bool,
    max_workers: int | None,
):
    """Run the shared process pool and return raw samples for both metrics."""
    kernel_samples = np.empty((phase_rms.size, crosstalk.size, bootstrap_samples), dtype=float)
    null_samples = np.empty_like(kernel_samples)
    seed_sequence = np.random.SeedSequence(seed)
    point_seed_sequences = seed_sequence.spawn(phase_rms.size * crosstalk.size)
    point_seeds = [
        tuple(child.generate_state(1)[0] for child in point_sequence.spawn(bootstrap_samples))
        for point_sequence in point_seed_sequences
    ]
    tasks = []
    point_index = 0
    for phase_index, phase_rms_value in enumerate(phase_rms):
        for crosstalk_index, crosstalk_value in enumerate(crosstalk):
            tasks.append(
                (phase_index, crosstalk_index, crosstalk_value, phase_rms_value, point_seeds[point_index])
            )
            point_index += 1
    iterator = tqdm(
        total=phase_rms.size * crosstalk.size,
        desc="Kernel/null grid",
        unit="point",
        disable=not show_progress,
    )
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _simulate_grid_point,
                phase_index,
                crosstalk_index,
                crosstalk_value,
                phase_rms_value,
                bootstrap_samples,
                observations,
                sample_seed_tuple,
            ): (phase_index, crosstalk_index)
            for phase_index, crosstalk_index, crosstalk_value, phase_rms_value, sample_seed_tuple in tasks
        }
        try:
            for future in as_completed(futures):
                phase_index, crosstalk_index, null_values, kernel_values = future.result()
                null_samples[phase_index, crosstalk_index, :] = null_values
                kernel_samples[phase_index, crosstalk_index, :] = kernel_values
                iterator.update()
        finally:
            iterator.close()

    return null_samples, kernel_samples, _summarize_samples(kernel_samples), _summarize_samples(null_samples)


def simulate_kernel_and_null_grids(
    crosstalk_levels: Iterable[float],
    phase_rms_levels_nm: Iterable[float],
    bootstrap_samples: int = 100,
    observations: int = 100,
    seed: int | None = 0,
    show_progress: bool = True,
    max_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, KernelGridStatistics, KernelGridStatistics]:
    """Simulate kernel and classical null maps using identical realizations."""
    crosstalk = np.asarray(tuple(crosstalk_levels), dtype=float)
    phase_rms = np.asarray(tuple(phase_rms_levels_nm), dtype=float)
    if crosstalk.ndim != 1 or crosstalk.size == 0 or np.any(crosstalk <= 0):
        raise ValueError("crosstalk_levels must be a non-empty sequence of positive values")
    if phase_rms.ndim != 1 or phase_rms.size == 0 or np.any(phase_rms < 0):
        raise ValueError("phase_rms_levels_nm must be a non-empty sequence of non-negative values")
    if bootstrap_samples < 1 or observations < 1:
        raise ValueError("bootstrap_samples and observations must be positive")
    _, _, kernel_statistics, null_statistics = _simulate_metric_grids(
        crosstalk, phase_rms, bootstrap_samples, observations, seed, show_progress, max_workers
    )
    return crosstalk, phase_rms, kernel_statistics, null_statistics


def _plot_grid(crosstalk, phase_rms_nm, values, title, colorbar_label, ax=None, relative_error=False):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    if relative_error:
        color_values = np.nan_to_num(values * 100, nan=0.0, posinf=0.0, neginf=0.0)
        norm = Normalize(vmin=0, vmax=np.max(color_values))
    else:
        color_values = np.maximum(values, np.finfo(float).tiny)
        positive_values = values[values > 0]
        if positive_values.size == 0:
            raise ValueError("Kernel values must contain at least one positive value")
        norm = LogNorm(vmin=np.min(positive_values), vmax=np.max(positive_values))
    positive_phase = phase_rms_nm > 0
    if not np.any(positive_phase):
        raise ValueError("A logarithmic RMS axis requires at least one positive RMS value")
    image = ax.pcolormesh(
        crosstalk * 100,
        phase_rms_nm[positive_phase],
        color_values[positive_phase],
        shading="auto",
        norm=norm,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(np.min(phase_rms_nm[positive_phase]), 100)
    ax.set_xlabel("Maximum off-diagonal crosstalk coefficient (%)")
    ax.set_ylabel("Input cophasing RMS error (nm)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    colorbar = ax.figure.colorbar(image, ax=ax, label=colorbar_label)
    if relative_error:
        colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
    return ax


def plot_classical_null_grid(
    crosstalk: np.ndarray,
    phase_rms_nm: np.ndarray,
    statistics: KernelGridStatistics,
    ax=None,
):
    """Plot the mean classical null depth ``Null / Bright``."""
    return _plot_grid(
        crosstalk,
        phase_rms_nm,
        statistics.mean,
        "Mean classical null depth",
        "Mean Null / Bright",
        ax,
    )


def plot_classical_null_relative_error(
    crosstalk: np.ndarray,
    phase_rms_nm: np.ndarray,
    statistics: KernelGridStatistics,
    ax=None,
):
    """Plot bootstrap relative uncertainty of the classical null depth."""
    return _plot_grid(
        crosstalk,
        phase_rms_nm,
        statistics.relative_error,
        "Relative uncertainty of the classical null depth",
        "Relative error",
        ax,
        relative_error=True,
    )


def plot_kernel_grid(crosstalk: np.ndarray, phase_rms_nm: np.ndarray, statistics: KernelGridStatistics, ax=None):
    """Plot the mean attainable kernel as a log-log colour map."""
    return _plot_grid(
        crosstalk, phase_rms_nm, statistics.mean, "Mean attainable kernel-null", "Mean |Dark 1 - Dark 2|", ax
    )


def plot_kernel_relative_error(crosstalk: np.ndarray, phase_rms_nm: np.ndarray, statistics: KernelGridStatistics, ax=None):
    """Plot bootstrap relative uncertainty as a log-log colour map."""
    return _plot_grid(
        crosstalk,
        phase_rms_nm,
        statistics.relative_error,
        "Relative bootstrap uncertainty of the kernel",
        "Relative error",
        ax,
        relative_error=True,
    )


if __name__ == "__main__":
    crosstalk, phase_rms, kernel_statistics, null_statistics = simulate_kernel_and_null_grids(
        np.geomspace(1e-6, 1e-1, 16),
        np.geomspace(1e-3, 100, 51),
    )
    plot_kernel_grid(crosstalk, phase_rms, kernel_statistics)
    plot_kernel_relative_error(crosstalk, phase_rms, kernel_statistics)
    plot_classical_null_grid(crosstalk, phase_rms, null_statistics)
    plot_classical_null_relative_error(crosstalk, phase_rms, null_statistics)
    plt.show()