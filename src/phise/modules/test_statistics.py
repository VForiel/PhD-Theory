"""Test statistics to compare two samples (H0 vs H1).

This module provides simple statistics (mean, median, AUC of distances,
etc.) as well as non-parametric tests (SciPy) to measure the separability
between two data distributions.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.origin'] = 'lower'
from copy import deepcopy as copy
import astropy.units as u
from scipy import stats


def get_vectors(ctx=None, nmc: int = 1000, size: int = 1000, progress_callback=None, flatten: bool = True):
    """Generate two sets of statistic vectors under H0 and H1.

    This simulates observations with and without companion(s) to build
    statistic arrays of shape ``(3, nmc, size)`` under H0 (no companion)
    and H1 (with companions), then concatenates each set along the first
    axis.

    Notes:
        - Assumes a compatible ``Context`` object exists (see
          ``phise.classes.context.Context``) and relies on its ``observe()``
          method.
        - To avoid circular imports, the ``Context`` import is local and only
          used if ``ctx`` is ``None``.

    Args:
        ctx: Observation context (if ``None``, a default VLTI context is
            instantiated). The context must contain at least one companion to
            generate H1.
        nmc: Number of Monte-Carlo realizations.
        size: Number of samples per realization.
        progress_callback (callable, optional): function accepting a float (0-1)
            representing the progress.
        flatten (bool, optional): If True, concatenates independent kernel outputs.
            If False, returns shapes (3, nmc, size). Defaults to True.

    Returns:
        Tuple ``(T0, T1)`` where:
        - T0: Concatenated vectors under H0, shape ``(3 * nmc * size,)`` if flatten=True.
        - T1: Concatenated vectors under H1, shape ``(3 * nmc * size,)`` if flatten=True.

    Raises:
        ValueError: If ``ctx`` contains no companions.
    """
    if ctx is None:
        # import local pour éviter un import circulaire lors du chargement
        from phise.classes.context import Context

        ctx = Context.get_VLTI()
        ctx.interferometer.chip.σ = np.zeros(14) * u.m

    if ctx.target.companions == []:
        raise ValueError(
            'No companions in the context. Please add companions to the context before generating vectors.'
        )

    ctx_h1 = copy(ctx)
    ctx_h0 = copy(ctx)
    ctx_h0.target.companions = []

    T0 = np.zeros((3, nmc, size))
    T1 = np.zeros((3, nmc, size))

    fov = ctx.interferometer.fov.to(u.mas).value

    for i in range(nmc):
        if progress_callback:
            progress_callback(i / nmc)
        else:
            print(f'⌛ Generating vectors... {round(i / nmc * 100, 2)}%', end='\r')
        
        for j in range(size):
            for c in ctx_h1.target.companions:
                c.θ = np.random.uniform(0, 2 * np.pi) * u.rad
                c.ρ = np.random.uniform(fov / 10, fov) * u.mas

            outs_h0 = ctx_h0.observe()
            outs_h1 = ctx_h1.observe()

            k_h0 = ctx_h0.interferometer.chip.process_outputs(outs_h0)
            k_h1 = ctx_h1.interferometer.chip.process_outputs(outs_h1)

            T0[:, i, j] = k_h0
            T1[:, i, j] = k_h1

    if progress_callback:
        progress_callback(1.0)
    else:
        print('✅ Vectors generation complete')
    
    if flatten:
        return (np.concatenate(T0), np.concatenate(T1))
    else:
        return (T0, T1)


def mean(u, v):
    """Absolute value of the mean of ``u``.

    Args:
        u: Samples under H1 (or any real/complex vector).
        v: Samples under H0 (unused).

    Returns:
        float: |mean(u)|
    """
    return np.abs(np.mean(u))


def median(u, v):
    """Absolute value of the median of ``u``.

    Args:
        u: Samples under H1.
        v: Samples under H0 (unused).

    Returns:
        float: |median(u)|
    """
    return np.abs(np.median(u))


def argmax(u, v, bins: int = 100):
    """Approximate mode of ``u`` using a histogram (most frequent bin).

    Args:
        u: Samples under H1.
        v: Samples under H0 (unused).
        bins: Number of histogram bins.

    Returns:
        float: Absolute center value of the bin maximizing the histogram.
    """
    (hist, bin_edges) = np.histogram(u, bins=bins)
    bin_edges = (bin_edges[1:] + bin_edges[:-1]) / 2
    return np.abs(bin_edges[np.argmax(hist)])


def argmax50(u, v):
    """Shortcut for ``argmax(u, v, bins=50)``."""
    return argmax(u, v, 50)


def argmax100(u, v):
    """Shortcut for ``argmax(u, v, bins=100)``."""
    return argmax(u, v, 100)


def argmax500(u, v):
    """Shortcut for ``argmax(u, v, bins=500)``."""
    return argmax(u, v, 500)


def kolmogorov_smirnov(u, v):
    """Two-sample Kolmogorov–Smirnov statistic.

    Returns:
        float: |D|, the maximal distance between empirical CDFs.
    """
    return np.abs(stats.ks_2samp(u, v).statistic)


def cramer_von_mises(u, v):
    """Two-sample Cramér–von Mises statistic.

    Returns:
        float: Absolute value of the statistic.
    """
    return np.abs(stats.cramervonmises_2samp(u, v).statistic)


def mannwhitneyu(u, v):
    """Mann–Whitney U statistic (Wilcoxon rank-sum).

    Returns:
        float: Absolute value of the statistic.
    """
    return np.abs(stats.mannwhitneyu(u, v).statistic)


def wilcoxon_mann_whitney(u, v):
    """Wilcoxon signed-rank test statistic (paired samples).

    Returns:
        float: Absolute value of the statistic.
    """
    return np.abs(stats.wilcoxon(u, v).statistic)


def anderson_darling(u, v):
    """Anderson–Darling k-sample statistic.

    Returns:
        float: Absolute value of the statistic.
    """
    return np.abs(stats.anderson_ksamp([u, v]).statistic)


def brunner_munzel(u, v):
    """Brunner–Munzel test statistic.

    Returns:
        float: Absolute value of the statistic.
    """
    return np.abs(stats.brunnermunzel(u, v).statistic)


def wasserstein_distance(u, v):
    """Wasserstein distance (earth mover's distance, order 1).

    Returns:
        float: Distance |W_1(u, v)|.
    """
    return np.abs(stats.wasserstein_distance(u, v))


def flattening(u, v):
    """Sum of deviations from the median: Σ |u - median(u)|.

    Returns:
        float: Measure of "flattening" around the median.
    """
    med = np.median(u)
    return np.sum(np.abs(u - med))


def shift_and_flattening(u, v):
    """Area under the curve of sorted distances to the median (AUC).

    Definition: if d_i = |u_i - median(u)| sorted, returns the area
    ∫ (d(x) + |median|) dx, x ∈ [0, 1].

    Returns:
        float: Area under the curve approximated via numerical integration.
    """
    med = np.median(u)
    distances = np.sort(np.abs(u - med))
    x = np.linspace(0, 1, len(u))
    auc = np.trapz(distances + np.abs(med), x)
    return auc


def median_of_abs(u, v):
    """Median of absolute values: median(|u|)."""
    return np.median(np.abs(u))


def full_sum(u, v):
    """Sum of absolute values: Σ |u|."""
    return np.sum(np.abs(u))


ALL_TESTS = {
    'Mean': mean,
    'Median': median,
    'Kolmogorov-Smirnov': kolmogorov_smirnov,
    'Cramer von Mises': cramer_von_mises,
    'Flattening': flattening,
    'Median of Abs': median_of_abs,
}
