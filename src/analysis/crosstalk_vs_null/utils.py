"""Shared helpers for crosstalk studies of the PHOB N4x4-T8."""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from scipy.linalg import expm

from phise.classes.archs import N4x4_T8


@dataclass
class CurveStatistics:
    """Summary statistics over bootstrap realizations."""

    mean: np.ndarray
    median: np.ndarray
    percentile_05: np.ndarray
    percentile_95: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray


def random_energy_conserving_matrix(max_crosstalk: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a unitary 4-by-4 matrix with a prescribed crosstalk maximum."""
    if not 0 <= max_crosstalk < 1:
        raise ValueError("max_crosstalk must be in [0, 1)")
    if max_crosstalk == 0:
        return np.eye(4, dtype=np.complex128)

    amplitudes = rng.random((4, 4))
    phases = rng.uniform(0, 2 * np.pi, (4, 4))
    generator = np.triu(amplitudes * np.exp(1j * phases), 1)
    generator = generator - generator.conj().T

    def off_diagonal_max(scale: float) -> float:
        matrix = expm(scale * generator)
        return float(np.max(np.abs(matrix - np.diag(np.diag(matrix)))))

    lower, upper = 0.0, 1.0
    while off_diagonal_max(upper) < max_crosstalk:
        upper *= 2
        if upper > 64:
            raise RuntimeError("Could not reach the requested crosstalk")

    for _ in range(60):
        midpoint = (lower + upper) / 2
        if off_diagonal_max(midpoint) < max_crosstalk:
            lower = midpoint
        else:
            upper = midpoint
    return expm(upper * generator)


def validate_energy_conservation(matrix: np.ndarray, atol: float = 1e-12) -> bool:
    """Return whether a crosstalk matrix conserves power for every input."""
    matrix = np.asarray(matrix, dtype=np.complex128)
    return bool(np.allclose(matrix.conj().T @ matrix, np.eye(4), atol=atol))


def make_phob_context(cin: np.ndarray, cout: np.ndarray):
    """Build an ideal-camera PHOB context with no static chip OPD error."""
    from phise.examples.contexts.PHOB import get

    context = get()
    context.Γ = 0 * u.nm
    context.monochromatic = True
    context.camera.ideal = True
    wavelength = context.interferometer.λ
    context.interferometer.chip = N4x4_T8(
        φ=np.zeros(4) * wavelength,
        σ=np.zeros(4) * wavelength,
        λ0=wavelength,
        Cin=cin,
        Cout=cout,
        name="N4x4-T8 bootstrap realization",
    )
    return context


def statistics(samples: np.ndarray, axis: int = -1) -> CurveStatistics:
    """Compute summary statistics along the bootstrap axis."""
    return CurveStatistics(
        mean=np.mean(samples, axis=axis),
        median=np.median(samples, axis=axis),
        percentile_05=np.percentile(samples, 5, axis=axis),
        percentile_95=np.percentile(samples, 95, axis=axis),
        minimum=np.min(samples, axis=axis),
        maximum=np.max(samples, axis=axis),
    )