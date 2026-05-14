"""
null_self_calibration.py
========================
Null Self-Calibration (NSC) model fitting using profile likelihood (for Na)
and parametric bootstrap (for sigma_inst, sigma_bias).

Model (simplified from Hanot et al. 2011)
------------------------------------------
The null depth n for a single measurement is modelled as::

    n = Na + 0.25 * xi^2 + eps

where:
  - Na        : astrophysical null (the quantity of interest)
  - xi        ~ N(0, sigma_inst)  residual OPD / instrumental error
  - eps       ~ N(0, sigma_bias)  detector / read-out bias noise
  - 0.25*xi^2 : follows a scaled chi-squared(1) distribution, approximated
                by an Exponential with mean tau = 0.5*sigma_inst^2

The resulting distribution of n is an Exponentially Modified Gaussian (EMG),
implemented in scipy as ``scipy.stats.exponnorm``.

Why the classic approach breaks down
--------------------------------------
In the Gaussian / space regime (sigma_inst -> 0) the chi-squared term
vanishes, leaving a pure Normal(Na, sigma_bias).  The EMG shape parameter
K = tau/sigma_bias -> 0 sits on the **boundary** of its feasible domain.
Asymptotic (Hessian-based) confidence intervals are unreliable there.

Robust approach implemented here
----------------------------------
1. **Point estimates** (Na, sigma_inst, sigma_bias): multi-start MLE via
   L-BFGS-B, same as before.

2. **Profile-likelihood CI for Na** (Wilks theorem):
   For each fixed Na value, re-optimise (sigma_inst, sigma_bias).  The 95 % CI
   is the set of Na values for which
   ``2 * n * (NLL_prof - NLL_min) <= chi2(1, 0.95) = 3.84``.
   This is valid even on the boundary sigma_inst = 0.

3. **Parametric bootstrap CIs for sigma_inst and sigma_bias**:
   Generate B replicates from the fitted forward model, re-fit each, take
   percentile intervals.  In the Gaussian regime (sigma_inst_hat ~ 0) the
   bootstrap distribution of sigma_inst collapses to zero, so we report a
   one-sided upper bound instead of a symmetric interval.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, exponnorm

__all__ = [
    "nsc_forward_sample",
    "fit_nsc",
    "fit_one_hypothesis",
]


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

def nsc_forward_sample(
    size: int,
    na: float,
    sigma_inst: float,
    sigma_bias: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw *size* null-depth samples from the simplified NSC forward model.

    Parameters
    ----------
    size : int
        Number of samples.
    na : float
        Astrophysical null depth.
    sigma_inst : float
        Standard deviation of the instrumental OPD error (xi).
    sigma_bias : float
        Standard deviation of the detector bias noise (eps).
    rng : numpy.random.Generator

    Returns
    -------
    numpy.ndarray, shape (size,)
    """
    xi = rng.standard_normal(size) * sigma_inst
    eps = rng.standard_normal(size) * sigma_bias
    return na + 0.25 * xi ** 2 + eps


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _emg_nll(
    x: np.ndarray,
    na: float,
    sigma_inst: float,
    sigma_bias: float,
) -> float:
    """Negative mean log-likelihood of the EMG approximation.

    EMG parameterisation:
      scale = sigma_bias
      tau   = 0.5 * sigma_inst^2  (exponential mean)
      K     = tau / scale          (exponnorm shape)
      loc   = na
    """
    scale = max(sigma_bias, 1e-12)
    tau = 0.5 * sigma_inst ** 2
    k = tau / scale
    lp = exponnorm.logpdf(x, K=k, loc=na, scale=scale)
    if not np.all(np.isfinite(lp)):
        return np.inf
    return -float(np.mean(lp))


def _objective_free(x: np.ndarray):
    """Return NLL objective over [na, log(sigma_inst), log(sigma_bias)]."""
    def obj(p: np.ndarray) -> float:
        return _emg_nll(x, float(p[0]), float(np.exp(p[1])), float(np.exp(p[2])))
    return obj


def _objective_na_fixed(x: np.ndarray, na: float):
    """Return NLL objective over [log(sigma_inst), log(sigma_bias)] with na fixed."""
    def obj(p: np.ndarray) -> float:
        return _emg_nll(x, na, float(np.exp(p[0])), float(np.exp(p[1])))
    return obj


def _initial_params(x: np.ndarray, sigma_bias_max: float) -> dict:
    """Compute moment-based initial guesses and search bounds."""
    x_mean = float(np.mean(x))
    x_med = float(np.median(x))
    x_std = max(float(np.std(x)), 1e-12)

    sigma_inst_moment = max(np.sqrt(2.0 * x_std), 1e-6)
    sigma_inst_min = 1e-6
    sigma_inst_max = max(20.0 * sigma_inst_moment, 1.0)

    na_min = float(np.min(x) - x_std)
    na_max = float(np.max(x) + x_std)

    tau_guess = max(x_mean - x_med, 1e-9)
    sigma_inst_guess = max(np.sqrt(2.0 * tau_guess), 0.5 * sigma_inst_moment)
    sigma_bias_guess = float(
        np.clip(np.sqrt(max(x_std ** 2 - tau_guess ** 2, 1e-12)), 1e-9, sigma_bias_max)
    )

    return {
        "x_std": x_std,
        "x_med": x_med,
        "na_min": na_min,
        "na_max": na_max,
        "sigma_inst_min": sigma_inst_min,
        "sigma_inst_max": sigma_inst_max,
        "sigma_inst_moment": sigma_inst_moment,
        "sigma_inst_guess": sigma_inst_guess,
        "sigma_bias_guess": sigma_bias_guess,
    }


def _run_mle(
    x: np.ndarray,
    sigma_bias_max: float,
    na_fixed: Optional[float] = None,
) -> dict:
    """Run multi-start L-BFGS-B and return the result with the lowest NLL.

    Parameters
    ----------
    x : 1-D array of finite null-depth values
    sigma_bias_max : upper bound for sigma_bias
    na_fixed : if given, Na is held fixed during optimisation
    """
    ip = _initial_params(x, sigma_bias_max)

    if na_fixed is None:
        bounds = [
            (ip["na_min"], ip["na_max"]),
            (np.log(ip["sigma_inst_min"]), np.log(ip["sigma_inst_max"])),
            (np.log(1e-9), np.log(sigma_bias_max)),
        ]
        starts = [
            np.array([
                np.clip(ip["x_med"], ip["na_min"], ip["na_max"]),
                np.log(np.clip(ip["sigma_inst_guess"], ip["sigma_inst_min"], ip["sigma_inst_max"])),
                np.log(ip["sigma_bias_guess"]),
            ]),
            np.array([
                np.clip(ip["x_med"] - 0.2 * ip["x_std"], ip["na_min"], ip["na_max"]),
                np.log(np.clip(ip["sigma_inst_moment"], ip["sigma_inst_min"], ip["sigma_inst_max"])),
                np.log(np.clip(0.05 * ip["x_std"], 1e-9, sigma_bias_max)),
            ]),
        ]
        obj = _objective_free(x)
    else:
        bounds = [
            (np.log(ip["sigma_inst_min"]), np.log(ip["sigma_inst_max"])),
            (np.log(1e-9), np.log(sigma_bias_max)),
        ]
        starts = [
            np.array([
                np.log(np.clip(ip["sigma_inst_guess"], ip["sigma_inst_min"], ip["sigma_inst_max"])),
                np.log(ip["sigma_bias_guess"]),
            ]),
            np.array([
                np.log(np.clip(ip["sigma_inst_moment"], ip["sigma_inst_min"], ip["sigma_inst_max"])),
                np.log(np.clip(0.05 * ip["x_std"], 1e-9, sigma_bias_max)),
            ]),
        ]
        obj = _objective_na_fixed(x, na_fixed)

    best = None
    for p0 in starts:
        res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res

    if na_fixed is None:
        na_hat = float(best.x[0])
        si_hat = float(np.exp(best.x[1]))
        sb_hat = float(np.exp(best.x[2]))
    else:
        na_hat = na_fixed
        si_hat = float(np.exp(best.x[0]))
        sb_hat = float(np.exp(best.x[1]))

    return {
        "na": na_hat,
        "sigma_inst": si_hat,
        "sigma_bias": sb_hat,
        "nll": float(best.fun),
        "success": bool(best.success),
        "sigma_inst_moment": ip["sigma_inst_moment"],
    }


def _profile_na_ci(
    x: np.ndarray,
    na_hat: float,
    nll_min: float,
    sigma_bias_max: float,
    x_std: float,
    ci_level: float,
    n_grid: int = 80,
) -> tuple:
    """Compute profile-likelihood CI for Na (Wilks theorem).

    Scans Na over a grid, profiles out (sigma_inst, sigma_bias) at each
    point, and finds the CI boundaries by linear interpolation.
    """
    n = len(x)
    chi2_thresh = float(chi2.ppf(ci_level, df=1))

    na_grid = np.linspace(na_hat - 5.0 * x_std, na_hat + 5.0 * x_std, n_grid)
    profile_nll = np.array([
        _run_mle(x, sigma_bias_max=sigma_bias_max, na_fixed=na_val)["nll"]
        for na_val in na_grid
    ])

    # Wilks statistic: 2 * n * (prof_nll - min_nll)  (mean NLL -> multiply by n)
    wilks = 2.0 * n * (profile_nll - nll_min)

    def _find_boundary(side: str) -> float:
        idx_min = int(np.argmin(wilks))
        if side == "left":
            sub_stat = wilks[: idx_min + 1][::-1]
            sub_grid = na_grid[: idx_min + 1][::-1]
        else:
            sub_stat = wilks[idx_min:]
            sub_grid = na_grid[idx_min:]
        for j in range(len(sub_stat) - 1):
            if sub_stat[j] <= chi2_thresh <= sub_stat[j + 1]:
                frac = (chi2_thresh - sub_stat[j]) / (sub_stat[j + 1] - sub_stat[j])
                return float(sub_grid[j] + frac * (sub_grid[j + 1] - sub_grid[j]))
        # CI boundary not reached within scan range -> return edge
        return float(sub_grid[-1])

    return _find_boundary("left"), _find_boundary("right")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_nsc(
    null_sample: Any,
    sigma_bias_max: Optional[float] = None,
    n_bootstrap: int = 200,
    ci_level: float = 0.95,
    gaussian_threshold: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Fit the NSC model using profile likelihood (Na) and parametric bootstrap.

    Robust in both observation regimes:

    * **Turbulent / ground-based** (sigma_inst >> 0): the EMG tail is strong
      and all parameters are well-determined.

    * **Gaussian / space** (sigma_inst -> 0): the distribution collapses to
      Normal.  The profile-likelihood CI for Na remains valid even at the
      boundary sigma_inst = 0.  The bootstrap reports a one-sided upper bound
      for sigma_inst instead of a symmetric interval.

    Parameters
    ----------
    null_sample : array-like
        Observed null-depth values for one estimator channel (normalised by
        the bright channel).
    sigma_bias_max : float, optional
        Hard upper bound for sigma_bias.  Defaults to ``0.25 * std(sample)``.
    n_bootstrap : int
        Number of parametric bootstrap replicates for sigma_inst / sigma_bias
        CIs.  Pass 0 to skip (faster; CIs will be NaN).
    ci_level : float
        Confidence level (default 0.95 -> 95 % CIs).
    gaussian_threshold : float
        Regime flag: if ``sigma_inst_hat < gaussian_threshold * std(sample)``
        the regime is flagged as *Gaussian* and sigma_inst CI is one-sided.
    rng : numpy.random.Generator, optional
        Random-number generator for reproducibility.

    Returns
    -------
    dict with keys:

    na : float
        MLE estimate of the astrophysical null depth.
    sigma_inst : float
        MLE estimate of the instrumental OPD noise.
    sigma_bias : float
        MLE estimate of the detector bias noise.
    na_ci : (float, float)
        Profile-likelihood CI for Na.
    sigma_inst_ci : (float, float)
        Bootstrap CI.  In Gaussian regime: ``(0.0, upper)``.
    sigma_bias_ci : (float, float)
        Bootstrap CI for sigma_bias.
    gaussian_regime : bool
        True when sigma_inst is consistent with zero.
    sigma_inst_moment : float
        Moment-based sigma_inst estimate (diagnostic).
    sigma_bias_max : float
        Upper bound used during optimisation.
    success : bool
        Whether MLE optimisation converged.
    objective : float
        Mean NLL at the MLE solution.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(null_sample, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    x_std = max(float(np.std(x)), 1e-12)

    if sigma_bias_max is None:
        sigma_bias_max = max(1e-9, 0.25 * x_std)
    sigma_bias_max = max(1e-9, float(sigma_bias_max))

    # ------------------------------------------------------------------
    # 1. MLE point estimates
    # ------------------------------------------------------------------
    mle = _run_mle(x, sigma_bias_max=sigma_bias_max)
    na_hat = mle["na"]
    si_hat = mle["sigma_inst"]
    sb_hat = mle["sigma_bias"]
    nll_min = mle["nll"]

    # ------------------------------------------------------------------
    # 2. Profile-likelihood CI for Na
    # ------------------------------------------------------------------
    na_ci = _profile_na_ci(x, na_hat, nll_min, sigma_bias_max, x_std, ci_level)

    # ------------------------------------------------------------------
    # 3. Parametric bootstrap CI for sigma_inst and sigma_bias
    # ------------------------------------------------------------------
    gaussian_regime = si_hat < gaussian_threshold * x_std

    if n_bootstrap > 0:
        boot_si = np.empty(n_bootstrap)
        boot_sb = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            boot_x = nsc_forward_sample(n, na_hat, si_hat, sb_hat, rng)
            res_b = _run_mle(boot_x, sigma_bias_max=sigma_bias_max)
            boot_si[b] = res_b["sigma_inst"]
            boot_sb[b] = res_b["sigma_bias"]

        alpha = 1.0 - ci_level
        lo_p = 100.0 * alpha / 2.0
        hi_p = 100.0 * (1.0 - alpha / 2.0)

        if gaussian_regime:
            si_ci = (0.0, float(np.percentile(boot_si, 100.0 * ci_level)))
        else:
            si_ci = (float(np.percentile(boot_si, lo_p)), float(np.percentile(boot_si, hi_p)))

        sb_ci = (float(np.percentile(boot_sb, lo_p)), float(np.percentile(boot_sb, hi_p)))
    else:
        si_ci = (float("nan"), float("nan"))
        sb_ci = (float("nan"), float("nan"))

    return {
        "na": na_hat,
        "sigma_inst": si_hat,
        "sigma_bias": sb_hat,
        "na_ci": (float(na_ci[0]), float(na_ci[1])),
        "sigma_inst_ci": si_ci,
        "sigma_bias_ci": sb_ci,
        "gaussian_regime": gaussian_regime,
        "sigma_inst_moment": mle["sigma_inst_moment"],
        "sigma_bias_max": sigma_bias_max,
        "success": mle["success"],
        "objective": nll_min,
    }


def fit_one_hypothesis(
    null_estimators: dict,
    sigma_bias_factor: float = 0.20,
    n_bootstrap: int = 200,
    ci_level: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Fit the NSC model independently for each null estimator channel.

    Parameters
    ----------
    null_estimators : dict
        Mapping from estimator index to array of null-depth measurements
        (normalised by the bright channel).
    sigma_bias_factor : float
        Sets ``sigma_bias_max = sigma_bias_factor * std(estimator)``.
    n_bootstrap : int
        Number of parametric bootstrap replicates.  Pass 0 to skip.
    ci_level : float
        Confidence level for returned intervals.
    rng : numpy.random.Generator, optional

    Returns
    -------
    dict
        Mapping from estimator index to the result dict returned by
        :func:`fit_nsc`.
    """
    if rng is None:
        rng = np.random.default_rng()

    fits = {}
    for idx, samples in null_estimators.items():
        x = np.asarray(samples, dtype=float)
        x_std = max(float(np.std(x)), 1e-12)
        sb_max = max(1e-9, sigma_bias_factor * x_std)
        fits[idx] = fit_nsc(
            x,
            sigma_bias_max=sb_max,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            rng=rng,
        )
    return fits
