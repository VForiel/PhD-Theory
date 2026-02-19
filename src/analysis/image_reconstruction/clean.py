"""
Nulling Interferometry Image Reconstruction Pipeline
=====================================================
Reconstructs the sky brightness distribution from nulling interferometer data
using a multi-step approach:
  0. Stellar flux estimation from the Bright output
  1. Robust sparse reconstruction (LASSO) on Kernel maps only
  2. Parametric refinement on all Dark maps with piston-inflated covariance
  3. Field rotation consistency check

Parameters
----------
data : np.ndarray, shape (N, 10)
    Measured photon counts for N hour angles and 10 outputs.
    Order: [Bright, Dark1..6, Kernel1..3]
maps : np.ndarray, shape (N, 10, M, M)
    Transmission maps for each hour angle and output.
    Same ordering as data.
piston_rms : float
    Estimated RMS piston per telescope, in nanometres.
fov : float
    Full field of view in mas (e.g. 10.0 for ±5 mas). Default 10.0.

Returns
-------
dict with keys:
    'f_star'      : estimated stellar flux (photons)
    'sources'     : list of dicts {x_mas, y_mas, contrast, snr}
    'image_lasso' : 2D reconstructed image from Kernel LASSO (M×M)
    'image_final' : 2D image with parametric sources placed on grid (M×M)
    'pixel_scale' : mas/pixel
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.ndimage import label
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Index constants
# ---------------------------------------------------------------------------
IDX_BRIGHT  = 0
IDX_DARK    = slice(1, 7)   # Dark 1-6
IDX_KERNEL  = slice(7, 10)  # Kernel 1-3
N_TEL       = 4             # number of telescopes
WAVELENGTH_NM = 2200.0      # reference wavelength in nm (K-band default, adjust if needed)


# ===========================================================================
# Step 0 — Stellar flux estimation
# ===========================================================================

def estimate_stellar_flux(data: np.ndarray, maps: np.ndarray) -> tuple[float, float]:
    """
    Estimate the stellar flux f from the Bright output alone.

    The model for the Bright output at angle i is:
        data[i, 0] ≈ T_bright(0, 0, i) * f  +  T_bright(r, θ, i) * f * c

    Since the planet contribution is at most a few percent of the stellar term
    (c << 1), and since the Bright output is constructive (T_bright(0,0) >> 0),
    we can approximate:
        f̂_i = data[i, 0] / T_bright(0, 0, i)

    where T_bright(0, 0, i) is the transmission at the centre pixel of the
    Bright map for angle i, i.e. maps[i, 0, M//2, M//2].

    The final estimate is the median of all f̂_i, which is robust to:
      - occasional outliers due to atmospheric events
      - the small residual planet contamination (biases each f̂_i slightly
        but the median is unaffected if fewer than half the angles are
        strongly contaminated)

    Uncertainty is estimated via the Median Absolute Deviation (MAD), scaled
    by 1.4826 to be consistent with a Gaussian standard deviation.

    Parameters
    ----------
    data : np.ndarray, shape (N, 10)
        Measured photon counts. data[:, 0] is the Bright output used here.
        Units: photons (or ADU with gain=1).
    maps : np.ndarray, shape (N, 10, M, M)
        Transmission maps. maps[:, 0, :, :] are the Bright maps.
        The stellar position is assumed to be at the map centre (M//2, M//2).

    Returns
    -------
    f_star : float
        Median estimate of the stellar flux in photons per integration.
        This accounts for telescope collecting area, integration time,
        throughput, and quantum efficiency (all already folded into the maps).
    f_star_std : float
        Robust 1-sigma uncertainty on f_star, derived from the MAD of the
        per-angle estimates. Reflects both photon noise and any angle-to-angle
        variability (e.g. transparency variations, residual planet signal).

    Raises
    ------
    ValueError
        If the Bright transmission at the centre is zero for all angles,
        which would indicate a normalisation problem in the maps.
    """
    M = maps.shape[-1]
    cx = M // 2

    # Transmission at star position (centre of the map)
    T_bright_centre = maps[:, IDX_BRIGHT, cx, cx]  # (N,)

    # Guard against zero/near-zero transmission at centre
    valid = T_bright_centre > 0.01 * T_bright_centre.max()
    if valid.sum() == 0:
        raise ValueError("Bright transmission at centre is zero for all angles — "
                         "check map normalisation.")

    f_estimates = data[valid, IDX_BRIGHT] / T_bright_centre[valid]

    f_star = float(np.median(f_estimates))
    mad     = float(np.median(np.abs(f_estimates - f_star)))
    f_star_std = 1.4826 * mad  # MAD → std for Gaussian

    return f_star, f_star_std


# ===========================================================================
# Covariance helpers
# ===========================================================================

def compute_data_covariance(data: np.ndarray,
                            maps: np.ndarray,
                            f_star: float,
                            piston_rms_nm: float) -> np.ndarray:
    """
    Build the diagonal of the data covariance matrix for all observations.

    The total variance on each measurement d[i, o] has two independent
    contributions that are added in quadrature:

    1. Shot noise (Poisson statistics):
           Var_shot(d[i, o]) = d[i, o]
       For photon-counting detectors, variance equals the mean count.
       We floor at 1.0 to avoid zero variance on empty pixels.

    2. Piston-induced noise:
       Atmospheric piston on telescope k introduces a random OPD φ_k, which
       perturbs the actual transmission map away from its nominal value:
           T_real(r,θ) ≈ T_nom(r,θ) + Σ_k (∂T/∂φ_k) · φ_k
       This propagates into a flux variance:
           Var_piston(d[i,o]) = f² · Σ_k σ_φ² · (∂T/∂φ_k)²
                               ≈ f² · N_tel · σ_φ² · <(∂T/∂φ)²>

       Since we do not have access to the complex amplitudes needed to compute
       ∂T/∂φ_k analytically, we use the following proxy: the temporal variance
       of T(0,0,i) across angles (i.e. np.var(maps[:, o, M//2, M//2])) is
       proportional to the piston sensitivity squared, because the maps change
       with angle primarily through the projected baseline geometry, which
       modulates the phase sensitivity similarly to a piston perturbation.

       The piston RMS is converted from nm to radians via:
           σ_φ [rad] = 2π * piston_rms_nm / WAVELENGTH_NM

       For Kernel outputs, the piston contribution is reduced by a factor of
       10 (i.e. multiplied by 0.1), reflecting the fact that Kernel maps are
       built to be first-order insensitive to piston by construction.

    The function returns only the diagonal of the covariance (a 2D array),
    since the full covariance matrix is diagonal — measurements at different
    angles and outputs are assumed independent.

    Parameters
    ----------
    data : np.ndarray, shape (N, 10)
        Measured photon counts. Used directly for shot noise variance.
    maps : np.ndarray, shape (N, 10, M, M)
        Transmission maps for all angles and outputs. Used to estimate piston
        sensitivity via the temporal variance of T at the map centre.
    f_star : float
        Estimated stellar flux in photons, from estimate_stellar_flux().
        Used to convert transmission variance into flux variance.
    piston_rms_nm : float
        RMS atmospheric piston per telescope in nanometres. This is the key
        instrumental parameter that controls the piston noise floor.

    Returns
    -------
    cov : np.ndarray, shape (N, 10)
        Diagonal elements of the covariance matrix.
        cov[i, o] = Var(data[i, o]) = Var_shot + Var_piston.
        Units: photons² (or ADU²).
        Kernel outputs (indices 7, 8, 9) have reduced piston variance.
    """
    N, n_out, M, _ = maps.shape
    cx = M // 2

    # Convert piston RMS from nm to radians (at reference wavelength)
    sigma_phi = 2.0 * np.pi * piston_rms_nm / WAVELENGTH_NM  # radians

    # Shot noise variance: Var = counts (Poisson)
    # Use max(data, 1) to avoid zero variance
    cov_shot = np.maximum(data, 1.0)  # (N, 10)

    # Piston covariance via finite differences on maps
    # ∂T/∂φ_k : we approximate by shifting the OPD of telescope k by δφ
    # Here we estimate it analytically as the imaginary part sensitivity.
    # Since we don't have access to the complex amplitudes, we use a
    # numerical proxy: the inter-angle variance of T at star position
    # captures the effective piston sensitivity.
    # 
    # More precisely, for each output o, the piston sensitivity is:
    #   σ_T_piston² = N_tel * σ_φ² * <|∂T/∂φ|²>
    # We estimate <|∂T/∂φ|²> from the temporal variance of T(0,0)
    # divided by the measured phase variance — a self-consistent estimator.
    #
    # Kernel outputs have near-zero ∂T/∂φ by construction → small piston cov.

    T_centre = maps[:, :, cx, cx]  # (N, 10) — transmission at star position

    # Temporal variance of T(0,0) as proxy for piston sensitivity
    # (valid when hour-angle changes drive piston-like phase variations)
    T_var = np.var(T_centre, axis=0, keepdims=True)  # (1, 10)

    # Expected phase variance from piston_rms
    phi_var = sigma_phi ** 2  # rad²

    # Piston-induced flux variance at star position
    # = f² * T_var * (sigma_phi² / phi_var_nominal)
    # where phi_var_nominal is 1 rad² (normalisation reference)
    cov_piston = (f_star ** 2) * T_var * phi_var  # (1, 10) broadcast to (N, 10)
    cov_piston = np.broadcast_to(cov_piston, (N, n_out)).copy()

    # Kernel outputs: reduce piston cov by a factor (robustness by construction)
    # Empirically, Kernel maps suppress piston at first order → factor ~10 reduction
    cov_piston[:, IDX_KERNEL] *= 0.1

    cov_total = cov_shot + cov_piston  # (N, 10)
    return cov_total


# ===========================================================================
# Step 1 — Robust sparse reconstruction (LASSO on Kernel maps)
# ===========================================================================

def build_kernel_system(data: np.ndarray,
                        maps: np.ndarray,
                        f_star: float,
                        cov: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the weighted linear system  δd_K = A_K · x  for Kernel outputs only.

    The forward model for a Kernel output k at angle i is:
        data[i, k] = T_k(0, 0, i) * f  +  T_k(r, θ, i) * f * c  +  noise

    Subtracting the stellar term gives the residual:
        δd[i, k] = data[i, k] - T_k(0, 0, i) * f
                 = Σ_pixels  T_k(pixel, i) * f * x(pixel)

    where x(pixel) is the contrast at that sky pixel (what we want to
    reconstruct). Stacking all (angle, kernel) pairs gives:
        b = A · x

    with:
      - b of shape (N*3,): one entry per (angle, kernel output) pair
      - A of shape (N*3, M²): each row is the flattened transmission map
        T_k(i) * f_star for one (angle, kernel) pair
      - x of shape (M²,): the flattened contrast image

    The system is pre-weighted by 1/σ (i.e. each row is divided by the
    standard deviation of that measurement) so that the unweighted least
    squares ||Ax - b||² on the output of this function is equivalent to
    a weighted least squares with weights W = diag(1/σ²).

    Only the 3 Kernel outputs (indices 7, 8, 9) are used here, because:
      - they are first-order robust to piston aberrations
      - they are antisymmetric around the origin, which helps break
        degeneracies in the image reconstruction
      - their noise covariance is more reliable (see compute_data_covariance)

    Parameters
    ----------
    data : np.ndarray, shape (N, 10)
        Measured photon counts. Only columns 7, 8, 9 (Kernels) are used.
    maps : np.ndarray, shape (N, 10, M, M)
        Transmission maps. Only maps[:, 7:10, :, :] are used.
    f_star : float
        Estimated stellar flux in photons, used to scale the design matrix
        rows from transmission (dimensionless) to flux (photons).
    cov : np.ndarray, shape (N, 10)
        Diagonal covariance from compute_data_covariance(). Used to compute
        per-row weights w = 1/sqrt(cov[i, k]).

    Returns
    -------
    A : np.ndarray, shape (N*3, M*M)
        Weighted design matrix. Each row corresponds to one (angle i,
        kernel k) pair and contains the flattened map T_k(i) * f_star / σ[i,k].
    b : np.ndarray, shape (N*3,)
        Weighted residual data vector. b[j] = δd[i,k] / σ[i,k] for the
        j-th (i, k) pair.
    W : np.ndarray, shape (N*3,)
        Per-row weights 1/σ[i,k], stored for reference (not used downstream
        directly since A and b are already pre-weighted).
    """
    N, n_out, M, _ = maps.shape

    kernel_indices = [7, 8, 9]
    n_k = len(kernel_indices)

    A_list = []
    b_list = []
    W_list = []

    for i in range(N):
        for ki, k in enumerate(kernel_indices):
            # Residual: subtract star contribution
            T_star = maps[i, k, M // 2, M // 2]
            delta_d = data[i, k] - T_star * f_star

            # Design matrix row: T_k(r,θ) * f_star for each pixel
            row = maps[i, k].ravel() * f_star  # (M*M,)

            sigma = np.sqrt(cov[i, k])
            w = 1.0 / sigma if sigma > 0 else 0.0

            A_list.append(row * w)
            b_list.append(delta_d * w)
            W_list.append(w)

    A = np.array(A_list)   # (N*3, M²)
    b = np.array(b_list)   # (N*3,)
    W = np.array(W_list)   # (N*3,)

    return A, b, W


def lasso_reconstruct(A: np.ndarray,
                      b: np.ndarray,
                      lambda_reg: float = None,
                      n_iter: int = 500) -> np.ndarray:
    """
    Solve the non-negative LASSO problem:
        min_{x ≥ 0}  ||Ax - b||²  +  λ ||x||_1

    using ISTA (Iterative Shrinkage-Thresholding Algorithm).

    This formulation is chosen because:
      - The L1 penalty (||x||_1) promotes sparsity: most pixels will be
        driven to exactly zero, and only the pixels corresponding to real
        point sources will be non-zero. This is physically motivated since
        we expect a small number of planets in the field.
      - The non-negativity constraint (x ≥ 0) enforces physical meaning:
        contrast cannot be negative (a planet cannot be darker than nothing).
      - ISTA is simple, memory-efficient, and guaranteed to converge for
        this convex problem. It alternates between a gradient step on the
        least-squares term and a soft-threshold step for the L1 term.

    ISTA update rule at iteration t:
        grad   = A^T A x_t - A^T b          (gradient of ||Ax-b||²)
        x_t+1  = max( x_t - (1/L)*grad - λ/L,  0 )
    where L = ||A^T A||_2 is the Lipschitz constant, used as step size.

    The combined max(..., 0) is equivalent to: soft-threshold then clip to
    non-negative, which implements the proximal operator of λ||x||_1 + I_{x≥0}.

    Convergence is checked by relative change in x: stops early if
    ||x_{t+1} - x_t|| < 1e-8 * ||x_t||.

    Auto-setting of lambda_reg:
        If None, λ is set to 5% of max(|A^T b|), which corresponds to the
        smallest λ that sets the highest-response pixel to zero — scaled down
        to allow the strongest source to emerge. This is a data-driven
        heuristic; tune manually for very faint or very bright sources.

    Parameters
    ----------
    A : np.ndarray, shape (m, n)
        Weighted design matrix from build_kernel_system(). Here m = N*3
        (observations) and n = M² (sky pixels).
    b : np.ndarray, shape (m,)
        Weighted residual data vector from build_kernel_system().
    lambda_reg : float or None
        L1 regularisation strength. Controls the sparsity of the solution:
          - Large λ → fewer, more confident detections (misses faint sources)
          - Small λ → more pixels non-zero (more false positives possible)
        If None, auto-set to 0.05 * max(|A^T b|).
    n_iter : int
        Maximum number of ISTA iterations. Default 500 is usually sufficient
        for convergence; increase for very large M or ill-conditioned A.

    Returns
    -------
    x : np.ndarray, shape (n,) = (M²,)
        Non-negative sparse contrast image, flattened. Reshape to (M, M)
        to get the 2D sky image. Units are dimensionless contrast values
        (planet flux / stellar flux). Most pixels will be exactly 0.0.
    """
    m, n = A.shape
    AtA = A.T @ A
    Atb = A.T @ b

    # Lipschitz constant for step size
    L = np.linalg.norm(AtA, ord=2)
    if L == 0:
        return np.zeros(n)
    step = 1.0 / L

    if lambda_reg is None:
        lambda_reg = 0.05 * np.max(np.abs(Atb))

    x = np.zeros(n)
    for _ in range(n_iter):
        grad = AtA @ x - Atb
        x_new = x - step * grad
        # Soft threshold + positivity
        x_new = np.maximum(x_new - lambda_reg * step, 0.0)
        if np.linalg.norm(x_new - x) < 1e-8 * np.linalg.norm(x + 1e-12):
            break
        x = x_new

    return x


# ===========================================================================
# Step 2 — Parametric refinement on all Dark maps
# ===========================================================================

def find_sources_from_image(image: np.ndarray,
                             pixel_scale: float,
                             fov: float,
                             n_sources_max: int = 5,
                             threshold_sigma: float = 3.0) -> list[dict]:
    """
    Detect point source candidates in the LASSO contrast image using
    connected-component labelling above an adaptive noise threshold.

    Detection procedure:
      1. Estimate the image noise level σ from the positive pixels only
         (the LASSO image is extremely sparse — most pixels are exactly 0.0
         by construction, so classical percentile-based estimators fail).
         The noise is estimated as the std of the lower half of positive
         pixels (below their median), which excludes the source peaks.
         Fallback chain if the above yields NaN or zero:
           a) std of all positive pixels
           b) 1% of the image maximum (handles perfectly sparse solutions)
      2. Threshold the image at max(threshold_sigma * σ,  peak * 1e-6).
         The absolute floor prevents zero-threshold edge cases.
      3. Label connected components in the binary mask using scipy.ndimage.
         Each connected component is treated as one source candidate.
      4. For each component, find the pixel with the maximum contrast value
         and record it as the source position.
      5. Convert pixel coordinates to angular coordinates in mas, with the
         map centre (M//2, M//2) corresponding to (0, 0) mas.
      6. Sort candidates by contrast (brightest first) and keep at most
         n_sources_max.

    The resulting source list is used as the initialisation point for the
    parametric refinement in refine_sources(). Position accuracy at this
    stage is limited to ±1 pixel (±pixel_scale mas); the refinement step
    will improve it to sub-pixel precision.

    Parameters
    ----------
    image : np.ndarray, shape (M, M)
        2D contrast image from lasso_reconstruct(), reshaped from (M²,).
        Values are dimensionless contrast (planet/star flux ratio).
        Most pixels should be 0.0 due to the LASSO sparsity constraint.
    pixel_scale : float
        Angular size of one pixel in mas/pixel. Used to convert pixel
        coordinates to angular positions. Computed as fov / M.
    fov : float
        Full field of view in mas. Used only for documentation / consistency
        with other functions; the actual conversion uses pixel_scale.
    n_sources_max : int
        Maximum number of source candidates to return. Candidates are ranked
        by peak contrast before truncation. Default 5.
    threshold_sigma : float
        Detection threshold in units of the image noise σ. Default 3.0
        corresponds to a 3-sigma detection threshold. Lower this value
        (e.g. to 2.0) if you expect very faint sources and are willing to
        accept more false positives for the refinement step to sort out.

    Returns
    -------
    sources : list of dict
        Each dict contains:
          'x_pix'        : int   — column index of peak pixel (0-indexed)
          'y_pix'        : int   — row index of peak pixel (0-indexed)
          'x_mas'        : float — RA-like angular position in mas
                                   (positive = East, i.e. increasing column)
          'y_mas'        : float — Dec-like angular position in mas
                                   (positive = North, i.e. increasing row)
          'contrast_init': float — peak contrast value at this pixel,
                                   used as initial guess for refine_sources()
        Sorted by 'contrast_init' descending. Empty list if no pixel exceeds
        the detection threshold.
    """
    M = image.shape[0]

    # Robust noise estimation on the LASSO image.
    # Problem: the LASSO image is extremely sparse — the vast majority of pixels
    # are exactly 0.0 by construction. This means:
    #   - np.percentile(image, 80) is often 0.0
    #   - image[image < 0.0] is an empty array → np.std([]) = NaN
    # Strategy: estimate noise only on strictly positive pixels (the "halo"
    # around real sources), falling back to the global non-zero std, and
    # ultimately to a fraction of the image maximum as a last resort.
    positive_pixels = image[image > 0]
    if len(positive_pixels) > 10:
        # Use the lower half of positive pixels as background noise estimate
        # (avoids the source peaks biasing the noise estimate upward)
        low_positive = positive_pixels[positive_pixels < np.median(positive_pixels)]
        noise = np.std(low_positive) if len(low_positive) > 1 else np.std(positive_pixels)
    else:
        noise = 0.0

    # If noise is zero or NaN (pure spike with no halo), fall back to a
    # fraction of the image maximum — this handles perfectly sparse solutions
    if not np.isfinite(noise) or noise == 0.0:
        noise = image.max() * 0.01  # 1% of peak as effective noise floor

    threshold = threshold_sigma * max(noise, image.max() * 1e-6)

    binary = image > threshold
    labeled, n_features = label(binary)

    sources = []
    for lbl in range(1, n_features + 1):
        region = np.where(labeled == lbl)
        peak_idx = np.argmax(image[region])
        y_pix = region[0][peak_idx]
        x_pix = region[1][peak_idx]
        contrast = image[y_pix, x_pix]

        # Convert pixel to mas (centre of map = 0,0)
        x_mas = (x_pix - M / 2) * pixel_scale
        y_mas = (y_pix - M / 2) * pixel_scale

        sources.append({
            'x_pix': x_pix, 'y_pix': y_pix,
            'x_mas': x_mas, 'y_mas': y_mas,
            'contrast_init': contrast
        })

    # Sort by contrast descending, keep top n_sources_max
    sources.sort(key=lambda s: s['contrast_init'], reverse=True)
    return sources[:n_sources_max]


def interp_transmission(maps_i: np.ndarray,
                         x_mas: float, y_mas: float,
                         fov: float) -> np.ndarray:
    """
    Evaluate all 10 transmission maps at a given sky position (x_mas, y_mas)
    for a single hour angle, using bilinear interpolation.

    The transmission maps are defined on a discrete M×M pixel grid. During
    parametric refinement (refine_sources), the optimiser needs to evaluate
    T(r, θ) at arbitrary sub-pixel positions. Bilinear interpolation provides
    a smooth, differentiable (almost everywhere) approximation suitable for
    gradient-based optimisers.

    Bilinear interpolation formula:
        T(x, y) ≈ (1-dx)(1-dy) T[y0,x0]
                 +    dx (1-dy) T[y0,x1]
                 + (1-dx)   dy T[y1,x0]
                 +    dx    dy T[y1,x1]
    where (x0, y0) = floor of fractional pixel coords, dx/dy are the
    fractional remainders.

    Coordinate convention:
        - (x_mas=0, y_mas=0) maps to pixel (M/2, M/2) — the map centre
        - x_mas increases to the right (East), x_pix increases right
        - y_mas increases upward (North), y_pix increases downward in array
          indexing (origin='lower' assumed in plotting)
        - Positions outside the map boundary are clamped to the nearest edge
          pixel (no extrapolation).

    Parameters
    ----------
    maps_i : np.ndarray, shape (10, M, M)
        The 10 transmission maps for a single hour angle index i.
        Typically maps[i] where maps has shape (N, 10, M, M).
        Order: [Bright, Dark1-6, Kernel1-3].
    x_mas : float
        East angular offset from the star in mas.
        Positive values correspond to higher column indices.
    y_mas : float
        North angular offset from the star in mas.
        Positive values correspond to higher row indices (with origin='lower').
    fov : float
        Full field of view in mas. Used to compute pixel_scale = fov / M,
        which converts angular positions to pixel coordinates.

    Returns
    -------
    T : np.ndarray, shape (10,)
        Interpolated transmission values for all 10 outputs at (x_mas, y_mas).
        T[0]   = Bright output
        T[1:7] = Dark outputs 1-6
        T[7:10]= Kernel outputs 1-3
        Values are dimensionless and should lie in [0, 1] for Bright/Dark
        outputs, and in [-0.5, 0.5] approximately for Kernel outputs.
    """
    M = maps_i.shape[-1]
    pixel_scale = fov / M  # mas/pixel

    # Convert mas to fractional pixel coordinates
    x_pix = x_mas / pixel_scale + M / 2
    y_pix = y_mas / pixel_scale + M / 2

    # Bilinear interpolation
    x0 = int(np.floor(x_pix))
    y0 = int(np.floor(y_pix))
    x1 = x0 + 1
    y1 = y0 + 1

    # Clamp to map boundaries
    x0 = np.clip(x0, 0, M - 1)
    x1 = np.clip(x1, 0, M - 1)
    y0 = np.clip(y0, 0, M - 1)
    y1 = np.clip(y1, 0, M - 1)

    dx = x_pix - np.floor(x_pix)
    dy = y_pix - np.floor(y_pix)

    T = ((1 - dx) * (1 - dy) * maps_i[:, y0, x0] +
              dx  * (1 - dy) * maps_i[:, y0, x1] +
         (1 - dx) *      dy  * maps_i[:, y1, x0] +
              dx  *      dy  * maps_i[:, y1, x1])
    return T  # (10,)


def parametric_residual(params: np.ndarray,
                         data: np.ndarray,
                         maps: np.ndarray,
                         f_star: float,
                         cov: np.ndarray,
                         fov: float,
                         use_dark_only: bool = True) -> float:
    """
    Compute the weighted chi-squared residual for a parametric multi-source model.

    This function is the objective to minimise in refine_sources(). It takes
    a parameter vector describing all point sources, builds the corresponding
    model prediction for every (angle, output) pair, and computes the goodness
    of fit against the measured data, weighted by the data covariance.

    Forward model:
        model[i, o] = T_o(0, 0, i) * f                      (star)
                    + Σ_s T_o(x_s, y_s, i) * f * c_s        (planets)

    where the sum is over all sources s described by params.

    Chi-squared:
        χ² = Σ_{i,o}  (data[i,o] - model[i,o])² / cov[i,o]

    The stellar transmission at the centre pixel T_o(0,0,i) = maps[i,o,M//2,M//2]
    is used for the star term, since the star is always at the map centre.

    The planet transmission T_o(x_s, y_s, i) is evaluated via bilinear
    interpolation using interp_transmission().

    If a source position falls outside the field of view, a large penalty
    (1e6) is added to χ² per source, effectively preventing the optimiser
    from placing sources outside the observable region.

    Note: the `use_dark_only` flag currently always uses outputs 1-9
    (Dark + Kernel). The Bright output (index 0) is excluded because it is
    dominated by the stellar flux, making it less sensitive to planets and
    more prone to systematic biases from imperfect f_star estimation.

    Parameters
    ----------
    params : np.ndarray, shape (3 * n_sources,)
        Flattened parameter vector encoding all sources. For n_sources sources:
          params[3*s]   = x_mas of source s (East offset in mas)
          params[3*s+1] = y_mas of source s (North offset in mas)
          params[3*s+2] = contrast c_s of source s (dimensionless, ≥ 0)
        Example for 2 sources: [x1, y1, c1, x2, y2, c2]
    data : np.ndarray, shape (N, 10)
        Measured photon counts.
    maps : np.ndarray, shape (N, 10, M, M)
        Transmission maps for all angles and outputs.
    f_star : float
        Estimated stellar flux in photons.
    cov : np.ndarray, shape (N, 10)
        Diagonal covariance matrix from compute_data_covariance().
        Used as the denominator in the chi-squared sum.
    fov : float
        Full field of view in mas. Passed to interp_transmission() for
        coordinate conversion, and used for out-of-bounds checking.
    use_dark_only : bool
        Currently unused (kept for API compatibility). Always uses outputs
        1-9 (all Dark and Kernel outputs).

    Returns
    -------
    chi2 : float
        Weighted chi-squared value. Lower is better. Used by scipy.optimize
        as the scalar objective to minimise. Dimensionless (photons² / photons²).
        Typical well-fit values: reduced chi² ≈ 1.0 for a correct model.
    """
    N, n_out, M, _ = maps.shape
    n_sources = len(params) // 3
    cx = M // 2

    # Which outputs to use
    out_indices = list(range(1, 10))  # Dark + Kernel (exclude Bright)

    chi2 = 0.0
    for i in range(N):
        # Model prediction for each output (in photons)
        model = maps[i, :, cx, cx] * f_star  # star at centre

        # Planet contributions
        for s in range(n_sources):
            x_mas    = params[3 * s]
            y_mas    = params[3 * s + 1]
            contrast = params[3 * s + 2]

            # Penalty for out-of-bounds position
            if abs(x_mas) > fov / 2 or abs(y_mas) > fov / 2:
                chi2 += 1e6
                continue

            T = interp_transmission(maps[i], x_mas, y_mas, fov)
            model = model + T * f_star * contrast

        for o in out_indices:
            diff = data[i, o] - model[o]
            # Normalise by f_star so residuals are O(contrast) ~ O(1e-4 to 1e-1)
            # This prevents numerical issues when f_star ~ 1e13
            diff_norm = diff / f_star
            cov_norm  = max(cov[i, o], 1.0) / (f_star ** 2)
            chi2 += diff_norm ** 2 / max(cov_norm, 1e-30)

    return chi2


def refine_sources(sources_init: list[dict],
                   data: np.ndarray,
                   maps: np.ndarray,
                   f_star: float,
                   cov: np.ndarray,
                   fov: float) -> list[dict]:
    """
    Refine the positions and contrasts of source candidates via parametric
    chi-squared minimisation over all Dark and Kernel outputs.

    This is the second main step of the pipeline. Whereas the LASSO step
    (Step 1) works only on Kernel outputs and is limited to pixel-grid
    resolution, this step:
      - Uses all 9 outputs (6 Dark + 3 Kernel), providing more constraints
      - Operates in continuous (x_mas, y_mas, contrast) space, giving
        sub-pixel astrometric precision
      - Uses the piston-inflated covariance from compute_data_covariance()
        to down-weight the less reliable Dark outputs appropriately
      - Fits all sources simultaneously to account for cross-source confusion

    Optimisation strategy:
        The LASSO image provides a good initial guess close to the true
        solution, so a local gradient-based method (L-BFGS-B) is sufficient.
        L-BFGS-B handles the box constraints (x, y within FoV; c ≥ 0)
        natively. The objective function is parametric_residual().

    Parameter bounds:
        x_mas ∈ [-fov/2, +fov/2]
        y_mas ∈ [-fov/2, +fov/2]
        contrast ∈ [0.0, 1.0]

    Degrees of freedom:
        n_dof = N * 9 - 3 * n_sources
        (N angles × 9 outputs, minus 3 free parameters per source)

    The reduced chi-squared χ²_red = χ²_opt / n_dof is a goodness-of-fit
    indicator. χ²_red ≈ 1 indicates the model fits the data within noise.
    χ²_red >> 1 may indicate additional sources, a bad noise model, or
    systematic errors in the transmission maps.

    Note on SNR proxy:
        The snr_proxy returned is a rough heuristic (contrast / scaled
        residual), NOT a rigorous detection significance. A proper SNR
        would require computing sqrt(diag((A^T C^-1 A)^-1)), which is not
        currently implemented for performance reasons.

    Parameters
    ----------
    sources_init : list of dict
        Initial source candidates from find_sources_from_image(). Each dict
        must contain 'x_mas', 'y_mas', 'contrast_init'. Can contain multiple
        sources which are fitted simultaneously.
    data : np.ndarray, shape (N, 10)
        Measured photon counts.
    maps : np.ndarray, shape (N, 10, M, M)
        Transmission maps.
    f_star : float
        Estimated stellar flux in photons.
    cov : np.ndarray, shape (N, 10)
        Diagonal data covariance from compute_data_covariance().
    fov : float
        Full field of view in mas. Used for bounds and coordinate conversion.

    Returns
    -------
    sources_out : list of dict
        One dict per source, with refined parameters:
          'x_mas'        : float — refined East angular offset in mas
          'y_mas'        : float — refined North angular offset in mas
          'contrast'     : float — refined contrast (dimensionless, ≥ 0)
          'snr_proxy'    : float — heuristic signal-to-noise proxy
                                   (not a rigorous detection significance)
          'reduced_chi2' : float — reduced chi-squared of the global fit,
                                   shared by all sources in this call
        Same length as sources_init. Returns empty list if sources_init
        is empty.
    """
    if not sources_init:
        return []

    n_sources = len(sources_init)
    half_fov = fov / 2.0

    # Initial parameter vector
    p0 = []
    bounds = []
    for s in sources_init:
        p0 += [s['x_mas'], s['y_mas'], s.get('contrast_init', s.get('contrast'))]
        bounds += [(-half_fov, half_fov),
                   (-half_fov, half_fov),
                   (0.0, 1.0)]

    p0 = np.array(p0)

    # Local optimisation from LASSO initialisation
    result = minimize(
        parametric_residual,
        p0,
        args=(data, maps, f_star, cov, fov),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-12}
    )

    params_opt = result.x
    chi2_opt = result.fun

    # Parse results
    N = data.shape[0]
    n_dof = N * 9 - len(params_opt)  # 9 outputs (Dark+Kernel)
    reduced_chi2 = chi2_opt / max(n_dof, 1)

    sources_out = []
    for s in range(n_sources):
        x_mas = params_opt[3 * s]
        y_mas = params_opt[3 * s + 1]
        contrast = params_opt[3 * s + 2]

        # Simple SNR estimate: contrast / uncertainty from covariance
        # Uncertainty ≈ sqrt(diag of (A^T C^-1 A)^-1) — approximated here
        snr = contrast / max(np.sqrt(reduced_chi2) * 1e-3, 1e-12)

        sources_out.append({
            'x_mas': float(x_mas),
            'y_mas': float(y_mas),
            'contrast': float(contrast),
            'snr_proxy': float(snr),
            'reduced_chi2': float(reduced_chi2)
        })

    return sources_out


# ===========================================================================
# Step 3 — Field rotation consistency check
# ===========================================================================

def field_rotation_check(sources: list[dict],
                          data: np.ndarray,
                          maps: np.ndarray,
                          f_star: float,
                          fov: float) -> list[dict]:
    """
    Validate each detected source by testing its consistency with field rotation.

    Physical principle:
        As the Earth rotates, the projected baseline geometry of the
        interferometer changes with hour angle h. This causes the transmission
        pattern T_h(r, θ) at any fixed sky position (r, θ) to vary with h.
        A real astrophysical source at (r, θ) will therefore produce a Dark
        output signal that tracks T_h(r, θ) * f * c as h varies.

        An instrumental artefact (e.g. a bad pixel, a persistent speckle, a
        cross-talk signal) is fixed in the detector frame and does NOT follow
        this rotation. Its apparent signal will not correlate with the
        predicted T_h(r, θ) curve.

    Method:
        For each source candidate (x_mas, y_mas, contrast), and for each of
        the 6 Dark outputs:
          1. Compute the measured residual over all N angles:
               δd[i, o] = data[i, o] - T_o(0, 0, i) * f_star
          2. Compute the predicted signal from the source over all N angles:
               pred[i, o] = T_o(x_mas, y_mas, i) * f_star * contrast
          3. Compute the Pearson correlation coefficient between δd[:, o]
             and pred[:, o] across the N angles.

        The mean correlation across the 6 Dark outputs is stored as
        'field_rotation_corr_mean'. A source is flagged as consistent
        (is_consistent = True) if this mean correlation exceeds 0.3.

    Interpretation:
        - corr_mean > 0.5 : strong evidence for a real source
        - corr_mean 0.3-0.5 : marginal, worth further investigation
        - corr_mean < 0.3 : source position is not well-supported by the
                            field rotation pattern — likely spurious
        - Negative correlation : the source model predicts the wrong sign
                                 of variation — very suspicious

    Limitations:
        - Requires sufficient diversity in hour angle (N should span a
          meaningful range of parallactic angles). With very few angles
          or a small parallactic rotation, all correlations will be low.
        - Uses only the 6 Dark outputs (not Kernel) because the Kernel maps
          are antisymmetric and their temporal variation is harder to
          interpret directly in correlation terms.
        - The correlation threshold (0.3) is a heuristic; adjust based on
          your knowledge of the parallactic angle coverage.

    Parameters
    ----------
    sources : list of dict
        Source candidates from refine_sources(). Each dict must contain
        'x_mas', 'y_mas', 'contrast'. Modified in-place to add new keys.
    data : np.ndarray, shape (N, 10)
        Measured photon counts.
    maps : np.ndarray, shape (N, 10, M, M)
        Transmission maps. maps[:, 1:7, M//2, M//2] used for star subtraction.
    f_star : float
        Estimated stellar flux in photons.
    fov : float
        Full field of view in mas. Passed to interp_transmission().

    Returns
    -------
    sources : list of dict
        Same list as input, with three new keys added to each source dict:
          'field_rotation_corr_mean'      : float  — mean Pearson correlation
                                            across the 6 Dark outputs
          'field_rotation_corr_per_output': list[float] of length 6 — per-output
                                            Pearson correlation coefficients
                                            (indices correspond to Dark 1-6)
          'is_consistent'                 : bool — True if corr_mean > 0.3,
                                            indicating the source is consistent
                                            with field rotation
    """
    N, n_out, M, _ = maps.shape
    cx = M // 2

    for src in sources:
        correlations = []
        for o in range(1, 7):  # Dark outputs only
            # Measured residual
            delta_d = data[:, o] - maps[:, o, cx, cx] * f_star

            # Predicted signal from this source
            predicted = np.array([
                interp_transmission(maps[i], src['x_mas'], src['y_mas'], fov)[o]
                * f_star * src['contrast']
                for i in range(N)
            ])

            if np.std(predicted) > 0 and np.std(delta_d) > 0:
                corr = float(np.corrcoef(delta_d, predicted)[0, 1])
            else:
                corr = 0.0
            correlations.append(corr)

        src['field_rotation_corr_mean'] = float(np.mean(correlations))
        src['field_rotation_corr_per_output'] = correlations
        src['is_consistent'] = src['field_rotation_corr_mean'] > 0.3

    return sources


# ===========================================================================
# Utility — Build final image from parametric sources
# ===========================================================================

def subtract_source(data: np.ndarray,
                    maps: np.ndarray,
                    f_star: float,
                    x_mas: float,
                    y_mas: float,
                    contrast: float,
                    fov: float) -> np.ndarray:
    """
    Subtract the predicted signal of a single point source from the data.

    This is the core operation of the iterative CLEAN loop. After a source
    has been detected and its parameters (x_mas, y_mas, contrast) refined,
    its expected contribution to each (angle, output) measurement is computed
    from the transmission maps and removed from the data. The remaining
    residual data then contains only the contributions of fainter sources
    (and noise), allowing the next LASSO iteration to detect them without
    being dominated by the bright source that was just peeled.

    Model contribution of this source at angle i, output o:
        signal[i, o] = T_o(x_mas, y_mas, i) * f_star * contrast

    The subtraction is applied to ALL 10 outputs (including Bright), since
    the residual data is used for the next CLEAN iteration, which may inspect
    any output. The stellar term T_o(0,0,i)*f_star is NOT subtracted here —
    it is handled separately in build_kernel_system via the centre-pixel term.

    Note on bias: if the contrast estimate is slightly wrong (which is always
    the case due to noise), the residual will retain a fraction of the source
    signal. This is why a joint re-refinement of all sources on the original
    data is performed at the end of the CLEAN loop (Step 2b in reconstruct),
    which corrects these sequential bias accumulations simultaneously.

    Parameters
    ----------
    data : np.ndarray, shape (N, 10)
        Current residual data (photon counts). On the first CLEAN iteration,
        this is the original data. On subsequent iterations, it is the data
        after previous sources have been subtracted. NOT modified in-place —
        a new array is returned.
    maps : np.ndarray, shape (N, 10, M, M)
        Transmission maps for all angles and outputs.
    f_star : float
        Estimated stellar flux in photons.
    x_mas : float
        Refined East angular offset of the source to subtract, in mas.
    y_mas : float
        Refined North angular offset of the source to subtract, in mas.
    contrast : float
        Refined contrast of the source to subtract (dimensionless, ≥ 0).
    fov : float
        Full field of view in mas. Passed to interp_transmission().

    Returns
    -------
    data_residual : np.ndarray, shape (N, 10)
        New data array with the source signal subtracted.
        data_residual[i, o] = data[i, o] - T_o(x_mas, y_mas, i) * f_star * contrast
        Always a new array (input data is not modified).
    """
    N = data.shape[0]
    data_residual = data.copy()
    for i in range(N):
        T = interp_transmission(maps[i], x_mas, y_mas, fov)  # (10,)
        data_residual[i] -= T * f_star * contrast
    return data_residual


def sources_to_image(sources: list[dict], M: int, fov: float) -> np.ndarray:
    """
    Build a 2D contrast image by placing detected point sources on a pixel grid.

    Each source is represented as a single bright pixel at the nearest grid
    point to its refined (x_mas, y_mas) position, with a pixel value equal
    to its contrast. All other pixels are zero.

    This is a simple "delta function" representation — it does not convolve
    with a PSF. The output is intended as a compact, readable summary of the
    reconstruction result, suitable for display and comparison with the raw
    LASSO image. For scientific measurements, use the 'sources' list directly
    (which gives sub-pixel astrometry).

    Coordinate convention:
        Map centre (pixel M//2, M//2) corresponds to (0, 0) mas.
        x_mas → column index:  x_pix = round(x_mas / pixel_scale + M/2)
        y_mas → row index:     y_pix = round(y_mas / pixel_scale + M/2)
        Positions outside the grid are clamped to the boundary.

    Parameters
    ----------
    sources : list of dict
        Refined sources from field_rotation_check() (or refine_sources()).
        Each dict must contain:
          'x_mas'    : float — East angular offset in mas
          'y_mas'    : float — North angular offset in mas
          'contrast' : float — dimensionless contrast value (planet/star)
    M : int
        Image size in pixels (M × M). Should match the map resolution.
    fov : float
        Full field of view in mas. Used to compute pixel_scale = fov / M.

    Returns
    -------
    image : np.ndarray, shape (M, M), dtype float64
        2D contrast image. Zero everywhere except at source positions, where
        the pixel value equals the source contrast. If two sources map to the
        same pixel, the second overwrites the first (last-write wins) — this
        should not occur in practice for well-separated sources.
    """
    pixel_scale = fov / M
    image = np.zeros((M, M))
    for src in sources:
        x_pix = int(round(src['x_mas'] / pixel_scale + M / 2))
        y_pix = int(round(src['y_mas'] / pixel_scale + M / 2))
        x_pix = np.clip(x_pix, 0, M - 1)
        y_pix = np.clip(y_pix, 0, M - 1)
        image[y_pix, x_pix] = src['contrast']
    return image


# ===========================================================================
# Main entry point
# ===========================================================================

def reconstruct(data: np.ndarray,
                maps: np.ndarray,
                piston_rms: float,
                fov: float = 10.0,
                lasso_lambda: float = None,
                n_sources_max: int = 5,
                detection_threshold_sigma: float = 3.0,
                verbose: bool = True) -> dict:
    """
    Full nulling interferometry image reconstruction pipeline,
    with iterative CLEAN source peeling for multi-source fields.

    Orchestrates the following pipeline:

      Step 0 — Stellar flux estimation
          Estimates f from the Bright output using a robust median estimator.
          All subsequent steps depend on this estimate.

      Steps 1+2 — Iterative CLEAN loop  (repeated up to n_sources_max times)
          The naive approach of running LASSO once and detecting all sources
          simultaneously fails when sources have very different brightnesses:
          the L1 penalty that is weak enough to let the faint source through
          is also too weak to suppress LASSO artefacts around the bright one,
          and the penalty strong enough to clean the bright source artefacts
          kills the faint source signal entirely.

          The CLEAN loop solves this by peeling sources one by one:
            1a. Run Kernel LASSO on the current residual data
                → detect the single brightest remaining source candidate
            1b. Refine its (x_mas, y_mas, contrast) parametrically against
                all Dark+Kernel outputs
            1c. Subtract its predicted signal from the residual data
            Repeat until no source is detected or n_sources_max is reached.

          This is the interferometric analogue of the CLEAN algorithm used
          in radio aperture synthesis imaging.

      Step 2b — Joint re-refinement
          After CLEAN peeling, all detected sources are re-fitted jointly
          on the original (un-peeled) data. This corrects the sequential
          bias accumulated during peeling: each subtraction used a slightly
          imprecise contrast estimate, which leaves residuals that bias the
          next source's detection. A joint fit has no such bias.

      Step 3 — Field rotation consistency check
          Computes the Pearson correlation between measured residuals and
          source model predictions across hour angles. Sources that do not
          follow the expected field rotation pattern are flagged as suspicious.

    Parameters
    ----------
    data : np.ndarray, shape (N, 10)
        Measured photon counts (or ADU with gain=1) for N hour angles and
        10 interferometric outputs.
        Column ordering (fixed):
          [0]    Bright output
          [1-6]  Dark outputs 1 through 6
          [7-9]  Kernel outputs 1 through 3
        Units: photons (Poisson statistics assumed).
    maps : np.ndarray, shape (N, 10, M, M)
        Transmission maps corresponding to each row of data. maps[i] gives
        the 10 maps (M×M each) for hour angle index i. Same column ordering
        as data. The star is assumed to be at the map centre (M//2, M//2).
        Maps should be normalised such that the peak Bright transmission is
        close to 1.0.
    piston_rms : float
        Estimated RMS atmospheric piston per telescope in nanometres.
        This is the main instrument-characterisation input. It controls the
        piston noise floor added to the data covariance. Typical values for
        ground-based interferometry: 50-500 nm depending on conditions and
        AO correction quality.
    fov : float, optional
        Full angular field of view in mas. Default 10.0 (i.e. ±5 mas).
        Used to convert pixel coordinates to angular coordinates and vice versa.
        Must match the angular extent of the provided maps.
    lasso_lambda : float or None, optional
        L1 regularisation strength for the LASSO step. If None (default),
        auto-set to 5% of max(|A^T b|) — a data-driven heuristic.
        Increase to suppress faint spurious detections.
        Decrease if known planets are being missed by the LASSO step.
    n_sources_max : int, optional
        Maximum number of point source candidates to detect and refine.
        Default 5. Set to 1 if you are certain of a single-planet system
        to speed up computation and avoid false positives.
    detection_threshold_sigma : float, optional
        Detection threshold for the LASSO image, in units of the image noise
        standard deviation. Default 3.0 (3-sigma). Lower this (e.g. to 2.0)
        to detect fainter sources at the cost of more false positives, which
        the field rotation check (Step 3) will then help to reject.
    verbose : bool, optional
        If True (default), print step-by-step progress with key results to
        stdout. Set to False for silent operation in batch processing.

    Returns
    -------
    result : dict
        Dictionary with the following keys:

        'f_star' : float
            Estimated stellar flux in photons per integration, from Step 0.
            Accounts for collecting area, integration time, throughput, etc.

        'f_star_std' : float
            1-sigma uncertainty on f_star (MAD-based, robust). Reflects
            photon noise and angle-to-angle variability.

        'image_lasso' : np.ndarray, shape (M, M)
            2D contrast image from the Kernel LASSO reconstruction (Step 1).
            Most pixels are 0.0; bright spots indicate source candidates.
            Use this for a first visual inspection of the field.

        'image_final' : np.ndarray, shape (M, M)
            2D image with each refined source placed as a single bright pixel
            at its fitted position, with value equal to its contrast. Built
            from sources_to_image(). Zero array if no sources were detected.

        'sources' : list of dict
            One dict per detected and refined source. Each dict contains:
              'x_mas'                       : float — refined East offset (mas)
              'y_mas'                       : float — refined North offset (mas)
              'contrast'                    : float — planet/star flux ratio
              'snr_proxy'                   : float — heuristic SNR estimate
              'reduced_chi2'                : float — reduced χ² of the fit
              'field_rotation_corr_mean'    : float — mean field rotation corr.
              'field_rotation_corr_per_output': list[float] — per-Dark corr.
              'is_consistent'               : bool — passes field rotation test
            Empty list if no sources were detected above the threshold.

        'pixel_scale' : float
            Angular size of one map pixel in mas/pixel = fov / M.

        'reduced_chi2' : float or None
            Reduced chi-squared of the final parametric fit (Step 2).
            None if no sources were detected.

    Raises
    ------
    AssertionError
        If data.shape != (N, 10) or maps.shape != (N, 10, M, M).
    ValueError
        If the Bright transmission at the map centre is zero for all angles
        (propagated from estimate_stellar_flux).

    Examples
    --------
    >>> result = reconstruct(data, maps, piston_rms=100.0, fov=10.0)
    >>> print(result['sources'])
    >>> plot_results(result, fov=10.0)
    """
    data = np.asarray(data, dtype=float)
    maps = np.asarray(maps, dtype=float)

    N, n_out, M, _ = maps.shape
    pixel_scale = fov / M  # mas/pixel

    assert data.shape == (N, 10), f"data must be (N,10), got {data.shape}"
    assert maps.shape == (N, 10, M, M), f"maps must be (N,10,M,M), got {maps.shape}"

    # ------------------------------------------------------------------
    # Step 0: Stellar flux estimation
    # ------------------------------------------------------------------
    if verbose:
        print("[Step 0] Estimating stellar flux from Bright output...")
    f_star, f_star_std = estimate_stellar_flux(data, maps)
    if verbose:
        print(f"         f_star = {f_star:.3e} ± {f_star_std:.3e} photons")

    # ------------------------------------------------------------------
    # Covariance (shot + piston)
    # ------------------------------------------------------------------
    if verbose:
        print("[Cov]    Building data covariance (shot + piston)...")
    cov = compute_data_covariance(data, maps, f_star, piston_rms)

    # ------------------------------------------------------------------
    # Steps 1+2: Iterative CLEAN loop
    # ------------------------------------------------------------------
    # Each iteration:
    #   a) LASSO on residual data (Kernel outputs only)  → find brightest source
    #   b) Parametric refinement of that single source   → precise (x,y,c)
    #   c) Subtract the refined source from the data     → updated residuals
    #   d) Repeat until no new source is detected or max_sources reached
    #
    # This avoids the confusion problem: a bright source monopolises the L1
    # penalty and suppresses faint sources. By peeling sources one by one,
    # the faint residuals are progressively unveiled.
    # ------------------------------------------------------------------
    all_sources   = []       # accumulates all refined sources across iterations
    image_lasso   = None     # LASSO image from the first iteration (for display)
    data_residual = data.copy()  # starts as raw data, gets source-subtracted

    for iteration in range(n_sources_max):
        if verbose:
            print(f"\n[CLEAN iter {iteration + 1}/{n_sources_max}]")

        # --- Step 1a: Kernel LASSO on residual data ----------------------
        if verbose:
            print("  [1] Kernel LASSO on residual data...")
        A_K, b_K, W_K = build_kernel_system(data_residual, maps, f_star, cov)
        x_lasso_iter  = lasso_reconstruct(A_K, b_K, lambda_reg=lasso_lambda)
        image_lasso_iter = x_lasso_iter.reshape(M, M)

        if verbose:
            pos_pix = image_lasso_iter[image_lasso_iter > 0]
            if len(pos_pix) > 0:
                print(f"     LASSO max={image_lasso_iter.max():.3e}, "
                      f"n_positive={len(pos_pix)}, "
                      f"median_positive={np.median(pos_pix):.3e}")
            else:
                print("     All LASSO pixels zero — no more sources.")

        # Save the first-iteration LASSO image for diagnostic display
        if iteration == 0:
            image_lasso = image_lasso_iter

        # --- Step 1b: detect ONE source (the brightest) ------------------
        candidates = find_sources_from_image(
            image_lasso_iter, pixel_scale, fov,
            n_sources_max=1,   # only the brightest per iteration
            threshold_sigma=detection_threshold_sigma
        )

        if not candidates:
            if verbose:
                print("     No source detected above threshold — stopping CLEAN.")
            break

        src_init = candidates[0]
        if verbose:
            print(f"     Candidate: ({src_init['x_mas']:.2f}, "
                  f"{src_init['y_mas']:.2f}) mas  "
                  f"contrast_init={src_init['contrast_init']:.3e}")

        # --- Step 2: parametric refinement of this single source ---------
        if verbose:
            print("  [2] Parametric refinement on residual data...")
        refined = refine_sources(
            [src_init], data_residual, maps, f_star, cov, fov
        )
        if not refined:
            if verbose:
                print("     Refinement failed — stopping CLEAN.")
            break

        src = refined[0]
        if verbose:
            print(f"     Refined: ({src['x_mas']:.2f}, {src['y_mas']:.2f}) mas  "
                  f"contrast={src['contrast']:.3e}  "
                  f"χ²_red={src['reduced_chi2']:.3f}")

        # --- Source subtraction: peel this source from the data ----------
        # Subtract model signal of this source from all outputs (Dark+Kernel)
        # at all N angles, so the next LASSO iteration works on cleaner data.
        data_residual = subtract_source(
            data_residual, maps, f_star,
            src['x_mas'], src['y_mas'], src['contrast'], fov
        )
        if verbose:
            print(f"     Source subtracted from residual data.")

        all_sources.append(src)

    if verbose:
        print(f"\n[CLEAN] {len(all_sources)} source(s) found in total.")

    if not all_sources:
        return {
            'f_star':      f_star,
            'f_star_std':  f_star_std,
            'image_lasso': image_lasso if image_lasso is not None
                           else np.zeros((M, M)),
            'image_final': np.zeros((M, M)),
            'sources':     [],
            'pixel_scale': pixel_scale,
            'reduced_chi2': None
        }

    # ------------------------------------------------------------------
    # Step 2b: joint re-refinement of ALL sources together on the
    # original (non-peeled) data, now that we have good initial positions
    # for everyone. This corrects for any bias introduced by the sequential
    # peeling (each subtraction used a slightly imprecise contrast estimate).
    # ------------------------------------------------------------------
    if verbose:
        print("\n[Step 2b] Joint re-refinement of all sources on original data...")
    all_sources_joint = refine_sources(
        all_sources, data, maps, f_star, cov, fov
    )
    if verbose:
        for s in all_sources_joint:
            print(f"         ({s['x_mas']:.2f}, {s['y_mas']:.2f}) mas  "
                  f"contrast={s['contrast']:.3e}  "
                  f"χ²_red={s['reduced_chi2']:.3f}")

    # ------------------------------------------------------------------
    # Step 3: Field rotation consistency check
    # ------------------------------------------------------------------
    if verbose:
        print("\n[Step 3] Field rotation consistency check...")
    sources_final = field_rotation_check(
        all_sources_joint, data, maps, f_star, fov
    )
    if verbose:
        for s in sources_final:
            status = "✓ consistent" if s['is_consistent'] else "✗ suspicious"
            print(f"         ({s['x_mas']:.2f}, {s['y_mas']:.2f}) mas  "
                  f"mean corr={s['field_rotation_corr_mean']:.2f}  [{status}]")

    # ------------------------------------------------------------------
    # Build final image
    # ------------------------------------------------------------------
    image_final  = sources_to_image(sources_final, M, fov)
    reduced_chi2 = sources_final[0]['reduced_chi2'] if sources_final else None

    return {
        'f_star':      f_star,
        'f_star_std':  f_star_std,
        'image_lasso': image_lasso,
        'image_final': image_final,
        'sources':     sources_final,
        'pixel_scale': pixel_scale,
        'reduced_chi2': reduced_chi2
    }


# ===========================================================================
# Optional: quick diagnostic plot
# ===========================================================================

def plot_results(result: dict, fov: float = 10.0):
    """
    Generate a quick two-panel diagnostic plot of the reconstruction output.

    Panel 1 — Kernel LASSO image:
        Shows result['image_lasso'], the raw sparse image from Step 1.
        Detected sources are overlaid as markers:
          - Cyan circle  : source passed the field rotation consistency check
          - Red cross    : source flagged as suspicious (is_consistent = False)
        Each marker is labelled with the source contrast value.

    Panel 2 — Parametric reconstruction:
        Shows result['image_final'], the delta-function image from Step 2.
        Each pixel represents a refined source at sub-pixel precision,
        with value equal to the fitted contrast.

    The figure title shows the estimated stellar flux f_star and the
    reduced chi-squared of the parametric fit.

    Parameters
    ----------
    result : dict
        Output dictionary from reconstruct(). Must contain keys:
          'image_lasso', 'image_final', 'sources', 'f_star', 'reduced_chi2'.
    fov : float, optional
        Full field of view in mas. Default 10.0. Used to set the axis extent
        so that axes are labelled in mas rather than pixel indices.
        Should match the fov passed to reconstruct().

    Returns
    -------
    None
        Displays the figure via plt.show(). Does not return the figure object.
        To save the figure, call plt.savefig() before plt.show(), or modify
        this function to return fig.
    """
    import matplotlib.pyplot as plt

    M = result['image_lasso'].shape[0]
    extent = [-fov / 2, fov / 2, -fov / 2, fov / 2]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    im0 = axes[0].imshow(result['image_lasso'], origin='lower',
                          extent=extent, cmap='hot')
    axes[0].set_title('Kernel LASSO image')
    axes[0].set_xlabel('θ_x (mas)')
    axes[0].set_ylabel('θ_y (mas)')
    plt.colorbar(im0, ax=axes[0], label='Contrast')

    # Mark detected sources
    for s in result['sources']:
        marker = 'o' if s.get('is_consistent', True) else 'x'
        color  = 'cyan' if s.get('is_consistent', True) else 'red'
        axes[0].plot(s['x_mas'], s['y_mas'], marker, color=color,
                     ms=8, label=f"c={s['contrast']:.2e}")
    if result['sources']:
        axes[0].legend(fontsize=7)

    im1 = axes[1].imshow(result['image_final'], origin='lower',
                          extent=extent, cmap='hot')
    axes[1].set_title('Parametric reconstruction')
    axes[1].set_xlabel('θ_x (mas)')
    axes[1].set_ylabel('θ_y (mas)')
    plt.colorbar(im1, ax=axes[1], label='Contrast')

    plt.suptitle(
        f"f★ = {result['f_star']:.2e} ph  |  "
        f"χ²_red = {result['reduced_chi2']:.2f}" if result['reduced_chi2'] else
        f"f★ = {result['f_star']:.2e} ph"
    )
    plt.tight_layout()
    plt.show()