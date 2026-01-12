"""
Streamlit page for Kernel Nulling explanation and simulation.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import astropy.units as u
from copy import deepcopy as copy

# --- Path Setup ---
ROOT = Path(__file__).parent.parent.parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

WEB = ROOT / "web"
if str(WEB) not in sys.path:
    sys.path.insert(0, str(WEB))

from phise import Context
import phise.classes.archs.superkn as superkn_module
import importlib
importlib.reload(superkn_module)
from phise.classes.archs.superkn import SuperKN
from utils.context_widget import context_widget

st.set_page_config(
    page_title="Kernel Nulling",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Kernel Nulling 🧬")

st.markdown(r"""
## Principle

**Kernel Nulling** is an interferometric post-processing technique designed to reject optical path differences (piston errors) that limit the nulling depth of classical interferometers.

Unlike classical nulling, which aims to minimize starlight through destructive interference, Kernel Nulling combines multiple "dark" outputs to form a robust observable called **Kernel**.

## Demonstration

For Kernel Nulling, we construct two combinations of signals placed in **phase quadrature**.

An output intensity is a linear combination of the input fields:
$$
I = |E_1 + E_2 e^{i\frac{\pi}{2}} + E_3 e^{i\pi}+ E_4 e^{i\frac{3\pi}{2}}|^2
$$
Where $E = a e^{i\phi}$.

If we introduce small perturbations $\theta$ on these fields, using the Taylor expansion $e^{i\theta} \approx 1 + i\theta$, we can express a pair of output intensities $I_A$ and $I_B$ (corresponding to different mixing matrices) such as:

$$
\begin{aligned}
I_A &\approx |a - a(1 + i\theta_2) + ia(1 + i\theta_3) - ia(1 + i\theta_4)|^2 \\
I_B &\approx |a - a(1 + i\theta_2) - ia(1 + i\theta_3) + ia(1 + i\theta_4)|^2
\end{aligned}
$$

By simplifying the terms, we find that both outputs have the same second-order dependency on phase errors:

$$
I_A \approx I_B \approx |a|^2 \times (\theta_2^2 + (\theta_3 - \theta_4)^2)
$$

Therefore, by subtracting them to form the Kernel, the common noise terms cancel out:

$$
K = I_A - I_B \approx 0
$$

This demonstrates that the Kernel observable effectively cancels out second-order piston noise, leaving only the signal from the astrophysical scene (e.g., a planet).

### Key Properties

- **Robustness to Piston**: The Kernel is insensitive to second-order piston noise (for small aberrations).
- **Planet Detection**: The presence of a planet breaks symmetry and produces a non-zero Kernel signal.
""")
st.divider()

# =============================================================================
# Interactive Simulation
# =============================================================================

st.header("Interactive Simulation")

# Configure Context
ctx = context_widget()

# Force ideal component parameters (φ=0, σ=0) for this simulation
ctx.interferometer.chip.φ = np.zeros(14) * u.nm
ctx.interferometer.chip.σ = np.zeros(14) * u.nm
st.info("ℹ️ **Simulation Note:** The photonic component is forced to be **ideal** (no phase errors, no manufacturing defects) for this demonstration.")

col_left, col_right = st.columns([1, 1.5])

# --- Controls (Left Column) ---
with col_left:
    st.subheader("Parameters")
    
    # 1. Atmosphere / Instrument
    st.caption("**Atmosphere & Instrument**")
    gamma_nm = st.slider(
        "RMS Piston Error Γ (nm)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        help="Standard deviation of random phase delays introduced by the atmosphere or instrument."
    )
    
    # 2. Planet
    st.caption("**Companion (Planet)**")
    # Limit separation to FOV/2
    max_sep = float(ctx.interferometer.fov.to(u.mas).value / 2)
    sep_mas = st.slider(
        "Separation (mas)",
        min_value=0.0,
        max_value=max_sep,
        value=max_sep/2,
        # step=1.0 # Let streamlit decide or keep 10? 
        # Variable step is safer if FOV is small, but let's try to keep it granular enough
        step=max(0.1, max_sep/100.0) 
    )
    pa_deg = st.slider(
        "Polar Angle (0°=East)",
        min_value=0.0,
        max_value=360.0,
        value=0.0,
        step=1.0
    )
    log_contrast = st.slider(
        "Contrast (log10)",
        min_value=-8.0,
        max_value=-1.0,
        value=-4.0,
        step=0.1,
        help="Logarithm of the Planet/Star flux ratio"
    )
    contrast = 10**log_contrast

# --- Logic & Calculations ---

# Context is setup above

# Apply interactive simulation overrides
# Gamma override (from slider)
ctx.Γ = gamma_nm * u.nm

# Setup Planet (Override context companions for demonstration)
# We want to maintain whatever the user set in the widget for the star/telescopes,
# but we force the companion list to match our interactive control.
ctx.target.companions = [] # Clear existing logic
from phise.classes.companion import Companion
ctx.target.companions.append(Companion(
    name="Planet",
    c=contrast,
    ρ=sep_mas * u.mas,
    θ=pa_deg * u.deg # Math Convention: 0=East, 90=North
))

# Ensure we have a SuperKN chip (Kernel Nuller)
# Note: Context.get_VLTI might have a standard chip. We force a SuperKN if needed or use existing if compatible.
# For demonstration, we'll instantiate a clean SuperKN logic locally or use the one from context if it's already a SuperKN.
if not isinstance(ctx.interferometer.chip, SuperKN):
    # Create a default SuperKN chip
    # Assuming VLTI geometry inputs match SuperKN expectations (4 beams)
    # We need a dummy phi/sigma for initialization
    phi = np.zeros(14) * u.nm
    sigma = np.zeros(14) * u.nm
    # For a canonical Kernel Nuller, we might need specific phase shifts.
    # SuperKN defaults might be sufficient or require calibration.
    # Let's assume standard defaults for now.
    kn_chip = SuperKN(φ=phi, σ=sigma, λ0=ctx.interferometer.λ)
    ctx.interferometer.chip = kn_chip



# --- Visualization (Right Column) ---
with col_right:
    st.subheader("Phase Visualization")
    
    # We focus on the first Kernel (K1) formed by Dark 1 and Dark 2
    # Output indices: Dark 1 = 1, Dark 2 = 2 (in 0-indexed raw outputs of SuperKN, Bright=0)
    
    # Propagate Star and Planet SEPARATELY to see phasors
    
    # 1. Get input fields (complex) for Star and Planet
    # Context.get_input_fields returns (n_sources, n_telescopes)
    # Source 0 is star, Source 1 is planet
    input_fields = ctx.get_input_fields()
    
    psi_star = input_fields[0]
    psi_planet = input_fields[1]
    
    # 2. Get output fields for these inputs
    # SuperKN.get_output_fields returns (7,) [Bright, D1, D2, D3, D4, D5, D6]
    # We assume standard SuperKN logic (no extra random piston here, we want the "ideal" response or mean response?)
    # The user asked for "contribution de chaque télescope". 
    # To do this, we need to propagate each telescope individually.
    
    def get_telescope_phasors(chip, inputs, wavelength):
        """Returns phasors for each telescope at each output."""
        # inputs: (4,) complex
        phasors = np.zeros((4, 7), dtype=complex)
        for t in range(4):
            psi_t = np.zeros(4, dtype=complex)
            psi_t[t] = inputs[t]
            phasors[t] = chip.get_output_fields(ψ=psi_t, λ=wavelength)
        return phasors

    # Wavelength
    wl = ctx.interferometer.λ
    
    # Compute phasors for Star (ideal, no piston error for visualization of "contribution")
    # If we want to show the effect of Gamma on the phasors, we should add random piston to psi_star?
    # Usually "contribution" plots are static ideal phasors, but user said "jouer avec ... l'erreur de piston ... et voir sur des polar plot".
    # So we should apply a realization of the piston error.
    
    # Generate ONE realization of piston error
    piston_error = np.random.normal(0, ctx.Γ.to(u.nm).value, 4) * u.nm
    from phise.modules import phase
    
    psi_star_corrupted = phase.shift_jit(psi_star, piston_error.to(u.m).value, wl.to(u.m).value)
    psi_planet_corrupted = phase.shift_jit(psi_planet, piston_error.to(u.m).value, wl.to(u.m).value)
    
    phasors_star = get_telescope_phasors(ctx.interferometer.chip, psi_star_corrupted, wl)
    phasors_planet = get_telescope_phasors(ctx.interferometer.chip, psi_planet_corrupted, wl)
    
    # Plotting Options
    use_log = st.toggle("Log Scale", value=True, help="Use logarithmic scale for amplitude.")

    with st.spinner("Computing phasors..."):
        # We want to show D1 and D2 (indices 1 and 2)
        # Polar plots
        fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(10, 5))
        
        outputs_to_show = [(1, "Dark 1"), (2, "Dark 2")]
        colors = ['C0', 'C1', 'C2', 'C3'] # 4 Telescopes
        
        max_amp = 0
        min_nonzero_amp = 1.0

        # First pass to find scales
        for out_idx, _ in outputs_to_show:
            for t in range(4):
                z_s = phasors_star[t, out_idx]
                z_p = phasors_planet[t, out_idx]
                local_max = max(np.abs(z_s), np.abs(z_p))
                local_min = min(np.abs(z_s), np.abs(z_p)) if min(np.abs(z_s), np.abs(z_p)) > 0 else 1.0
                max_amp = max(max_amp, local_max)
                min_nonzero_amp = min(min_nonzero_amp, local_min)
                
        # Avoid 0 in log scale
        if min_nonzero_amp == 0: min_nonzero_amp = 1e-10
        
        # Define plot limits
        if use_log:
            # Lower limit well below the smallest signal to minimize the "hole" visual effect
            # or exactly at the noise floor if we want to show it.
            # Let's set it to a fraction of the min amplitude to define the "center"
            rmin = min_nonzero_amp * 0.1 
            rmax = max_amp * 1.5
        else:
            rmin = 0
            rmax = max_amp * 1.1

        for ax, (out_idx, label) in zip(axes, outputs_to_show):
            # Star Phasors
            ax.set_title(label)
            
            # Plot Star phasors (lines + star marker)
            for t in range(4):
                z = phasors_star[t, out_idx]
                # Line starts at rmin in log mode to appear coming from center
                r_start = rmin if use_log else 0
                ax.plot([0, np.angle(z)], [r_start, np.abs(z)], color=colors[t], alpha=0.9, linewidth=2)
                ax.scatter(np.angle(z), np.abs(z), color=colors[t], marker='*', s=100, zorder=10)
                
            # Plot Planet phasors
            for t in range(4):
                z = phasors_planet[t, out_idx]
                r_start = rmin if use_log else 0
                ax.plot([0, np.angle(z)], [r_start, np.abs(z)], color=colors[t], linestyle=':', alpha=0.9, linewidth=1)
                ax.scatter(np.angle(z), np.abs(z), color=colors[t], marker='o', s=30)
                
            if use_log:
                ax.set_rscale('log')
                ax.set_rlim(rmin, rmax)
            else:
                ax.set_ylim(0, rmax)

            ax.set_yticklabels([])
        
        st.pyplot(fig)
        st.caption("Vector contributions of the 4 telescopes (1 color = 1 telescope) for the star (★) and the planet (●) in the two dark outputs.")


# --- Distributions ---
st.subheader("Statistical Distributions")

# Generate statistics
n_samples = 1000
# We need to run observations with random piston noise
# Context.observe_monochromatic takes upstream_pistons.
# We can loop calling it with random pistons.

# Faster approach: Generate batch of pistons and use batch processing if available, or simple loop
dist_pistons = np.random.normal(0, ctx.Γ.to(u.nm).value, (n_samples, 4)) * u.nm

# We can use the Chip's batch processing if `get_output_fields` supports batch psi/phi?
# SuperKN `get_output_fields` seems to support batch if phi has batch dim, or psi has batch.
# But here psi depends on piston, so psi is batch.
# Let's try to prepare batch psi.

# Base inputs (no piston)
base_psi = ctx.get_input_fields() # (2, 4) Star, Planet
# We sum them to get total field per telescope (coherent sum? No, incoherent usually for different sources unless they interfere? Star and Planet don't interfere with each other)
# Wait, for intensity distributions, we sum INTENSITIES of Star and Planet outputs.
# So we process Star and Planet separately through the nuller, get interaction with piston, then square modulus, then add.

# Batch Piston Shift
# Shift takes (psi, opd, lambda). opd shape (N, 4). psi can be (4,).
# We broadcast psi to (N, 4) to ensure phase.shift_jit works as expected.

psi_star_batch_in = np.tile(psi_star, (n_samples, 1)) # (N, 4)
psi_planet_batch_in = np.tile(psi_planet, (n_samples, 1)) # (N, 4)

psi_star_batch = phase.shift_jit(psi_star_batch_in, dist_pistons.to(u.m).value, wl.to(u.m).value)
psi_planet_batch = phase.shift_jit(psi_planet_batch_in, dist_pistons.to(u.m).value, wl.to(u.m).value)

# Propagate through Chip
star_outs = ctx.interferometer.chip.get_output_fields(psi_star_batch, wl) # (N, 7)
planet_outs = ctx.interferometer.chip.get_output_fields(psi_planet_batch, wl) # (N, 7)

# Intensities (Detectors)
I_star = np.abs(star_outs)**2
I_planet = np.abs(planet_outs)**2
I_total = I_star + I_planet

# Extract D1, D2 and Kernel
# D1 is index 1, D2 is index 2
D1 = I_total[:, 1]
D2 = I_total[:, 2]

# Kernel calculation: The user said "distribution du kernel formé par la soustraction de ces deux sorties sombres"
# K = D1 - D2 (or similar)
# SuperKN usually defines K1 = D1 - D2.
K = D1 - D2

# Intensity for Star Only
D1_star = I_star[:, 1]
D2_star = I_star[:, 2]
K_star = D1_star - D2_star

# Plotting Histograms
st.markdown("### Distributions")
dist_log = st.toggle("Log Scale (Y-axis)", value=False, help="Use logarithmic scale for histogram counts.")

with st.spinner("Computing distributions..."):
    fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # Helper to handle constant data
    def safe_hist(ax, data, **kwargs):
        dmin, dmax = np.min(data), np.max(data)
        if dmin == dmax:
            # Constant data: force a range roughly centered
            margin = max(abs(dmin) * 0.1, 1e-5)
            kwargs['range'] = (dmin - margin, dmax + margin)
        ax.hist(data, log=dist_log, **kwargs)

    # Plot D1 - Star+Planet
    safe_hist(ax1, D1, bins=30, alpha=0.7, color='C0', density=True, label='Star + Planet')
    # Plot D1 - Star Only
    safe_hist(ax1, D1_star, bins=30, alpha=1.0, color='k', density=True, histtype='step', linestyle='--', label='Star Only')
    
    ax1.set_xlabel("Intensity")
    ax1.set_title(f"Dark 1\nMean: {np.mean(D1):.2e}")
    ax1.legend()

    # Plot D2 - Star+Planet
    safe_hist(ax2, D2, bins=30, alpha=0.7, color='C1', density=True, label='Star + Planet')
    # Plot D2 - Star Only
    safe_hist(ax2, D2_star, bins=30, alpha=1.0, color='k', density=True, histtype='step', linestyle='--', label='Star Only')
    
    ax2.set_xlabel("Intensity")
    ax2.set_title(f"Dark 2\nMean: {np.mean(D2):.2e}")
    ax2.legend()

    # Plot Kernel distribution - Star+Planet
    safe_hist(ax3, K, bins=30, color='purple', alpha=0.7, density=True, label='Star + Planet')
    # Plot Kernel distribution - Star Only
    safe_hist(ax3, K_star, bins=30, color='k', alpha=1.0, density=True, histtype='step', linestyle='--', label='Star Only')
    
    ax3.set_xlabel("Kernel Signal")
    ax3.set_title(f"Kernel (D1 - D2)\nMean: {np.mean(K):.2e}")
    ax3.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax3.legend()

    st.pyplot(fig2)

st.divider()

# --- Transmission Maps ---
st.header("Transmission Maps")
st.caption("Ideal response of the outputs to a source at a given position (no piston noise).")

with st.spinner("Computing transmission maps..."):
    # Grid parameters
    # Use context FOV. grid_1d spans [-FOV/2, +FOV/2]
    fov_param = ctx.interferometer.fov.to(u.mas).value
    fov_radius = fov_param / 2
    n_grid = 100
    grid_1d = np.linspace(-fov_radius, fov_radius, n_grid)
    xx, yy = np.meshgrid(grid_1d, grid_1d) # xx=RA (mas), yy=DEC (mas) -- usually RA is inverted but let's stick to simple
    
    # Flatten
    ra_flat = xx.flatten() * u.mas
    dec_flat = yy.flatten() * u.mas
    
    # Get UV coords (meters)
    # Context calculates projected positions 'p' (N, 3) or (N, 2)?
    # Usually p = [u, v, w].
    # Let's verify shape or assume standard.
    
    p_proj = ctx.p.to(u.m).value # (4, 3) usually
    u_base = p_proj[:, 0] # (4,)
    v_base = p_proj[:, 1] # (4,)
    
    # Compute phases: k * (u*alpha + v*delta)
    # ra_flat: (M,), u_base: (4,)
    # Output: (M, 4)
    # alpha, delta in rad
    alpha_rad = ra_flat.to(u.rad).value[:, np.newaxis]
    delta_rad = dec_flat.to(u.rad).value[:, np.newaxis]
    
    k = 2 * np.pi / wl.to(u.m).value
    
    # Phase delay (meters?) No, phase uses k.
    # OPD = u*alpha + v*delta
    opd = alpha_rad * u_base[np.newaxis, :] + delta_rad * v_base[np.newaxis, :] # (M, 4) meters
    
    phi_grid = k * opd # (M, 4) radians
    psi_grid = np.exp(1j * phi_grid) # (M, 4) complex inputs
    
    # Propagate through Kernel Nuller (No Piston Error)
    # We use batch processing
    # Note: SuperKN handles batch. Since opd/sigma are not batch here (static), we just pass (M, 4) psi.
    
    outs_grid = ctx.interferometer.chip.get_output_fields(psi_grid, wl) # (M, 7)
    
    # Normalize by number of telescopes to get transmission relative to total input flux
    n_tel = 4 # Kernel Nuller is 4T
    intensity_grid = np.abs(outs_grid)**2 / n_tel
    
    I_D1 = intensity_grid[:, 1].reshape(n_grid, n_grid)
    I_D2 = intensity_grid[:, 2].reshape(n_grid, n_grid)
    I_K = (intensity_grid[:, 1] - intensity_grid[:, 2]).reshape(n_grid, n_grid)
    
    # Plotting
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    extent = [grid_1d[0], grid_1d[-1], grid_1d[0], grid_1d[-1]]
    
    # Dark 1
    im1 = axes3[0].imshow(I_D1, origin='lower', extent=extent, cmap='inferno')
    axes3[0].set_title("Dark 1 Map")
    axes3[0].set_xlabel("$\Delta$RA (mas)")
    axes3[0].set_ylabel("$\Delta$Dec (mas)")
    plt.colorbar(im1, ax=axes3[0], fraction=0.046, pad=0.04)
    
    # Dark 2
    im2 = axes3[1].imshow(I_D2, origin='lower', extent=extent, cmap='inferno')
    axes3[1].set_title("Dark 2 Map")
    axes3[1].set_xlabel("$\Delta$RA (mas)")
    plt.colorbar(im2, ax=axes3[1], fraction=0.046, pad=0.04)

    # Kernel
    # Use diverging colormap for Kernel
    vmax = max(abs(np.min(I_K)), abs(np.max(I_K)))
    im3 = axes3[2].imshow(I_K, origin='lower', extent=extent, cmap='bwr', vmin=-vmax, vmax=vmax)
    axes3[2].set_title("Kernel Map (D1 - D2)")
    axes3[2].set_xlabel("$\Delta$RA (mas)")
    plt.colorbar(im3, ax=axes3[2], fraction=0.046, pad=0.04)
    
    # Mark planet position
    for ax in axes3:
        # relative offset of planet
        # p_dRA = sep * sin(pa)
        # p_dDec = sep * cos(pa)
        # Check definitions usually: RA is East-Left? standard calculation:
        # x = -rho * sin(theta) if RA increases to East (left on sky)?
        # Let's use standard math conventions for now: x = rho * sin(theta), y = rho * cos(theta)
        # Assuming North Up, East Left -> +RA is usually -x direction.
        # But here X label is dRA. If +dRA is Right, that's West.
        # Let's stick to Cartesian x=RA, y=Dec for simplicity unless specified.
        
        # Plotting the planet marker
        # Math Convention: PA=0 is East (+X), PA=90 is North (+Y)
        p_x = sep_mas * np.cos(np.deg2rad(pa_deg))
        p_y = sep_mas * np.sin(np.deg2rad(pa_deg))
        
        # Planet
        ax.scatter(p_x, p_y, color='lime', marker='o', s=50, edgecolors='black', label='Planet', zorder=10)
        # Star
        ax.scatter(0, 0, color='gold', marker='*', s=150, edgecolors='black', label='Star', zorder=10)
        
        ax.legend(loc='upper right', fontsize='small', framealpha=0.8)

    plt.tight_layout()
    st.pyplot(fig3)

st.divider()

# References
with st.expander("📚 References", expanded=False):
    st.markdown("""
    *   **Martinache & Ireland (2018)** - *Kernel-nulling for a robust direct interferometric detection of exoplanets*.
    *   **Cvetojevic et al. (2022)** - *3-beam self-calibrated Kernel nulling photonic interferometer*.
    *   **Chingaipe et al. (2023)** - *High-contrast detection of exoplanets with a kernel-nuller at the VLTI*.
    """)
