"""
Streamlit page for Correlation Map analysis.
Visualize the correlation between observed data and theoretical kernel signatures.
"""

import sys
from pathlib import Path
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from copy import deepcopy as copy

# --- Path Setup ---
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
WEB = ROOT / "web"
if str(WEB) not in sys.path:
    sys.path.insert(0, str(WEB))

try:
    from src.analysis.sky_contribution import sky_contribution as sc_module
    from src.phise.modules import coordinates
except ImportError:
    if str(ROOT / 'src') not in sys.path:
        sys.path.append(str(ROOT / 'src'))
    from analysis.sky_contribution import sky_contribution as sc_module
    from phise.modules import coordinates

from phise import Context
from utils.context_widget import context_widget

# --- Mock tqdm ---
def streamlit_tqdm(iterable, desc=None):
    total = len(iterable)
    progress_text = desc if desc else "Processing..."
    progress_bar = st.progress(0, text=progress_text)
    for i, item in enumerate(iterable):
        yield item
        frac = (i + 1) / total
        progress_bar.progress(frac, text=f"{progress_text} ({i+1}/{total})")
    progress_bar.empty()

# --- Page Config ---
st.set_page_config(
    page_title="Correlation Map",
    page_icon="🎯",
    layout="wide",
)

st.title("Correlation Map 🎯")

st.markdown(r"""
The **Correlation Map** represents the likelihood of finding a source at a given position in the sky, based on the similarity between the observed signal and the theoretical signal expected from that position.

It corresponds to a **Matched Filter** approach: we scan the field of view and, for each position $(\alpha, \delta)$, we compute the correlation coefficient between the observed time-series data $d(t)$ and the theoretical modulation model $m(t, \alpha, \delta)$.

$$
C(\alpha, \delta) = \frac{1}{N_k} \sum_{k=1}^{N_k} \text{corr}(d_k(t), m_k(t, \alpha, \delta))
$$

Where:
- $d_k(t)$ is the observed signal for kernel $k$.
- $m_k(t, \alpha, \delta)$ is the theoretical model for kernel $k$ and source position.
- $\text{corr}$ is the Pearson correlation coefficient:

$$
\text{corr}(X, Y) = \frac{\text{cov}(X, Y)}{\sigma_X \sigma_Y} = \frac{\sum_{i=1}^{T} (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^{T} (X_i - \bar{X})^2} \sqrt{\sum_{i=1}^{T} (Y_i - \bar{Y})^2}}
$$

### Interpretation
- **Value $\approx 1$**: Strong match. The observed signal modulation matches the signature of a source at this position.
- **Value $\approx 0$**: No match.
- **Value $< 0$**: Anti-correlation (signal is opposite to model).

This map allows identifying the position of companions by finding the local maxima of correlation.
""")

st.divider()

# --- Configuration ---
st.subheader("Configuration")

presets = {
    "VLTI": Context.get_VLTI(),
    "LIFE": Context.get_LIFE(),
}

def setup_context(c: Context) -> Context:
    """Defaults for Correlation Map analysis."""
    c.monochromatic = True
    c.interferometer.chip.φ = np.zeros(14) * u.nm
    c.interferometer.chip.σ = np.zeros(14) * u.nm
    # Correlation often needs reasonable integration to have a signal, 
    # but here we simulate high SNR usually unless camera noise is added.
    c.Γ = 10 * u.nm
    if c.target.companions:
        c.target.companions[0].c = 1e-2
    c.Δh = 24 * u.hourangle
    return c

# Prepare default context
default_ctx = copy(presets["VLTI"])
default_ctx = setup_context(default_ctx)

ctx = context_widget(
    key_prefix="correl_map",
    presets=presets,
    default_preset="VLTI",
    expanded=True,
    show_advanced=True,
    initial_context=default_ctx,
)

st.divider()

# --- Parameters ---
st.subheader("Simulation Parameters")
col1, col2 = st.columns(2)
with col1:
    resolution = st.slider("Map Resolution (pixels)", 20, 150, 60, step=10)
    samples_per_ha = st.number_input("Samples per Hour Angle", value=60, min_value=10)

with col2:
    st.info("Simulation includes 'Observation' phase (generating data) and 'Correlation' phase (processing map).")
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        run_sim = True
    else:
        run_sim = False

# --- Processing ---
if run_sim:
    st.divider()
    
    try:
        # 1. Observation / Data Generation
        st.subheader("1. 🔭 Observing")
        
        # Determine Hour Angle Range (time steps)
        # Using ctx.h (center) and ctx.dh (span)
        # Logic adapted from Context.get_h_range or explicit linspace
        # Context.get_h_range() uses self.h and self.Δh.
        
        # We need to ensure we simulate enough points. 
        # But wait, sc_module.get_contribution_map iterates strictly over h_range?
        # Let's perform the observation loop manually to get the data.
        
        h_center = ctx.h.to(u.hourangle).value
        dh = ctx.Δh.to(u.hourangle).value
        
        # We assume simulation covers [-dh/2, +dh/2] around h? 
        # Or [h - dh/2, h + dh/2]?
        # Usually centered on h.
        h_start = h_center - dh/2
        h_stop = h_center + dh/2
        
        # Total samples = samples_per_ha * dh (roughly) or just straight number?
        # The UI says "Samples per Hour Angle", implying density.
        n_samples = int(samples_per_ha * max(dh, 1)) # Ensure at least some points
        
        h_vals = np.linspace(h_start, h_stop, n_samples)
        
        # Storage
        observed_kernels = np.zeros((3, n_samples))
        
        # Progress bar for observation
        obs_progress = st.progress(0, text="Observing target...")
        
        # Make a copy of context for iteration
        sim_ctx = copy(ctx)
        
        for i, h_v in enumerate(h_vals):
            sim_ctx.h = h_v * u.hourangle
            # Observe
            # observe() returns intensities.
            # process_outputs() converts to kernels.
            outs = sim_ctx.observe()
            ks = sim_ctx.interferometer.chip.process_outputs(outs)
            observed_kernels[:, i] = ks
            
            obs_progress.progress((i+1)/n_samples, text=f"Observing... ({i+1}/{n_samples})")
            
        obs_progress.empty()
        st.success("Observation Complete!")
        
        # 2. Correlation Mapping
        st.subheader("2. 🎯 Correlating")
        
        # Compute maps
        # We need to scan grid ρ, θ
        # sc_module.get_correlation_map does it but returns only total.
        # We implement the loop here to get all.
        
        # Grid setup
        _, _, theta_map, rho_map = coordinates.get_maps(N=resolution, fov=ctx.interferometer.fov)
        theta_vals = theta_map.to(u.rad).value
        rho_vals = rho_map.to(u.rad).value # In rad? sc_module used get_maps returning rads for theta, and... rho?
        # coordinates.get_maps doc: "rho (separation) ... in fov units?"
        # Let's check sc_module line 47: `rh_map.value / np.max(...)`
        # And line 215: `rho_map.to(u.rad).value` -> So rho_map has units.
        
        correl_maps = np.zeros((3, resolution, resolution))
        total_correl_map = np.zeros((resolution, resolution))
        
        calc_progress = st.progress(0, text="Computing correlation maps...")
        
        # Parallelization is hard here without overhead, simple loop is fine with numba jitted kernels_modulation
        for x in range(resolution):
            calc_progress.progress(x / resolution, text=f"Computing map row {x+1}/{resolution}")
            for y in range(resolution):
                ρ = rho_vals[x, y]
                θ = theta_vals[x, y]
                
                # Theoretical modulation
                # sc_module.kernels_modulation(ctx, h_range, rho, theta)
                # Note: h_range in kernels_modulation is expected in hourangle values?
                # sc_module line 174: h_rad = h * (np.pi / 12) -> Yes, expects hours.
                
                km = sc_module.kernels_modulation(sim_ctx, h_vals, ρ, θ)
                
                sum_c = 0
                for k in range(3):
                    # Correlation
                    # Handling constant signals (std=0)
                     if np.std(km[k]) == 0 or np.std(observed_kernels[k]) == 0:
                        c = 0
                     else:
                        c = np.corrcoef(observed_kernels[k], km[k])[0, 1]
                     
                     correl_maps[k, x, y] = c
                     sum_c += c
                     
                total_correl_map[x, y] = sum_c / 3
                
        calc_progress.progress(1.0, text="Done!")
        calc_progress.empty()
        
        st.success("Analysis Complete!")
        
        # 3. Visualization
        fov_val = ctx.interferometer.fov.to(u.mas).value
        extent = [-fov_val/2, fov_val/2, -fov_val/2, fov_val/2]
        
        # Calculate global limits for consistent colorbar
        all_maps = np.concatenate([correl_maps, total_correl_map[np.newaxis, ...]])
        vmin = np.min(all_maps)
        vmax = np.max(all_maps)
        
        # Individual Kernels
        st.markdown("### Individual Correlations")
        fig1, axs1 = plt.subplots(1, 3, figsize=(18, 5))
        
        for k in range(3):
            im = axs1[k].imshow(correl_maps[k], cmap='seismic', vmin=vmin, vmax=vmax, extent=extent, origin='lower')
            axs1[k].set_title(f'Kernel {k+1} Correlation')
            axs1[k].set_xlabel('$\Delta RA$ [mas]')
            if k == 0:
                axs1[k].set_ylabel('$\Delta Dec$ [mas]')
            plt.colorbar(im, ax=axs1[k], fraction=0.046, pad=0.04)
            
            # Companions
            for companion in ctx.target.companions:
                try:
                    px, py = coordinates.ρθ_to_xy(ρ=companion.ρ, θ=companion.θ, fov=ctx.interferometer.fov)
                    # px, py are normalized [-1, 1]
                    axs1[k].scatter(px * fov_val/2, py * fov_val/2, marker='x', color='lime', s=80, label="Real Position")
                except: pass
                
        st.pyplot(fig1)
        
        # Total
        st.markdown("### Total Correlation")
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 7))
        im2 = ax2.imshow(total_correl_map, cmap='seismic', vmin=vmin, vmax=vmax, extent=extent, origin='lower')
        ax2.set_title('Mean Correlation Map')
        ax2.set_xlabel('$\Delta RA$ [mas]')
        ax2.set_ylabel('$\Delta Dec$ [mas]')
        plt.colorbar(im2, ax=ax2, label='Mean Correlation')
        
        for companion in ctx.target.companions:
            try:
                px, py = coordinates.ρθ_to_xy(ρ=companion.ρ, θ=companion.θ, fov=ctx.interferometer.fov)
                ax2.scatter(px * fov_val/2, py * fov_val/2, marker='x', color='lime', s=100, label="Real Position")
            except: pass
        ax2.legend()
        st.pyplot(fig2)
        
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.expander("Traceback").code(traceback.format_exc())

st.divider()

with st.expander("🤝 Acknowledgements", expanded=False):
    st.markdown("""
    - **Frantz Martinache** for sharing me the idea of this analysis.
    """)

