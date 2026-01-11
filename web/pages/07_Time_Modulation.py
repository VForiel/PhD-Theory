import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from phise import Context, Companion
from copy import deepcopy
import pandas as pd
import sys
from pathlib import Path

# --- Path Setup ---
ROOT = Path(__file__).parent.parent.parent
# Add src to path for phise if needed (though typically installed or in path)
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Add web to path for utils
sys.path.insert(0, str(ROOT / "web"))
from utils.context_widget import context_widget

st.set_page_config(page_title="Time Modulation", page_icon="⏱️", layout="wide")

st.title("Time Modulation ⏱️")

st.markdown(r"""
This page illustrates the principle of **time modulation** of the planetary signal in a nulling interferometer.

### The Time Domain Problem

The goal is to detect an exoplanet by observing how its signal varies over time.
As the system rotates (or the projected baseline rotates), the planet crosses different zones of the transmission map (interference fringes).

However, direct analysis of this **time series** is complex for several reasons:
1.  **Signal Mixing**: If multiple planets are present, their signals sum up and mix over time.
2.  **Sensitivity to Initial Conditions**: Fitting a theoretical model to the light curve requires *a priori* knowledge of the approximate position ($\rho, \theta$) and contrast of the planet. Without this information, optimization is difficult and risks falling into local minima.
3.  **Noise**: The signal is often buried in noise (photon, instrumental), making the modulation shape difficult to discern by eye.

The animation below shows what happens "ideally" (without noise) to understand the geometry of the problem.
""")

st.divider()

# --- Context Configuration ---
st.subheader("Configuration")

presets = {
    "VLTI": Context.get_VLTI(),
    "LIFE": Context.get_LIFE(),
}

# The context widget handles instrument/target/wavelength basics
base_ctx = context_widget(
    key_prefix="time_mod",
    presets=presets,
    default_preset="VLTI",
    expanded=False, 
    show_advanced=True
)

# Force ideal component parameters (φ=0, σ=0) for this simulation
base_ctx.interferometer.chip.φ = np.zeros(14) * u.nm
base_ctx.interferometer.chip.σ = np.zeros(14) * u.nm
st.info("ℹ️ **Simulation Note:** The photonic component is forced to be **ideal** (no phase errors, no manufacturing defects) for this demonstration.")

# Allow overriding companions locally for the demo logic if needed
col_params1, col_params2 = st.columns(2)

with col_params1:
    st.markdown("**Planet Parameters (Simulation)**")
    rho_mas = st.slider("Separation (mas)", 0.1, 10.0, 2.0, 0.1, key="main_rho")
    theta_deg = st.slider("Polar Angle (0°=East)", 0.0, 360.0, 0.0, 1.0, key="main_theta")
    contrast_log = st.slider("Log10 Contrast", -10.0, -1.0, -4.0, 0.1, key="main_contrast")

with col_params2:
    st.markdown("**Observation Parameters**")
    h_val = st.slider("Hour Angle (h) - Instantaneous", -6.0, 6.0, 0.0, 0.1, help="Simulates Earth's rotation.")
    gamma_nm = st.slider("Atmospheric Piston RMS (nm)", 0.0, 100.0, 0.0, 1.0, key="gamma_val", help="Piston noise (Gamma)")

# 1. Prepare working context

# This is the "User Configured Context" (possibly polychromatic, non-ideal camera, etc)
ctx_obs = deepcopy(base_ctx)

# 2. Context Override (Defaults)
ctx_obs.Δh = 24 * u.hourangle
ctx_obs.Γ = gamma_nm * u.nm

# 3. Context Override (Planet)
c = 10**contrast_log
θ = theta_deg * u.deg # Math Convention: 0=East, 90=North
ρ = rho_mas * u.mas
ctx_obs.target.companions = [Companion(c=c, ρ=ρ, θ=θ, name="P1 Demo")]

# 4. Context Override (Time for Map)
ctx_obs.h = h_val * u.hourangle

# --- Compute Maps ---
# Map should represent the *Observed* state or Ideal? 
# Usually maps are shown "ideal" geometry on sky.
# Let's keep using the user context but maybe it's fine.
N_pix = 100
raw_maps, processed_maps = ctx_obs.get_transmission_maps(N=N_pix)

fov = ctx_obs.interferometer.fov.to(u.mas).value
extent = [-fov/2, fov/2, -fov/2, fov/2]

# Planet Position
comp = ctx_obs.target.companions[0]
radius = comp.ρ.to(u.mas).value
angle = comp.θ.to(u.rad).value
x_p = radius * np.cos(angle)
y_p = radius * np.sin(angle)


# --- Compute Time Series ---

def compute_time_series(ctx_user):
    # 1. Ideal Series (Continuous Curve)
    # Force parameters: Monochromatic, Ideal Camera, Gamma=0
    ctx_ideal = deepcopy(ctx_user)
    ctx_ideal.monochromatic = True
    ctx_ideal.Γ = 0 * u.nm
    ctx_ideal.interferometer.camera.ideal = True
    
    # 2. Observed Series (Data Points)
    # Use user context as is (respecting their choices of band, noise parameters implied by camera, etc.)
    # However, for "simulation" usually we want to see the effect of these choices.
    
    # Define range
    dh_val = ctx_ideal.Δh.to(u.hourangle).value if ctx_ideal.Δh.value != 0 else 24
    h_start = -dh_val/2
    h_end = dh_val/2
    
    # High resolution for ideal curve
    n_pts_ideal = 200
    h_ideal = np.linspace(h_start, h_end, n_pts_ideal)
    
    # Lower resolution for "observations" (e.g. one every X min)
    # Let's say 20 points for visualization of "data"
    n_pts_obs = 30
    h_obs = np.linspace(h_start, h_end, n_pts_obs)
    
    def get_series(c, h_vals):
        k_list = []
        c_copy = deepcopy(c)
        for h in h_vals:
            c_copy.h = h * u.hourangle
            raw = c_copy.observe()
            k = c_copy.interferometer.chip.process_outputs(raw)
            k_list.append(k)
        return np.array(k_list)

    k_ideal = get_series(ctx_ideal, h_ideal)
    k_obs = get_series(ctx_user, h_obs)
        
    return h_ideal, k_ideal, h_obs, k_obs

h_ideal, k_ideal, h_obs, k_obs = compute_time_series(ctx_obs)

# Find current point on IDEAL curve for the "cursor"
idx_h = (np.abs(h_ideal - h_val)).argmin()
if idx_h >= len(h_ideal): idx_h = len(h_ideal) - 1
current_val = k_ideal[idx_h]


# --- Visualization Layout ---

st.divider()

col_vis_left, col_vis_right = st.columns([1, 1])

with col_vis_left:
    st.subheader("Transmission Map")
    
    # Kernel selection
    map_idx = st.selectbox("Kernel / Output", range(len(processed_maps)), format_func=lambda x: f"Kernel {x+1}", key="map_select")
    
    cur_map = processed_maps[map_idx]
    
    fig_map, ax_map = plt.subplots(figsize=(5.5, 5))
    im = ax_map.imshow(cur_map, extent=extent, origin='lower', cmap='bwr')
    plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    
    # Star & Planet
    ax_map.scatter(0, 0, marker='*', s=200, color='yellow', edgecolors='black', label="Star", zorder=10)
    ax_map.plot(x_p, y_p, 'o', color='lime', markersize=10, label="Planet", markeredgecolor='k', zorder=11)
    
    ax_map.set_xlabel(r"$\Delta\alpha$ (mas)")
    ax_map.set_ylabel(r"$\Delta\delta$ (mas)")
    ax_map.grid(True, linestyle=':', alpha=0.5)
    ax_map.legend(loc='upper right')
    st.pyplot(fig_map)
    st.write("_The map rotates/evolves because the projected baseline changes._")

with col_vis_right:
    st.subheader("Observed Signal (Time Series)")
    
    fig_ts, ax_ts = plt.subplots(figsize=(6, 4))
    
    # 1. Plot Ideal Curve (Reference)
    ax_ts.plot(h_ideal, k_ideal[:, map_idx], label=f"Ideal Model (Mono, No Noise)", color='C0', alpha=0.6, linewidth=2, linestyle='-')
    
    # 2. Plot Simulated Observations (Scatter)
    ax_ts.scatter(h_obs, k_obs[:, map_idx], label="Simulated Data", color='k', s=20, alpha=0.8, zorder=3)
    
    # Indicator for current time
    ax_ts.axvline(h_val, color='r', linestyle='--', alpha=0.8)
    
    # Check if current point is within range for plotting
    if h_ideal[0] <= h_val <= h_ideal[-1]:
       ax_ts.plot(h_ideal[idx_h], current_val[map_idx], 'ro', markersize=8, zorder=5)
    
    ax_ts.set_xlabel("Hour Angle (h)")
    ax_ts.set_ylabel("Kernel Intensity")
    
    ax_ts.set_xlim(h_ideal[0], h_ideal[-1])
    
    ax_ts.grid(True, alpha=0.3)
    ax_ts.legend()
    
    st.pyplot(fig_ts)
    
    st.info("""
    *   **Blue line**: Theoretical signal (Ideal case: Monochromatic, $\Gamma=0$).
    *   **Black dots**: Simulated measurements based on your *current context* settings (Spectral Bandwidth, NullDepth, Detector Noise...).
    """)

st.divider()

# --- References ---
with st.expander("📚 References", expanded=False):
    st.markdown("""
    1. **Chingaipe, P.M. et al., 2023**. High-contrast detection of exoplanets with a kernel-nuller at the VLTI. *A&A* 676, A43.
    2. **Martinache, F. & Ireland, M.J., 2018**. Kernel-nulling for a robust direct interferometric detection of exoplanets. *A&A*.
    3. **Laugier, R. et al., 2020**. Kernel nullers for an arbitrary number of apertures. *A&A*.
    """)
