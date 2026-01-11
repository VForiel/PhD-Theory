"""
Streamlit page for On-Sky Contribution analysis.
Visualize the contribution zones of kernels on the sky.
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
# Ensure project root is on path
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Add web to path for utils
WEB = ROOT / "web"
if str(WEB) not in sys.path:
    sys.path.insert(0, str(WEB))

try:
    from src.analysis.sky_contribution import sky_contribution as sc_module
except ImportError:
    if str(ROOT / 'src') not in sys.path:
        sys.path.append(str(ROOT / 'src'))
    from analysis.sky_contribution import sky_contribution as sc_module

from phise import Context
from utils.context_widget import context_widget

# --- Mock tqdm for Streamlit Progress ---
def streamlit_tqdm(iterable, desc=None):
    """
    Mock tqdm that updates a streamlit progress bar.
    """
    total = len(iterable)
    progress_text = desc if desc else "Processing..."
    progress_bar = st.progress(0, text=progress_text)
    
    for i, item in enumerate(iterable):
        yield item
        # Update progress
        frac = (i + 1) / total
        progress_bar.progress(frac, text=f"{progress_text} ({i+1}/{total})")
    
    progress_bar.empty()

# Monkeypatch tqdm in the module
sc_module.tqdm = streamlit_tqdm


# --- Page Configuration ---
st.set_page_config(
    page_title="On-Sky Contribution",
    page_icon="🌌",
    layout="wide",
)

st.title("On-Sky Contribution 🌌")

st.markdown(r"""
From the obtained data, it is possible to build a 2D distribution of the perceived sky contribution zones. This distribution provides insights into the possible locations of objects, enabling accurate initial estimations to fit the data points obtained based on the parallactic angle.

This method involves stacking the transmission maps rotated by the baseline rotation and weighting each map by the corresponding data obtained for that baseline rotation.

The base idea was already explored as "image reconstruction" technic using classical nulling interferometry $^1$. However, the method here is based on Kernel-Nulls which makes it more complex but less sensitive to phase aberations and by considerig the different Kernels, we can reduce the degeneracy of the solutions.

Considering:
- $T_{n}$ represents the n-th kernel's normalized transmission map.
- $d_{n,\beta}$ denotes the data point obtained for kernel $n$ with baseline rotation $\beta$.
- $\theta$ is the parallactic angle.
- $\rho$ is the angular separation.

$$
r_n(\rho, \theta) = \sum_a T_{n,h}(\rho,\theta) d_{n,h}
$$


As the kernel outputs are antisymetric, we can filter the negative contributions:
$$
r'_n = \frac{1}{2}\max(r_n, 0)
$$


Finally, we can compute the product over all the kernels to get the final contribution zones:
$$
C(\rho, \theta) = \prod_n r'_n(\rho, \theta)
$$
""")

st.divider()

# --- Configuration ---
st.subheader("Configuration")

# Context Widget
presets = {
    "VLTI": Context.get_VLTI(),
    "LIFE": Context.get_LIFE(),
}

def setup_context(c: Context) -> Context:
    """
    Apply default settings for On-Sky Contribution analysis.
    - Monochromatic: True (faster)
    - Zero phase errors (ideal)
    """
    c.monochromatic = True
    c.interferometer.chip.φ = np.zeros(14) * u.nm
    c.interferometer.chip.σ = np.zeros(14) * u.nm
    c.Γ = 10 * u.nm
    if c.target.companions:
        c.target.companions[0].c = 1e-2
    c.Δh = 24 * u.hourangle
    return c

# Prepare default context
default_ctx = copy(presets["VLTI"])
default_ctx = setup_context(default_ctx)

ctx = context_widget(
    key_prefix="os_contrib",
    presets=presets,
    default_preset="VLTI",
    expanded=True,
    show_advanced=True,
    initial_context=default_ctx,
)

st.divider()

# --- Simulation Parameters ---
st.subheader("Simulation Parameters")

col1, col2 = st.columns(2)

with col1:
    resolution = st.slider("Map Resolution (pixels)", 20, 200, 100, step=10, help="Higher resolution is slower but more detailed.")
    samples_per_ha = st.number_input("Samples per Hour Angle", value=60, min_value=10, help="Number of time steps for the simulation.")

with col2:
    st.info("The simulation uses the parameters defined in the Context Configuration above.")
    if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
        run_simulation = True
    else:
        run_simulation = False

# --- Simulation Execution ---
if run_simulation:
    st.divider()
    st.subheader("Results")
    
    st.info("Running simulation... This may take a moment depending on resolution and time span.")
    
    try:
        # We need to capture the result directly to plot it ourselves
        # sc_module.get_contribution_map handles the heavy lifting
        
        # Ensure context has the correct integration properties from the widget
        # The widget updates 'ctx' in session state, so 'ctx' variable here is up to date.
        
        results = sc_module.get_contribution_map(
            ctx=ctx,
            resolution=resolution,
            n=int(samples_per_ha),
            map_func=np.median
        )
        
        images = results["images"]
        stack = results["stack"]
        ref_ctx = results["ctx"]
        
        st.success("Simulation Complete!")
        
        # --- Visualization ---
        
        # Determine extent
        fov = ref_ctx.interferometer.fov.to(u.mas)
        extent = [-fov.value / 2, fov.value / 2, -fov.value / 2, fov.value / 2]
        max_im = np.max(images)
        
        # 1. Individual Kernels
        st.markdown("### Individual Kernel Maps")
        st.markdown("Maps showing the contribution of the sky to each individual Kernel output.")
        
        fig1, axs1 = plt.subplots(1, 3, figsize=(18, 5))
        for k in range(3):
            img = images[6+k] # Kernels are at indices 6, 7, 8 in raw_images returned by get_contribution_map
            # Wait, get_contribution_map returns `raw_images` which has 9 layers: 6 darks + 3 kernels?
            # Let's verify sc_module code. 
            # line 45: raw_images = np.zeros((9, resolution, resolution)) # 6 Darks + 3 Kernels
            # line 84: raw_images[6+k] += ...
            # Yes.
            
            img_disp = img.copy()
            #img_disp[img_disp < 0] = 0 # Optional clamping for display
            
            im = axs1[k].imshow(img_disp, cmap='hot', vmax=max_im, extent=extent, origin='lower')
            axs1[k].set_title(f'Kernel {k + 1}')
            axs1[k].set_xlabel('$\Delta RA$ [mas]')
            if k == 0:
                axs1[k].set_ylabel('$\Delta Dec$ [mas]')
            plt.colorbar(im, ax=axs1[k], fraction=0.046, pad=0.04)
            
            # Plot companions
            for companion in ref_ctx.target.companions:
                try:
                    from phise.modules import coordinates
                    (planet_x, planet_y) = coordinates.ρθ_to_xy(ρ=companion.ρ, θ=companion.θ, fov=fov)
                    # Coordinates are normalized [-1, 1], map to extent
                    axs1[k].scatter(planet_x * fov.value / 2, planet_y * fov.value / 2, color='tab:blue', edgecolors='white', s=50, label='Companion')
                except Exception:
                    pass
        
        st.pyplot(fig1)

        # 2. Combined Contribution
        st.markdown("### Combined Contribution Zone")
        st.markdown("Geometric mean of all outputs, highlighting the global sensitivity.")

        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 7))
        im2 = ax2.imshow(stack, cmap='hot', extent=extent, origin='lower')
        ax2.set_title('Global Contribution Map')
        ax2.set_xlabel('$\Delta RA$ [mas]')
        ax2.set_ylabel('$\Delta Dec$ [mas]')
        plt.colorbar(im2, ax=ax2, label='Contribution')
        
        # Plot companions
        for companion in ref_ctx.target.companions:
            try:
                from phise.modules import coordinates
                (planet_x, planet_y) = coordinates.ρθ_to_xy(ρ=companion.ρ, θ=companion.θ, fov=fov)
                ax2.scatter(planet_x * fov.value / 2, planet_y * fov.value / 2, color='tab:blue', edgecolors='white', s=80, label='Companion')
            except Exception:
                pass
        ax2.legend()
        
        st.pyplot(fig2)
        
    except Exception as e:
        st.error(f"An error occurred during simulation: {e}")
        import traceback
        st.expander("Traceback").code(traceback.format_exc())

st.divider()

with st.expander("📚 References", expanded=False):
    st.markdown("""
    1. Angel, J. R. P., et N. J. Woolf. "An Imaging Nulling Interferometer to Study Extrasolar Planets". *The Astrophysical Journal* 475, no 1 (1997): 373‑79. https://doi.org/10.1086/303529.
    """)

