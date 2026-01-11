
"""
Wavelength Scan Analysis Page.
"""

import sys
from pathlib import Path
import streamlit as st
import astropy.units as u
import numpy as np

# --- Path Setup ---
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
WEB = ROOT / "web"
if str(WEB) not in sys.path:
    sys.path.insert(0, str(WEB))

from phise import Context
from phise.classes.archs.superkn import SuperKN
from utils.context_widget import context_widget
import importlib
import src.analysis.wavelength_scan.wavelength_scan as wavelength_scan
importlib.reload(wavelength_scan)

st.set_page_config(
    page_title="Spectral Scan",
    page_icon="🌈",
    layout="wide",
)

st.title("Spectral Scan Analysis 🌈")

st.markdown(r"""
## Chromaticity & Dynamic Calibration

The photonic component (SuperKN) is typically designed and optimized for a specific central wavelength $\lambda_0$ (e.g., 1.65 µm).
However, manufacturing errors ($\sigma$) and the intrinsic behavior of components (MMIs, directional couplers) are chromatic.

When observing over a broad band or at a different wavelength $\lambda \neq \lambda_0$, the calibration performed at $\lambda_0$ may no longer be valid, leading to light leakage (poorer Null Depth).

This analysis demonstrates two scenarios:
1.  **Static Calibration**: The Chip is calibrated ONCE at $\lambda_0$. We apply these settings and measure the performance across the band.
    *   *Expected behavior*: Performance degrades as $|\lambda - \lambda_0|$ increases (Gray curve).
2.  **Dynamic Calibration (Wavelength Scan)**: The Chip is re-calibrated for **each** specific wavelength $\lambda$.
    *   *Expected behavior*: We recover optimal performance (deep null) across the entire band (Blue curve).

### Application: Spectral Features Detection
By scanning the wavelength and optimizing the null at each step ("scanning the null"), we can theoretically maintain high-contrast extinction of the starlight across the spectrum.
Any residual light or specific features observed in the dynamic mode could then be attributed to the off-axis planet (or uncorrected residues), allowing for **spectral characterization** even if the global band is dominated by chromatic leakage in static mode.
""")

st.success("""
##### ✨ My Contribution
Experimental verification of the Spectral Scan intuition.
While the original idea of "scanning the null" came from Frantz Martinache, my contribution was to **implement the complete simulation framework** to verify this intuition. This work quantifies the gain, validates the feasibility, and prepares the ground for the upcoming experimental verification on the bench.
""")


st.divider()

# Context configuration
presets = {
    "VLTI": Context.get_VLTI(),
    "LIFE": Context.get_LIFE(),
}

ctx = context_widget(
    key_prefix="wav_scan",
    presets=presets,
    default_preset="VLTI",
    expanded=False
)

# Simulation Parameters
st.subheader("Simulation Parameters")
col_sim = st.columns(3)

with col_sim[0]:
    scan_range_val = st.number_input(
        r"Scan Window $\Delta\lambda_{scan}$ (µm)",
        value=1.0,
        min_value=0.01,
        max_value=2.0,
        step=0.01,
        help="Total width of the spectral range to scan and plot."
    )
    scan_range = scan_range_val * u.um

    obs_bw_val = st.number_input(
        r"Obs. Bandwidth $\Delta\lambda_{obs}$ (µm)",
        value=0.1,
        min_value=0.0,
        max_value=0.5,
        step=0.001,
        format="%.3f",
        help="Bandwidth used for each observation point (after calibration)."
    )
    obs_bw = obs_bw_val * u.um

    n_points = st.number_input(
        "Number of points",
        value=11,
        min_value=3,
        max_value=51,
        step=2,
        help="Odd number preferred to include center."
    )
    
    # Validation
    resolution = scan_range_val / (n_points - 1)
    if obs_bw_val > resolution:
        st.warning(f"⚠️ $\Delta\lambda_{{obs}}$ ({obs_bw_val}) > Step size ({resolution:.3f}). Points significantly overlap.")


with col_sim[1]:
    algo = st.selectbox(
        "Calibration Algorithm",
        options=["Obstruction", "Genetic"],
        help="Method used to re-calibrate the chip at each wavelength."
    )
    
    algo_params = {}
    if algo == "Obstruction":
        n_samples = st.number_input("Samples (Obstruction)", value=1000, step=100, min_value=100)
        algo_params["n_samples"] = int(n_samples)
    else:
        beta = st.slider("Beta (Genetic)", min_value=0.5, max_value=0.999, value=0.961, step=0.001)
        algo_params["beta"] = beta

with col_sim[2]:
    
    st.info("""
    **Calibration is always Monochromatic** to simulate ideal correction.
    **Observation** uses $\Delta\lambda_{obs}$ which can include chromatic smearing if > 0.
    """)

if st.button("Run Spectral Scan"):
    
    # Progress UI
    sys_msg = st.empty()
    prog_bar = st.progress(0.0)
    
    def update_ui(p, msg):
        prog_bar.progress(p)
        sys_msg.text(msg)

    with st.spinner("Running..."):
        try:
            import importlib
            import src.analysis.wavelength_scan.wavelength_scan as wavelength_scan
            importlib.reload(wavelength_scan)

            # Context
            # We don't need to force monochromatic on the base ctx, 
            # the runner creates separate contexts for cal (mono) and obs (poly/mono).
            
            # Run the analysis
            img_bytes = wavelength_scan.run(
                ctx=ctx, # Pass base context
                scan_range=scan_range,
                obs_bandwidth=obs_bw,
                n=int(n_points),
                figsize=(8, 6),
                return_image=True,
                algo=algo,
                algo_params=algo_params,
                progress_callback=update_ui
            )
            
            # Clear progress UI after run
            sys_msg.empty()
            prog_bar.empty()
            
            st.image(img_bytes, caption="Null Depth vs Wavelength: Static vs Dynamic Calibration.", use_container_width=True)
            
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            import traceback
            st.code(traceback.format_exc())

st.divider()
st.markdown(r"""
**Graph Interpretation:**
*   **Gray Curve ($\lambda_{cal} = \lambda_0$)**: Shows the "V-shape" degradation of the null depth when we move away from the calibration wavelength. This limits the usable bandwidth for high-contrast imaging.
*   **Blue Curve ($\lambda_{cal} = \lambda$)**: Shows the achievable limit if we could effectively calibrate the chip at the specific observation wavelength (e.g., using a tunable laser source or extracting the solution from a spectral fit).
""")

with st.expander("🤝 Acknowledgements"):
    st.markdown("""
    *   **Frantz Martinache** for the original intuition of the spectral scan.
    """)
