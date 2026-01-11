"""
Streamlit page to visualize the projected positions of telescopes.

This page reproduces the functionality from cell 8 of the Thesis notebook
and provides an interactive interface to explore how telescope positions
change with the observatory latitude and the target star declination.
"""

from pathlib import Path
import sys

import streamlit as st
import numpy as np
import astropy.units as u
from copy import deepcopy as copy

# --- Path Setup ---
ROOT = Path(__file__).parent.parent.parent
# Add src to path for phise
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Add web to path for utils
WEB = ROOT / "web"
if str(WEB) not in sys.path:
    sys.path.insert(0, str(WEB))

from phise import Context
from utils.context_widget import context_widget

st.set_page_config(
    page_title="Projected Telescope Positions",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Projected Telescope Positions 🔭")

st.markdown(r"""
## Overview

The projected geometry of telescope positions is fundamental in interferometry.
It determines the baseline lengths and therefore the optical phase shifts observed.

For each observation, the projected positions are calculated using the following formula:

$$
\begin{pmatrix}
u \\
v
\end{pmatrix} =
\begin{pmatrix}
- \sin(l) \sin(h) & \cos(h)\\
\sin(l) \cos(h) \sin(\delta) + \cos(l) \cos(\delta) & \sin(h) \sin(\delta)
\end{pmatrix}
\begin{pmatrix}
B_\text{north} \\
B_\text{east}
\end{pmatrix}
$$

where:
- **l**: observatory latitude
- **h**: hour angle
- **δ**: target declination

For a space-based interferometer, the collectors are already in the plane 
perpendicular to the line of sight. Such an interferometer can be modeled by 
placing it at the north pole (i.e., $l=90^\circ$) and pointing vertically at a star
(i.e., $\delta=90^\circ$).
""")

st.divider()

# =======================
# Simulation Layout (Parameters LEFT | Visualization RIGHT)
# =======================

st.markdown("""
## Projected Positions
""")

presets = {
    "VLTI": Context.get_VLTI(),
    "LIFE": Context.get_LIFE(),
}
base_ctx = context_widget(
    key_prefix="proj_tels",
    presets=presets,
    default_preset="VLTI",
    expanded=False,  # COLLAPSED by default
    show_advanced=True,
)

left_col, right_col = st.columns([1, 1])

# LEFT COLUMN: Simulation-Specific Parameters
with left_col:
    st.caption("**Projection settings**")

    # Hour angle
    h_override = st.slider(
        "Hour angle h (hours)",
        min_value=-12.0,
        max_value=12.0,
        value=float(base_ctx.h.to(u.hourangle).value),
        step=0.1,
        help="Hour angle at the observation start"
    )

    # Observation span
    dh_override = st.slider(
        "Observation span Δh (hours)",
        min_value=0.1,
        max_value=24.0,
        value=float(base_ctx.Δh.to(u.hourangle).value),
        step=0.1,
        help="Duration of observation (affects rotation coverage)"
    )
    
    # Observatory latitude
    lat_override = st.slider(
        "Observatory latitude l (°)",
        min_value=-90.0,
        max_value=90.0,
        value=float(base_ctx.interferometer.l.to(u.deg).value),
        step=1.0,
        help="Observatory latitude (90° = North Pole = space-based)"
    )
    
    # Target declination
    dec_override = st.slider(
        "Target declination δ (°)",
        min_value=-90.0,
        max_value=90.0,
        value=float(base_ctx.target.δ.to(u.deg).value),
        step=1.0,
        help="Target declination (90° = celestial pole)"
    )
    
    # Telescope array info
    st.caption("**Telescope Array**")

    import pandas as pd
    tel_data = []
    for i, tel in enumerate(base_ctx.interferometer.telescopes):
        tel_data.append({
            "Tel": tel.name,
            "x (m)": f"{tel.r.to(u.m).value[0]:.1f}",
            "y (m)": f"{tel.r.to(u.m).value[1]:.1f}",
        })
    st.dataframe(pd.DataFrame(tel_data), hide_index=True, width="stretch")
    st.info(f"Defined in base context")
    
    st.divider()

# Apply overrides to create working context
ctx = copy(base_ctx)
ctx.Δh = dh_override * u.hourangle
ctx.interferometer.l = lat_override * u.deg
ctx.target.δ = dec_override * u.deg
ctx.h = h_override * u.hourangle

# RIGHT COLUMN: Visualization
with right_col:
        
    # Display controls
    st.caption("**Plot Settings**")
    n_baselines = st.slider(
        "Baseline pairs to show",
        min_value=1,
        max_value=20,
        value=10,
    )
    
    # Generate and display figure
    try:
        from matplotlib.figure import Figure
        import matplotlib.pyplot as plt

        img_bytes = ctx.plot_projected_positions(N=n_baselines, return_image=True)

        st.image(img_bytes, width="stretch", caption="Projected telescope positions during observation")

    except Exception as e:
        st.error(f"❌ Error generating figure: {str(e)}")
        st.info("Ensure the PHISE context is configured correctly.")

st.divider()

# Information sections
with st.expander("ℹ️ Technical information", expanded=False):
    st.subheader("About projected positions")

    st.markdown("""
    ### Physical meaning

    The **(u, v) coordinates** represent the components of the baseline vector in the plane
    of the sky perpendicular to the line of sight. They are expressed in wavelength units and
    determine the optical phase shift introduced by each telescope pair.

    ### Influential parameters

    - **Latitude (l)**: The observatory latitude affects the projection of baselines. For an
      observatory at the north pole (l=90°), baselines are already projected.
      
    - **Declination (δ)**: The target declination also modifies the projection. A declination
      of 90° corresponds to a target at the celestial pole.
      
    - **Hour angle (h)**: Represents the apparent motion of the target over time. At h=0, the
      target crosses the local meridian.

    ### Applications

    These projected positions are essential to:
    - Compute optical path differences (OPD)
    - Predict interference fringes
    - Optimize observation configurations
    - Analyze the interferometer sensitivity to spatial frequencies
    """)

with st.expander("📚 References", expanded=False):
    st.markdown("""
    1. **Chingaipe, P.M. et al., 2023**. High-contrast detection of exoplanets with a kernel-nuller at the VLTI. 
       *A&A* 676, A43. https://doi.org/10.1051/0004-6361/202346118
    
    2. **Ségransan, D., 2007**. Observability and UV coverage. 
       *New Astronomy Reviews* 51, 597–603. https://doi.org/10.1016/j.newar.2007.06.005
    """)
