
"""
Streamlit page for Photonic Architecture visualization.
Visualize the 4x7 component with 14 phase shifters.
"""

import sys
from pathlib import Path
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

# --- Path Setup ---
ROOT = Path(__file__).parent.parent.parent
# Ensure project root is on path so `src` is importable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
# Add web to path for utils
WEB = ROOT / "web"
if str(WEB) not in sys.path:
    sys.path.insert(0, str(WEB))

import importlib
try:
    import phise.classes.archs.superkn as superkn_module
    importlib.reload(superkn_module)
    from phise.classes.archs.superkn import SuperKN
except ImportError:
    # Fallback if regular import works
    from phise.classes.archs.superkn import SuperKN

from utils.context_widget import context_widget

st.set_page_config(
    page_title="Exploded Architecture",
    page_icon="🕸️",
    layout="wide",
)

st.title("Exploded Architecture 🕸️")

st.markdown(r"""
## Component Overview

Our architecture is a **4x7 integrated photonic component** (Super-Kernel-Nuller).  
It allows mixing 4 optical inputs to produce 7 outputs:
*   **1 Bright output**: Constructive interference sum of all inputs.
*   **6 Dark outputs**: Destructive interference outputs used to construct the Kernel Nulls.

The component contains **14 Thermo-Optic Phase Shifters** ($\phi_1$ to $\phi_{14}$) distributed across the circuit.
These shifters are used to:
1.  **Calibrate** the component (compensate for manufacturing errors).
2.  **Modulate** the signal (if used dynamically).
3.  **Route** the light (in general MMI usage).

### Interactive Control
Use the sliders below to inject phase into each of the 14 shifters and observe the impact on the outputs in the polar plots.
""")

scheme_path = ROOT / "docs" / "img" / "scheme.png"
if scheme_path.exists():
    st.image(str(scheme_path), caption="Schematic of the 4x7 Super-Kernel-Nuller component.", use_container_width=True)
else:
    st.warning(f"Image not found at {scheme_path}")

st.divider()

# Layout
st.subheader("Phase Shifters Control")
cols = st.columns([1, 1, 1])

# 14 Shifters
# Group them by layer based on SuperKN architecture
# Layer 1: Inputs (0-3) - 4 shifters
# Layer 2: Middle (4-7) - 4 shifters
# Layer 3: Outputs (8-13) - 6 shifters

phases = np.zeros(14)

with cols[0]:
    st.markdown(r"##### Layer 1 ($\phi_1 \rightarrow \phi_4$)")
    for i in range(4):
        phases[i] = st.slider(
            f"$\phi_{{{i+1}}}$",
            min_value=0.0,
            max_value=360.0,
            value=0.0,
            step=1.0,  # Degree step
            key=f"phi_{i}"
        )

    if st.button("Reset all phases"):
        for i in range(14):
            st.session_state[f"phi_{i}"] = 0.0
        st.rerun()
    use_ref = st.checkbox("Phase Ref (Input 1)", value=True, help="If checked, Input 1 phase is set to 0° (reference).")

with cols[1]:
    st.markdown(r"##### Layer 2 ($\phi_5 \rightarrow \phi_8$)")
    for i in range(4):
        phases[4+i] = st.slider(
            f"$\phi_{{{5+i}}}$",
            min_value=0.0,
            max_value=360.0,
            value=0.0,
            step=1.0,
            key=f"phi_{4+i}"
        )
            
with cols[2]:
    st.markdown(r"##### Layer 3 ($\phi_9 \rightarrow \phi_{14}$)")
    for i in range(6):
        phases[8+i] = st.slider(
            f"$\phi_{{{9+i}}}$",
            min_value=0.0,
            max_value=360.0,
            value=0.0,
            step=1.0,
            key=f"phi_{8+i}"
        )

# Convert deg to rad? SuperKN uses u.Quantity length units (OPD).
# Need to know the wavelength to convert phase angle to OPD.
# Let's set a default lambda = 1.65 um (H band center roughly) or allow user selection.

st.subheader("Output Response (Polar Plots)")

lam = 1.55 * u.um

# Convert phase (deg) to OPD (nm/um)
# Phase = 2*pi * OPD / lambda
# OPD = Phase * lambda / (2*pi)

phases_rad = np.deg2rad(phases)
opds = phases_rad * (lam.to(u.nm).value) / (2 * np.pi) * u.nm

# Create Chip
# We assume ideal manufacturing (sigma=0) to see pure effect of shifters
sigma = np.zeros(14) * u.nm

chip = SuperKN(
    φ=opds,
    σ=sigma,
    λ0=lam, # Match design wavelength to current for simplicity
    name="Interactive SuperKN"
)

# Plot
# plot_output_phase returns bytes if plot=False
try:
    # We need to capture the plot. plot_output_phase handles subplots creation.
    # It displays 4 overlapping curves (one for each input).
    
    img_bytes = chip.plot_output_phase(λ=lam, plot=False, n_cols=4, ref_input1=use_ref)

    st.image(img_bytes, width=None, caption="Polar plots of the 7 outputs. Each color represents the response to one of the 4 inputs.", use_container_width=True)

except Exception as e:
    st.error(f"Error generating plot: {e}")
    st.code(traceback.format_exc())

st.divider()
st.markdown("""
**Interpretation**:
*   The **angle** represents the phase of the output field.
*   The **radius** represents the amplitude (squared radius = intensity).
*   Each **color** corresponds to one of the 4 inputs being lit individually ($I_1$=Yellow, $I_2$=Green, $I_3$=Red, $I_4$=Blue).
*   Use the sliders to see how the phases rotate or interfere (amplitude change) at the outputs.
""")
