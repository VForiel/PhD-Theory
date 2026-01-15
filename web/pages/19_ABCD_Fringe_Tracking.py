
import streamlit as st
import os

st.set_page_config(page_title="ABCD Fringe Tracking", layout="wide")

st.title("ABCD Fringe Tracking with Active MMI")

# Helper for assets
def show_asset(filename, caption):
    candidates = [
        os.path.join("web", "assets", "abcd_fringe_tracking", filename),
        os.path.join("assets", "abcd_fringe_tracking", filename),
        os.path.join("abcd_fringe_tracking", filename)
    ]
    for path in candidates:
        if os.path.exists(path):
            st.image(path, caption=caption)
            return
    st.info(f"Asset not found: {filename}")

# --- Section 1: The Problem (Atmosphere) ---
st.header("1. The Problem: Atmospheric Turbulence")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    When observing astronomical objects from Earth, the atmosphere introduces random phase delays (piston errors) between the different sub-apertures of an interferometer.
    
    These errors fluctuate rapidly (on millisecond timescales), causing the interference fringes to move constantly. This creates a "blurring" effect that destroys the spatial information and prevents deep nulling (essential for exoplanet detection).
    
    The animation on the right shows a simulation of this turbulence:
    - **Colored Map**: The atmospheric phase screen moving over the telescope array.
    - **Right Plot**: The resulting phase delays for each telescope over time.
    
    Notice how the delays drift significantly, often exceeding several wavelengths ($\lambda$).
    """)

with col2:
    show_asset("atmosphere.gif", "Simulation of Atmospheric Turbulence moving over the telescope array")

st.divider()

# --- Section 2: The Solution (Active Component) ---
st.header("2. The Solution: Active Photonic Component")

st.markdown("""
To correct these errors, we use a **4x4 Multi-Mode Interferometer (MMI)** equipped with integrated thermo-optic phase shifters. This component acts as both a beam combiner and a fringe tracker.

### Behavior with 2 Inputs
When we inject light into only **2 of the 4 inputs** (e.g., Input 1 and Input 2), the 4x4 MMI behaves in a very specific way. Due to the internal modal interference, the light is distributed among the 4 outputs with specific phase relationships.

Ideally, the intensities $I_A, I_B, I_C, I_D$ at the four outputs are related to the phase difference $\phi$ between the two inputs by:

$$
\\begin{aligned}
I_A &\\propto 1 + \\cos(\\phi) \\\\
I_B &\\propto 1 + \\sin(\\phi) \\\\
I_C &\\propto 1 - \\cos(\\phi) \\\\
I_D &\\propto 1 - \\sin(\\phi)
\\end{aligned}
$$

### The ABCD Method
This specific quadrature relationship allows us to apply the **ABCD algorithm**. By combining these intensities, we can directly estimate the phase error $\phi$:

$$
\\tan(\\phi) = \\frac{I_B - I_D}{I_A - I_C}
$$

This provides a direct, unambiguous measurement of the phase error within a $[-\pi, \pi]$ range. We can then use this error signal to drive a feedback loop, actuating the integrated phase shifters to compensate for the atmospheric delay in real-time.
""")

st.divider()

# --- Section 3: Results ---
st.header("3. Experimental Results")

col3, col4 = st.columns([1, 1])

with col3:
    st.markdown("""
    The plot on the right demonstrates the performance of this control loop in a simulation.

    ### Top Row: Flux Stability
    - **Light Blue (Open Loop)**: The flux at each output fluctuates wildly due to the atmospheric turbulence.
    - **Orange (Closed Loop)**: Once the ABCD tracking is activated, the fluxes stabilize. Output 0 (Input A) is locked to a dark state (Nulling), while other outputs stabilize at their respective quadrature points.

    ### Bottom Row: Phase Tracking
    - **Dashed Lines**: The uncorrected atmospheric phase drift for the two inputs.
    - **Purple Line**: The differential phase error (Atmosphere).
    - **Green Line**: The correction applied by the phase shifter.
    
    **Result**: The correction (Green) perfectly tracks the disturbance (Purple), canceling out the error and stabilizing the fringes.
    """)

with col4:
    show_asset("results.png", "Fringe Tracking Performance: Open Loop vs Closed Loop")

st.divider()


with st.expander("🤝 Acknowledgements"):
    st.markdown("""
    * **Nick Cvetojevic** for the original idea of performing ABCD fringe tracking using only 2 inputs on a 4x4 MMI.
    """)
