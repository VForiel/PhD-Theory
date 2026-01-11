
import streamlit as st
from pathlib import Path

# --- Path Setup ---
ROOT = Path(__file__).parent.parent.parent

st.set_page_config(
    page_title="Thermo-optic Phase Shifters",
    page_icon="🔥",
    layout="wide",
)

st.title("Thermo-optic Phase Shifters 🔥")

st.markdown(r"""
## Overview

In practice, the performance of a Kernel-Nuller is often limited by the fabrication imperfections of the optical components. These imperfections introduce phase aberrations that degrade the extinction capability (null depth).

To overcome this, we employ **Active Photonics**: specifically, **Thermo-optic Phase Shifters**. These devices allow us to fine-tune the optical path lengths within the chip to compensate for both manufacturing errors and external disturbances (like atmospheric piston).
""")

st.header("Principle of Operation")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown(r"""
    A thermo-optic phase shifter consists of a metallic heater deposited on top of a waveguide.
    
    1.  **Joule Heating**: Electrical power is injected into the heater.
    2.  **Thermo-optic Effect**: The heat locally modifies the **refractive index ($n$)** of the waveguide material via the thermo-optic coefficient ($\frac{dn}{dT}$).
    3.  **Phase Shift**: The change in refractive index alters the speed of light in that section, resulting in a controlled phase shift ($\phi$) relative to an unheated path.

    The phase shift $P$ can be modeled as:
    """)
    
    st.latex(r"P = e^{i\phi} = e^{i\frac{2\pi}{\lambda} \Delta L}")
    
    st.markdown(r"""
    Where:
    *   $\phi$ is the phase shift.
    *   $\Delta L$ is the equivalent Optical Path Difference (OPD).
    *   Both $\phi$ and $\Delta L$ are directly proportional to the electrical power dissipated in the heater.
    """)

with col2:
    img_path = ROOT / "docs" / "img" / "thermo-optic_phase_shifter.png"
    if img_path.exists():
        st.image(str(img_path), caption="Diagram of an integrated thermo-optic phase shifter.", use_container_width=True)
    else:
        st.info("Diagram image not found.")

st.divider()

st.header("Key Advantages")

st.markdown("""
*   **Speed**: Due to the microscopic size of the waveguides, the thermal mass is very small. This results in **low thermal inertia**, allowing the phase to be modified on a **millisecond time scale**.
*   **Real-time Correction**: The fast response time makes these shifters suitable for compensating variable phase aberrations in real-time.
*   **Integration**: High density integration allows for complex photonic circuits with multiple active control points.
""")
