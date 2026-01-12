
import streamlit as st
from pathlib import Path

# --- Path Setup ---
ROOT = Path(__file__).parent.parent.parent

st.set_page_config(
    page_title="Optical Bench: PHOBos",
    page_icon="🔬",
    layout="wide",
)

st.title("The PHOBos Optical Bench 🔬")

# --- Overview ---
st.header("Overview")
st.markdown(r"""
**PHOBos (PHOtonic Bench Operating System)** is a dedicated optical testbed designed to characterize and validate photonic integrated circuits (PICs) for nulling interferometry. 
It simulates the conditions of a stellar system observation in the laboratory to assess the performance of the chips before on-sky deployment.

### Key Interests
*   **Validation**: confirm the theoretical performance of the designed chips.
*   **Calibration**: develop and test algorithms to correct for fabrication errors.
""")

# --- Simulation (Bench Setup) ---
st.header("Bench Setup")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Global View")
    img_path = ROOT / "docs" / "img" / "PHOB_full1.png"
    if img_path.exists():
        st.image(str(img_path), caption="Global view of the PHOBos optical bench.", use_container_width=True)
    else:
        st.info("Global bench image not found.")

with col2:
    st.markdown("### Bench Scheme")
    img_path = ROOT / "docs" / "img" / "PHOB_full3.PNG"
    if img_path.exists():
        st.image(str(img_path), caption="Detailed view of the bench.", use_container_width=True)
    else:
        st.info("Scheme not found.")

st.info("**ToDo**: Add bench setup description.")


# --- PHOBos interface ---

st.header("PHOBos Control Interface")

st.markdown("""
A critical aspect of operating such a complex optical bench is the ability to efficiently control and synchronize a multitude of different instruments. 

This interface allows for:
*   **Centralized Control**: Controlling the Deformable Mirror (DM), Chip Controller (NICSLAB), Motorized Stages (Zaber, Newport), and Cameras (C-RED 3, Point Grey) from a single environment.
*   **Automation**: Scripting complex experimental sequences with a high abstraction level.

This work transforms a collection of hardware into a unified, programmable scientific instrument, enabling the lab validation of all the results presented in this thesis.
""")

st.success("""
##### ✨ My contribution

My main contribution to the PHOBos bench was the development of the complete Python interface (the PHOBos package) to pilot all the controllable equipment.
""")

# --- Footer Details ---

st.divider()

with st.expander("🤝 Acknowledgements"):
    st.markdown("""
    *   **Marc-Antoine Martinod** for leading the mounting and alignment of the bench.
    """)
