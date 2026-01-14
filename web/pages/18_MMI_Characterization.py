import streamlit as st
from pathlib import Path

# --- Path Setup ---
PAGE_DIR = Path(__file__).parent
WEB_DIR = PAGE_DIR.parent
ASSETS_DIR = WEB_DIR / "assets" / "img" / "MMI"

st.set_page_config(
    page_title="MMI Characterization",
    page_icon="🔬",
    layout="wide",
)

st.title("Characterization of the 4x4 MMI 🔬")

# --- Introduction ---
st.header("1. Introduction")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(r"""
    **Nulling interferometry** allows for the direct detection of exoplanets by extinguishing the light from the host star. 
    Using **integrated photonic components** like Multi-Mode Interferometers (MMI) significantly improves the stability and compactness of such instruments.

    This page summarizes the characterization of a **Active 4x4 MMI** component tested on the PHOBos bench.
    The component is equipped with 4 thermal phase shifters to control the input phases and correct for fabrication errors.
    """)

    st.markdown(r"""
    **Objectives:**
    1.  **Calibrate Phase Shifters**: Determine the relationship between applied power and induced phase shift.
    2.  **Characterize Matrix**: Reconstruct the complex transfer matrix ($\mathbf{M}$) of the component.
    3.  **Validate Performance**: Ensure the component can perform the required beam combinations for Kernel-Nulling.
    """)

    st.success("""
    ##### ✨ My contribution

    This systematic scanning method and the associated matrix fitting algorithm were developed and implemented by myself. This approach enables the complete characterization of the component without requiring external phase sensors or metrology system.
    """)
    
with col2:
    st.image(str(ASSETS_DIR / "4x4.png"), caption="Concept of the 4x4 MMI with integrated phase shifters.", use_container_width=True)

# --- Methodology ---
st.divider()
st.header("2. Characterization Method")

st.markdown(r"""
To characterize the component without external metrology, we use a **systematic scan** approach:
*   We cycle through all possible input combinations (Single input, Pairs, Triplets, All).
*   For each combination, we scan the 4 phase shifters individually.
*   We analyze the output intensity modulation to extract phase and amplitude information.

This generates a dataset of 256 observables which allows us to solve the inverse problem and reconstruct the system's matrices using a global fit.

**The Model:**
We model the system with a unitary Transfer Matrix $\mathbf{A}$ and an input Crosstalk Matrix $\mathbf{C_{in}}$:
$$
\begin{align}
\vec{E}_{out} = \mathbf{C_{out}} \cdot \mathbf{M} \cdot \mathbf{P} \cdot \mathbf{C_{in}} \cdot \vec{E}_{in}
\end{align}
$$

As $\mathbf{C_{out}}$ and $\mathbf{M}$ are degenrated, we merge them into a single complex matrix $\mathbf{A}$.

$$
\begin{align}
\vec{E}_{out} = \mathbf{A} \cdot \mathbf{P} \cdot \mathbf{C_{in}} \cdot \vec{E}_{in}
\end{align}
$$

In practice, we only observe the intensity $\vec{O}$ of the output fields $\vec{E}_{out}$ where $O_i = |E_{in,i}|^2$. We then have :

$$
\begin{align}
\vec{O} &= \left( \mathbf{A} \cdot \mathbf{P} \cdot \mathbf{C_{in}} \cdot \vec{E}_{in} \right)^\dagger \mathbf{A} \cdot \mathbf{P} \cdot \mathbf{C_{in}} \cdot \vec{E}_{in}\\
        &= \vec{E}_{in}^\dagger \cdot \mathbf{C_{in}}^\dagger \cdot \mathbf{P}^\dagger \cdot \mathbf{A}^\dagger \cdot \mathbf{A} \cdot \mathbf{P} \cdot \mathbf{C_{in}} \cdot \vec{E}_{in}
\end{align}
$$
""")

# --- Results: Model Fit ---
st.divider()
st.header("3. Model Validation")

st.write(" The unitary model converges satisfactorily to the experimental data, validating our understanding of the component.")

# Gallery of results
tab1, tab2, tab3, tab4 = st.tabs(["1 Input", "2 Inputs", "3 Inputs", "4 Inputs"])

with tab1:
    st.image(str(ASSETS_DIR / "response_1_input.png"), caption="Model vs Measurement: Single Input Response", use_container_width=True)
with tab2:
    st.image(str(ASSETS_DIR / "response_2_input.png"), caption="Model vs Measurement: Two Inputs Interference", use_container_width=True)
with tab3:
    st.image(str(ASSETS_DIR / "response_3_input.png"), caption="Model vs Measurement: Three Inputs Interference", use_container_width=True)
with tab4:
    st.image(str(ASSETS_DIR / "response_4_input.png"), caption="Model vs Measurement: All Inputs Interference", use_container_width=True)

# --- Results: Performance ---
st.divider()
st.header("4. Performance & Beam Combination")

st.markdown("""
Using the reconstructed matrix, we can simulate and optimize the behavior of the component. 
The images below show the theoretical output phasors (complex amplitude) of the 4 outputs when the 4 inputs are illuminated.
""")

col_res1, col_res2 = st.columns(2)

with col_res1:
    st.subheader("Before Optimization")
    st.image(str(ASSETS_DIR / "not_calibrated_phases_matrix_model.png"), caption="Output phasors with uncalibrated (flat) inputs.", use_container_width=True)
    st.warning("Without phase control, the outputs are random and do not perform useful combinations.")

with col_res2:
    st.subheader("After Optimization")
    st.image(str(ASSETS_DIR / "calibrated_phases_matrix_model.png"), caption="Output phasors with optimized input phases.", use_container_width=True)
    st.success("""
    **Optimized Behavior:**
    *   **Output 1**: Constructive Interference (Bright).
    *   **Output 4**: Destructive Interference (Null / Double Bracewell).
    *   **Outputs 2 & 3**: Quadrature outputs (symmetric).
    
    This confirms the component is suitable for **Kernel-Nulling**.
    """)

# --- Footer Details ---

st.divider()

with st.expander("🤝 Acknowledgements"):
    st.markdown("""
    *   **Marc-Antoine Martinod** for his help with the component alignment and the setup of the experiment.
    *   **Nick Cvetojevic** for his advice on the bench setup and characterization methods (systematic scan).
    """)
