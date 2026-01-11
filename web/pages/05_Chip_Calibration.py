"""
Streamlit page for photonic chip calibration analysis.

This page explores chip calibration algorithms: Trial & Error (genetic approach)
and Obstruction approach. It reproduces the functionality from cells 23-25
in the Thesis notebook.
"""

from pathlib import Path
import sys

import streamlit as st
import numpy as np
import astropy.units as u
from copy import deepcopy as copy

# --- Path Setup ---
ROOT = Path(__file__).parent.parent.parent
# Ensure project root is on path so `src` is importable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WEB = ROOT / "web"
if str(WEB) not in sys.path:
    sys.path.insert(0, str(WEB))

from phise import Context
from utils.context_widget import context_widget
from src import analysis

st.set_page_config(
    page_title="Chip Calibration",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Photonic Chip Calibration ⚙️")

st.markdown(r"""
## Overview

The photonic chip contains 14 integrated thermo-optic phase shifters that must be
calibrated to compensate for fabrication imperfections and upstream optical errors.
This page demonstrates two calibration approaches:

1. **Trial & Error (Genetic Algorithm)**: Iteratively optimizes the phase settings
   by exploring the parameter space to maximize the kernel null depth.
   
2. **Obstruction Approach**: Sequentially calibrates pairs of inputs by obstructing
   others, simplifying the optimization to 1D problems and enabling robust null recovery.

Both methods aim to retrieve the ideal phase shifter settings (φ₁, ..., φ₁₄) that
compensate for unknown phase aberrations (σ₁, ..., σ₁₄).
""")

# Information sections
with st.expander("ℹ️ Trial & Error Details", expanded=False):
    st.subheader("Trial & Error")
    st.markdown(r"""
    In order to get the best shifts to inject to optimize the component performances, I made a genetic algorithme that iteratively mutate a shifter and keep the mutation if it minimize the associated metric. All the shifters that act on the bright channel are associated with the bright metric that must be maximized
    
    $$
    M_B = |B|^2
    $$
    
    While the other shifters are associated with the kernel metric that must be minimized.
    
    $$
    M_K = \sum_{n=1}^3|K_n|
    $$
    
    Merging these two metrics can induce local minimum since improving the bright metric can deterior the kernel metric. This separation is then necessary to ensure reaching a global minimum (empirically demonstrated)
    """)

with st.expander("ℹ️ Obstruction Approach Details", expanded=False):
    st.subheader("Obstruction")
    st.markdown(r"""
    By successively obstructing inputs, it is possible to simplify the optimization problem by adjusting only a single parameter and monitoring only a single output.
    
    There are different ways to proceed. Only one is detailed here.
    
    We start by obstructing inputs $I_2$ and $I_3$. Considering the architecture of our component, we can then describe the transfer function for the bright output $B$:
    
    $$
    B = \left|\left(a_1 e^{i(\theta_1 + \sigma_1 + \phi_1)} + a_2 e^{i(\theta_2 + \sigma_2 + \phi_2)}\right) e^{i(\sigma_5 + \phi_5)}\right|^2
    $$
    
    Where $a_n$ and $\theta_n$ represent the amplitude and phase of the input signals, respectively. $\sigma_n$ corresponds to the (unknown) phase perturbation associated with retarder $n$, and $\phi_n$ is the phase voluntarily injected via the retarder to attempt to compensate for this perturbation.
    
    As calibration is performed in the laboratory, we can assume a total intensity fixed at $1$ (arbitrary unit) and that each input receives the same flux, i.e. $a_1 = a_2 = 1/\sqrt{2}$, and is perfectly co-phased, i.e. $\theta_1 = \theta_2 = \theta$. Since we only have access to the signal intensity, we are insensitive to the global phase, allowing us to simplify the previous equation:
    
    $$
    B = \frac{1}{2} \left|e^{i(\sigma_1 + \phi_1)} + e^{i(\sigma_2 + \phi_2)}\right|^2
    $$
    
    By maximizing $B$, we should find $1$, which implies that:
    
    $$
    \sigma_1 + \phi_1 = \sigma_2 + \phi_2
    $$
    
    We can use $\phi_1$ as a reference (global phase) and thus fix it to 0, which yields:
    
    $$
    \phi_2 = \sigma_1 - \sigma_2
    $$
    
    We can then either perform different measurements of $B$ at fixed $\phi_2$ and deduce $\sigma_1$ and $\sigma_2$ by solving a system of equations, or dichotomously find the value of $\phi_2$ that maximizes $B$.
    """)

st.divider()

# =======================
# LAYER 2: Calibration-Specific Parameters (visible, override base context)
# =======================

st.subheader("Calibration")

presets = {
    "VLTI": Context.get_VLTI(),
    "LIFE": Context.get_LIFE(),
}
base_ctx = context_widget(
    key_prefix="calib",
    presets=presets,
    default_preset="VLTI",
    expanded=False,  # COLLAPSED by default
    show_advanced=True,
)

left_col, right_col = st.columns([1, 2])

# Output directory for generated figures
output_dir = ROOT / "generated" / "thesis" / "calibration"
output_dir.mkdir(parents=True, exist_ok=True)

left_col, right_col = st.columns([1, 2])

# LEFT COLUMN: Calibration Parameters
with left_col:

        
    # Manufacturing error settings
    st.caption("**Manufacturing Errors (σ)**")
    sigma_type = st.radio(
        "Error configuration",
        options=["Normal", "Zero (ideal)", "Custom"],
        index=0
    )
    
    if sigma_type == "Normal":
        sigma_std = st.number_input(
            "Error std dev (nm)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.1
        )

    elif sigma_type == "Custom":
        sigma_values = st.text_area(
            "Custom σ values (nm, comma-separated)",
            value=""
                "5.0, 3.2, 4.5, 6.1, 2.8, 5.5, 4.0, 3.3, 5.7, 4.8, 6.0, 2.9, 4.1, 5.2"
        )

    st.divider()

    st.caption("**Calibration Method**")
    
    # Calibration method selection
    calib_method = st.radio(
        "Calibration Method",
        options=["Trial & Error", "Obstruction"],
        index=0,
        help="Choose the calibration algorithm to apply"
    )
    
    # Monochromatic mode
    mono = st.checkbox(
        "Monochromatic mode",
        value=True,
        help="Use single wavelength (simplifies phase behavior)"
    )
    
    st.divider()
    
    if calib_method == "Trial & Error":
        st.caption("**Trial & Error Parameters**")
        beta = st.slider(
            "Target null depth β",
            min_value=0.5,
            max_value=1 - 1e-6,
            value=0.961,
            step=0.001,
            help="Decay factor for genetic algorithm fitness function (0.5 -> faster convergence, 1 -> more precise)"
        )
        
    else:  # Obstruction
        st.caption("**Obstruction Parameters**")
        n_samples = st.slider(
            "Number of samples",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100,
            help="Number of phase measurements per obstruction configuration"
        )

# Apply overrides to create working context
ctx = copy(base_ctx)
ctx.monochromatic = mono

# Set manufacturing errors based on selection
if sigma_type == "Zero (ideal)":
    ctx.interferometer.chip.σ = np.zeros(14) * u.nm
elif sigma_type == "Random":
    ctx.interferometer.chip.σ = np.abs(np.random.normal(0, sigma_std, 14)) * u.nm
elif sigma_type == "Custom":
    sigma_list = [float(s.strip()) for s in sigma_values.split(",") if s.strip()]
    if len(sigma_list) != 14:
        st.error("Please provide exactly 14 σ values for the phase shifters.")
    else:
        ctx.interferometer.chip.σ = np.array(sigma_list) * u.nm

ctx_pre = copy(ctx)

# RIGHT COLUMN: Visualization
with right_col:
    
    # Run calibration
    try:
        with st.spinner("Running calibration..."):
            if calib_method == "Trial & Error":

                trial_path = output_dir / "trial_error.png"

                ctx_post = analysis.calibration.genetic_approach(
                    ctx=ctx,
                    β=beta,
                    verbose=False,
                    figsize=(7, 7),
                    save_as=str(trial_path),
                )

                if trial_path.exists():
                    st.image(
                        str(trial_path),
                        width="stretch",
                        caption="Trial & Error Calibration (Genetic Algorithm)",
                    )
                else:
                    st.warning("Calibration image not found. Check save path or write permissions.")

            else:  # Obstruction

                obstruction_path = output_dir / "obstruction.png"

                ctx_post = analysis.calibration.obstruction_approach(
                    n=n_samples,
                    ctx=ctx,
                    figsize=(7, 7),
                    save_as=str(obstruction_path),
                )

                if obstruction_path.exists():
                    st.image(
                        str(obstruction_path),
                        width="stretch",
                        caption="Obstruction Calibration",
                    )
                else:
                    st.warning("Calibration image not found. Check save path or write permissions.")

    except Exception as e:
        st.error(f"❌ Calibration error: {str(e)}")
        st.info("Ensure the context is configured correctly and the analysis module is available.")

st.subheader("Null Depth Statistics: Before vs After Calibration")

# Controls
stats_cols = st.columns([1, 1])
with stats_cols[0]:
    context_mode = st.selectbox(
        "Context mode",
        options=["Laboratory", "Ground (VLTI)", "Space (LIFE)"],
        index=0,
        help="Choose the scenario for statistics"
    )
with stats_cols[1]:
    n_samples_stats = st.number_input(
        "Samples",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="Number of observations to sample for statistics"
    )

def compute_kernel_stats(ctx_stats: Context, n: int = 1000) -> np.ndarray:
    """
    Sample kernel outputs and return array of shape (3, n) for the three kernels.
    """
    data = np.empty((3, n))
    for i in range(n):
        outs = ctx_stats.observe()
        bright = outs[0]
        kernels = ctx_stats.interferometer.chip.process_outputs(outs)
        data[:, i] = kernels / bright
    return data

try:
    # Compute stats
    d_pre = compute_kernel_stats(ctx_pre, n=int(n_samples_stats))
    d_post = compute_kernel_stats(ctx_post, n=int(n_samples_stats))

    import pandas as pd

    def summarize(d: np.ndarray) -> pd.DataFrame:
        means = np.mean(d, axis=1)
        meds = np.median(d, axis=1)
        stds = np.std(d, axis=1)
        df = pd.DataFrame({
            "Kernel": ["K1", "K2", "K3"],
            "Mean": means,
            "Median": meds,
            "Std": stds,
        })
        return df

    df_pre = summarize(d_pre)
    df_post = summarize(d_post)

    tbl_cols = st.columns(2)
    with tbl_cols[0]:
        st.caption("Before calibration")
        st.dataframe(df_pre.style.format({"Mean": "{:.3e}", "Median": "{:.3e}", "Std": "{:.3e}"}), width="stretch")
    with tbl_cols[1]:
        st.caption("After calibration")
        st.dataframe(df_post.style.format({"Mean": "{:.3e}", "Median": "{:.3e}", "Std": "{:.3e}"}), width="stretch")

except Exception as e:
    st.error(f"❌ Error computing statistics: {e}")
    st.info("Ensure the context observation pipeline is available.")