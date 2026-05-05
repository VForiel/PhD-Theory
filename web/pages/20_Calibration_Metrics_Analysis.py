"""
Streamlit page for calibration metrics analysis on PHOBos N4x4-T8 chip.

This page analyzes different optimization metrics used in chip calibration,
specifically for the PHOBos N4x4-T8 photonic chip.
"""

from pathlib import Path
import sys

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from copy import deepcopy as copy
import tqdm
import pandas as pd

# --- Path Setup ---
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WEB = ROOT / "web"
if str(WEB) not in sys.path:
    sys.path.insert(0, str(WEB))

import phise
from src import analysis

st.set_page_config(
    page_title="Calibration Metrics Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Calibration Metrics Analysis 📊")

st.markdown("""
## PHOBos N4x4-T8 Chip Calibration

This page analyzes different optimization metrics used in photonic chip calibration.
Each metric defines how the optimization algorithm evaluates calibration quality on the **N4x4-T8 chip**.

**Available Metrics:**
- **sum(N) / B**: Ratio of total null outputs to bright output (lower is better)
- **max(N) / B**: Ratio of maximum null output to bright output (lower is better)  
- **sum(N)**: Total sum of null outputs (lower is better)
- **max(N)**: Maximum null output (lower is better)
""")

# =======================
# Initialize Session State
# =======================

if 'base_ctx' not in st.session_state:
    st.session_state.base_ctx = phise.examples.contexts.get_PHOB()
    st.session_state.base_ctx.camera.ideal = True

if 'metric_results' not in st.session_state:
    st.session_state.metric_results = {}

if 'bootstrap_data' not in st.session_state:
    st.session_state.bootstrap_data = None

# =======================
# Setup & Parameters
# =======================

st.subheader("Analysis Parameters")

col1, col2 = st.columns(2)

with col1:
    beta = st.slider(
        "Hooke-Jeeves step reduction (β):",
        min_value=0.5,
        max_value=0.99,
        value=0.9,
        step=0.05,
        help="Step reduction factor for Hooke-Jeeves optimization"
    )

with col2:
    aberration_scale = st.slider(
        "Aberration scale (λ/N):",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        help="Initial phase aberration scale (in units of λ/N)"
    )

n_bootstrap = st.slider(
    "Bootstrap samples (for comparison section):",
    min_value=5,
    max_value=100,
    value=20,
    step=5,
    help="Number of random initial conditions for bootstrap comparison"
)

# =======================
# Define Metrics
# =======================

def m1(outs):
    """Sum(N) / B"""
    outs = np.copy(outs)
    outs[outs <= 1] = 1
    return np.sum(outs[1:]) / outs[0]

def m2(outs):
    """Max(N) / B"""
    outs = np.copy(outs)
    outs[outs <= 1] = 1
    return np.max(outs[1:]) / outs[0]

def m3(outs):
    """Sum(N)"""
    outs = np.copy(outs)
    outs[outs <= 1] = 1
    return np.sum(outs[1:])

def m4(outs):
    """Max(N)"""
    outs = np.copy(outs)
    outs[outs <= 1] = 1
    return np.max(outs[1:])

metrics = {
    "sum(N) / B": m1,
    "max(N) / B": m2,
    "sum(N)": m3,
    "max(N)": m4,
}
    
# =======================
# SECTION 1: Individual Metric Tabs
# =======================

st.divider()
st.subheader("1️⃣ Individual Metric Calibration")

tab_names = list(metrics.keys())
tabs = st.tabs(tab_names)

for tab_idx, (tab, metric_name) in enumerate(zip(tabs, tab_names)):
    with tab:
        st.write(f"### {metric_name}")
        
        metric_func = metrics[metric_name]
        
        if st.button(
            f"▶ Run Calibration: {metric_name}",
            key=f"run_metric_{metric_name}",
            use_container_width=True,
            type="primary"
        ):
            # Create context copy with random aberrations
            ctx = copy(st.session_state.base_ctx)
            ctx.chip.σ = np.abs(np.random.normal(
                0, 1, len(ctx.chip.σ)
            )) * ctx.interferometer.λ / aberration_scale
            
            st.info(f"Running calibration with metric: **{metric_name}**...")
            
            # Run calibration
            history = ctx.chip.calibrate(
                plot=False,
                β=beta,
                hooke_jeeves_metric=metric_func
            )
            
            # Get results
            depths_history = history["depths"]
            mean_depth = np.mean(depths_history[-100:, :], axis=0)
            total_mean = np.mean(depths_history[-100:, :])
            
            # Store in session
            st.session_state.metric_results[metric_name] = {
                "history": history,
                "depths_history": depths_history,
                "mean_depth": mean_depth,
                "total_mean": total_mean,
                "ctx": ctx
            }
            
            # Display results
            st.success("✅ Calibration completed!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Total Mean Null Depth",
                    f"{total_mean:.2e}"
                )
            with col2:
                st.metric(
                    "Per-Output Mean",
                    f"{np.mean(mean_depth):.2e}"
                )
            with col3:
                st.metric(
                    "Best Output",
                    f"{np.min(mean_depth):.2e}"
                )
            
            # Show per-output results
            st.write("**Per-Output Results (last 100 iterations):**")
            output_data = []
            for i in range(len(mean_depth)):
                output_data.append({
                    "Output": i,
                    "Mean Null Depth": mean_depth[i],
                })
            
            df_output = pd.DataFrame(output_data)
            st.dataframe(df_output, use_container_width=True)
            
            # Plot convergence
            fig, ax = plt.subplots(figsize=(10, 5))
            for i in range(depths_history.shape[1]):
                ax.plot(depths_history[:, i], label=f"Output {i}", alpha=0.7, linewidth=2)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Null Depth")
            ax.set_title(f"Calibration Convergence: {metric_name}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        
        # Display cached results if available
        if metric_name in st.session_state.metric_results:
            st.info("📊 Last calibration result for this metric:")
            result = st.session_state.metric_results[metric_name]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Total Mean Null Depth",
                    f"{result['total_mean']:.2e}"
                )
            with col2:
                st.metric(
                    "Best Output",
                    f"{np.min(result['mean_depth']):.2e}"
                )
# =======================
# SECTION 2: Bootstrap Comparison
# =======================

st.divider()
st.subheader("2️⃣ Bootstrap Comparison")

st.markdown("""
Compare the performance of all metrics across multiple random initializations.
This section runs a bootstrap analysis to evaluate which metric is most robust.
""")

if st.button("▶ Run Bootstrap Comparison", type="primary", use_container_width=True):
    st.info(f"Running bootstrap analysis with {n_bootstrap} samples per metric...")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize data storage
    data = np.zeros((len(metrics), n_bootstrap, 3))  # (metric, sample, output)
    
    total_runs = len(metrics) * n_bootstrap
    current_run = 0
    
    for i in tqdm.tqdm(range(n_bootstrap), desc="Bootstrap samples"):
        for metric_idx, (name, metric) in enumerate(metrics.items()):
            
            # Update progress
            current_run += 1
            progress_bar.progress(current_run / total_runs)
            status_text.text(f"Running: {name} (sample {i+1}/{n_bootstrap})...")
            
            # Create random context copy
            ctx = copy(st.session_state.base_ctx)
            ctx.chip.σ = np.abs(np.random.normal(
                0, 1, len(ctx.chip.σ)
            )) * ctx.interferometer.λ / aberration_scale
            
            # Run calibration
            history = ctx.chip.calibrate(
                plot=False,
                β=beta,
                hooke_jeeves_metric=metric
            )
            
            # Store results (average of last 100 iterations)
            depths_history = history["depths"]
            mean_depth = np.mean(depths_history[-100:, :], axis=0)
            data[metric_idx, i] = mean_depth
    
    progress_bar.empty()
    status_text.empty()
    
    # Store in session
    st.session_state.bootstrap_data = data
    
    st.success("✅ Bootstrap analysis completed!")
    
    st.divider()
    
    # =======================
    # Results Visualization
    # =======================
    
    st.subheader("📊 Results Visualization")
    
    # Violin plots
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    
    for output_idx in range(3):
        violin_data = [data[i, :, output_idx] for i in range(len(metrics))]
        parts = axs[output_idx].violinplot(
            violin_data, 
            showmeans=True, 
            showmedians=True
        )
        
        axs[output_idx].set_xticks(np.arange(1, len(metrics) + 1))
        axs[output_idx].set_xticklabels(
            list(metrics.keys()), 
            rotation=45, 
            ha='right'
        )
        axs[output_idx].set_title(
            f"Output {output_idx} - Null Depth Distribution", 
            fontsize=12, 
            fontweight='bold'
        )
        axs[output_idx].set_ylabel("Mean Null Depth (last 100 iters)")
        axs[output_idx].grid(True, alpha=0.3, axis='y')
        axs[output_idx].set_yscale('log')
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    
    # =======================
    # Summary Statistics
    # =======================
    
    st.subheader("📈 Performance Summary")
    
    summary_cols = st.columns(len(metrics))
    
    for idx, name in enumerate(metrics.keys()):
        with summary_cols[idx]:
            st.write(f"**{name}**")
            
            # Calculate statistics
            metric_data = data[idx, :, :]
            
            st.metric("Mean", f"{np.mean(metric_data):.2e}")
            st.metric("Std", f"{np.std(metric_data):.2e}")
            st.metric("Min", f"{np.min(metric_data):.2e}")
            st.metric("Max", f"{np.max(metric_data):.2e}")
            
            # Per-output summary
            st.caption("**Per Output:**")
            for output_idx in range(3):
                output_data = data[idx, :, output_idx]
                st.caption(
                    f"Out {output_idx}: μ={np.mean(output_data):.2e}"
                )
    
    # =======================
    # Detailed Data Export
    # =======================
    
    st.subheader("💾 Export Results")
    
    # Create dataframe
    summary_list = []
    for metric_idx, name in enumerate(metrics.keys()):
        for sample_idx in range(n_bootstrap):
            for output_idx in range(3):
                summary_list.append({
                    "Metric": name,
                    "Sample": sample_idx + 1,
                    "Output": output_idx,
                    "Mean Null Depth": data[metric_idx, sample_idx, output_idx],
                })
    
    df = pd.DataFrame(summary_list)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Raw Data (first 20 rows):**")
        st.dataframe(df.head(20), use_container_width=True)
    
    with col2:
        st.write("**Summary Statistics:**")
        summary_df = df.groupby("Metric")["Mean Null Depth"].agg([
            "mean", "std", "min", "max", "count"
        ]).round(2e-10)
        st.dataframe(summary_df, use_container_width=True)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="phobos_metrics_bootstrap.csv",
        mime="text/csv",
    )

elif st.session_state.bootstrap_data is not None:
    # Display cached bootstrap results
    st.info("📊 Last bootstrap comparison results:")
    
    data = st.session_state.bootstrap_data
    
    # Quick summary
    summary_cols = st.columns(len(metrics))
    for idx, name in enumerate(metrics.keys()):
        with summary_cols[idx]:
            metric_data = data[idx, :, :]
            st.metric(f"{name}", f"{np.mean(metric_data):.2e}")

# =======================
# Info Section
# =======================

with st.expander("ℹ️ About This Analysis", expanded=False):
    st.markdown("""
    ### Context
    - **Setup**: PHOBos N4x4-T8 photonic chip with ideal camera
    - **Algorithm**: Hooke-Jeeves direct search optimization
    - **Goal**: Minimize null depth in kernel nulling outputs
    
    ### Methodology
    
    #### Individual Calibration
    For each metric, run a single calibration:
    1. Generate random initial phase aberrations
    2. Apply Hooke-Jeeves optimization with the chosen metric
    3. Visualize convergence and final null depths
    
    #### Bootstrap Comparison
    Compare all metrics across multiple random initializations:
    1. Run calibration for each metric with N random aberrations
    2. Record mean null depth for each output (last 100 iterations)
    3. Create violin plots showing performance distribution
    4. Compute summary statistics
    
    ### Interpretation
    - **Lower null depth** = better calibration performance
    - **Narrow distribution** = more stable metric
    - **Ratio metrics** (e.g., sum(N)/B) normalize by bright output
    - **Absolute metrics** (e.g., sum(N)) depend on flux levels
    
    ### References
    - PHISE: Photonic Interferometry Simulation for Exoplanets
    - PHOBos: Photonic optical bench for kernel nulling
    - Hooke-Jeeves: Classical direct search optimization method
    """)

st.divider()
st.markdown("""
<div style="text-align: center; font-size: 0.85em; color: #888;">
    Calibration Metrics Analysis for PHOBos N4x4-T8 Chip
</div>
""", unsafe_allow_html=True)
