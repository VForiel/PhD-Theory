"""Streamlit page for Hooke-Jeeves calibration metrics analysis."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# --- Path setup ---
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HOOKE_JEEVES_DIR = ROOT / "src" / "analysis" / "hooke_jeeves"
if str(HOOKE_JEEVES_DIR) not in sys.path:
    sys.path.insert(0, str(HOOKE_JEEVES_DIR))

import phise
import metrics as hj_metrics


def _display_metric_formulas() -> None:
    """Display metric formulas exactly as in the companion notebook."""
    st.markdown("### Metric definitions")

    formulas = [
        ("Average depth", r"\frac{\Sigma_i N_i}{B}"),
        ("Max depth", r"\frac{\max_i N_i}{B}"),
        ("Average flux", r"\Sigma_i N_i"),
        ("Max flux", r"\max_i N_i"),
    ]

    for label, equation in formulas:
        st.write(label)
        st.latex(equation)


def _build_considered_metrics(include_max_flux: bool) -> dict[str, callable]:
    """Build the calibration metrics dictionary from analysis functions."""
    selected_metrics = {
        "sum(Ni)/B": hj_metrics.average_depth,
        "max(Ni)/B": hj_metrics.max_depth,
        "sum(Ni)": hj_metrics.average_flux,
    }

    if include_max_flux:
        selected_metrics["max(Ni)"] = hj_metrics.max_flux

    return selected_metrics


st.set_page_config(
    page_title="Calibration Metrics Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Calibration Metrics Analysis")
st.markdown(
    """
Compare Hooke-Jeeves calibration metrics on the PHOBos context using the
same workflow as the notebook companion and the analysis module.
"""
)

_display_metric_formulas()

st.divider()
st.subheader("Run full analysis")

col1, col2 = st.columns(2)
with col1:
    sample_count = st.slider(
        "Number of samples",
        min_value=10,
        max_value=2000,
        value=200,
        step=10,
        help="Monte Carlo samples per calibration metric.",
    )
with col2:
    bins = st.slider(
        "Histogram bins",
        min_value=20,
        max_value=200,
        value=60,
        step=10,
    )

include_max_flux = st.checkbox("Include max(Ni) as calibration metric", value=False)

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

if st.button("Run analysis", type="primary", use_container_width=True):
    considered_metrics = _build_considered_metrics(include_max_flux)

    with st.spinner("Generating data..."):
        ctx = phise.examples.contexts.get_PHOB()
        data = hj_metrics.generate_data(ctx, metrics=considered_metrics, samples=sample_count)

    with st.spinner("Generating plots..."):
        fig_evolution, _, df_evolution = hj_metrics.plot_metric_evolution(data)
        fig_nulls, _, df_nulls = hj_metrics.plot_final_null_depth_distributions(
            data,
            bins=bins,
        )
        fig_metrics, _, df_metrics = hj_metrics.plot_final_metric_distributions(
            data,
            bins=bins,
        )

    st.session_state.analysis_results = {
        "fig_evolution": fig_evolution,
        "fig_nulls": fig_nulls,
        "fig_metrics": fig_metrics,
        "df_evolution": df_evolution,
        "df_nulls": df_nulls,
        "df_metrics": df_metrics,
    }

    st.success("Analysis completed.")

results = st.session_state.analysis_results
if results is not None:
    st.divider()
    st.subheader("Metric evolution")
    st.pyplot(results["fig_evolution"], use_container_width=True)
    plt.close(results["fig_evolution"])
    st.dataframe(results["df_evolution"], use_container_width=True)

    st.divider()
    st.subheader("Final null depth distributions")
    st.pyplot(results["fig_nulls"], use_container_width=True)
    plt.close(results["fig_nulls"])
    st.dataframe(results["df_nulls"], use_container_width=True)

    st.divider()
    st.subheader("Final metric distributions")
    st.pyplot(results["fig_metrics"], use_container_width=True)
    plt.close(results["fig_metrics"])
    st.dataframe(results["df_metrics"], use_container_width=True)

    export_df = pd.concat(
        [
            results["df_evolution"].assign(source="metric_evolution"),
            results["df_nulls"].assign(source="final_null_depth_distributions"),
            results["df_metrics"].assign(source="final_metric_distributions"),
        ],
        ignore_index=True,
    )

    st.download_button(
        label="Download analysis tables as CSV",
        data=export_df.to_csv(index=False),
        file_name="hooke_jeeves_metrics_analysis.csv",
        mime="text/csv",
    )
