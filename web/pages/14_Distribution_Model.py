import sys
from pathlib import Path
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from copy import deepcopy as copy
from scipy import stats
from scipy.optimize import minimize
import scipy.special

# --- Path Setup ---
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
WEB = ROOT / "web"
if str(WEB) not in sys.path:
    sys.path.insert(0, str(WEB))

from phise import Context
from utils.context_widget import context_widget
from phise.modules import test_statistics as ts

# --- Page Config ---
st.set_page_config(
    page_title="Distribution Model",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Distribution Model")

st.markdown(r"""
To implement the Likelihood Ratio test, we need an analytical model of the distribution of our test statistic under $H_0$ and $H_1$.
This page allows you to explore the distribution of the Kernel Nuller output and fit different analytical models to it.
""")

# --- IMB Model Definition ---
def imb(z, μ, σ, ν):
    """
    IMB distribution model defined by Dannert et al. 2025.
    Based on the Modified Bessel function of the second kind.
    """
    # Safety against invalid parameters
    if σ <= 0 or ν <= 0:
        return np.zeros_like(z)

    v = (ν - 1)/2
    # Normalization factors
    a = 2**((1-ν)/2) * np.sqrt(ν)
    b = σ * np.sqrt(np.pi) * scipy.special.gamma(ν/2)
    
    # Argument
    # Handle z-μ close to 0 carefully? Bessel K_v tends to infinity at 0 if v > 0
    # But usually c is absolute value.
    c = np.abs((z-μ) / (σ * np.sqrt(ν)))
    
    # Bessel K
    k_val = scipy.special.kv(v, c)
    
    # Combine
    # Handle potential NaNs at 0
    # if c -> 0, z->μ. 
    # c^v * K_v(c). For small x, K_v(x) ~ x^-v. So c^v * c^-v ~ constant.
    # We can use a safe computation or just let scipy handle it (usually okay).
    
    pdf = (a / b) * c**v * k_val
    return np.nan_to_num(pdf, nan=0.0)

# Cost function for IMB fitting
def imb_cost(params, data, bins):
    μ, σ, ν = params
    
    # Regularization/Bounds check handled by optimizer bounds, but double check
    if σ <= 1e-9 or ν <= 0:
        return np.inf

    # We fit against the histogram
    hist_vals, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    model_vals = imb(bin_centers, μ, σ, ν)
    
    # Simple L2 distance
    cost = np.sum((hist_vals - model_vals) ** 2)
    return cost

# --- Simulation Setup ---
st.header("Simulation Configuration")

# Context Widget
ctx = context_widget(
    key_prefix="dist_model",
    presets={
        "LIFE (Nulling)": Context.get_LIFE(),
        "VLTI": Context.get_VLTI()
    },
    default_preset="VLTI",
    expanded=False,
    show_advanced=True
)

col1, col2, col3 = st.columns([1,1,1])

with col1:
    st.subheader("Planet Signal")
    contrast = st.number_input("Planet Contrast", min_value=1e-9, max_value=1e-1, value=1e-2, format="%.1e")
    
with col2:
    st.subheader("Planet Position")
    fov_val = ctx.interferometer.fov.to(u.mas).value
    separation = st.number_input("Separation (mas)", min_value=0.0, max_value=fov_val/2, value=min(2.0, fov_val/2), step=0.1)
    angle = st.number_input("Position Angle (deg)", min_value=0.0, max_value=360.0, value=0.0, step=1.0)

with col3:
    st.subheader("Atmosphere")
    piston_rms = st.number_input("Piston RMS (nm)", min_value=0.0, max_value=5000.0, value=100.0, step=10.0)

# --- Run Analysis ---
if st.button("Generate Distributions & Fit Models", type="primary"):
    
    # 1. Update Context
    if len(ctx.target.companions) > 0:
        comp = ctx.target.companions[0]
        comp.c = contrast
        comp.ρ = separation * u.mas
        comp.θ = angle * u.deg
    
    ctx.Γ = piston_rms * u.nm
    
    # 2. Prepare Contexts (Star Only, Planet Only)
    # Complete
    ctx_full = copy(ctx)
    
    # Star Only
    ctx_so = copy(ctx)
    ctx_so.target.companions = []
    
    # Planet Only (Trick: Scale f and c to maintain signal but negligible star)
    ctx_po = copy(ctx)
    scale = 1e12
    ctx_po.target.f /= scale
    if len(ctx_po.target.companions) > 0:
        ctx_po.target.companions[0].c *= scale

    # 3. Simulation
    N_SAMPLES = 5000 # Enough for decent looking histograms
    
    with st.spinner("Simulating distributions..."):
        # We need shared pistons for coherence if we wanted to sum them exacty, 
        # but here we just want representative distributions. 
        # Independent sampling is fine for distribution shape analysis.
        # Actually, for "Planet Only" vs "Star Only", using same noise realization is good for checks,
        # but for distribution fitting, independent is also fine.
        # Let's use independent to be simple and fast using phise observe.
        
        # We need raw kernel outputs.
        # We'll stick to a loop or helper if available. 
        # ts.get_vectors returns H0 and H1 (StarOnly and Full).
        # We need PlanetOnly as well.
        # Let's do it manually to get all 3.
        
        # Pre-generate noise to share it (optional but nice)
        pistons = np.random.normal(0, ctx.Γ.value, size=(N_SAMPLES, len(ctx.interferometer.telescopes))) * ctx.Γ.unit
        
        # Star Only
        outs_so = []
        for i in range(N_SAMPLES):
            o = ctx_so.observe(upstream_pistons=pistons[i])
            outs_so.append(ctx_so.interferometer.chip.process_outputs(o))
        data_so = np.array(outs_so) # (N, 3)

        # Planet Only
        outs_po = []
        for i in range(N_SAMPLES):
            o = ctx_po.observe(upstream_pistons=pistons[i])
            outs_po.append(ctx_po.interferometer.chip.process_outputs(o))
        data_po = np.array(outs_po)

        # Full (Star + Planet)
        outs_full = []
        for i in range(N_SAMPLES):
            o = ctx_full.observe(upstream_pistons=pistons[i])
            outs_full.append(ctx_full.interferometer.chip.process_outputs(o))
        data_full = np.array(outs_full)
        
        st.success(f"Generated {N_SAMPLES} samples for each scenario.")

    # 4. Distributions Fitting & Plotting
    st.subheader("Distributions & Model Fitting")
    
    # Helper for fitting and plotting
    def fit_and_plot(ax, data, label, color, do_fits=True, log_scale=True):
        # Histogram
        bins = 100
        q1, q99 = np.percentile(data, [0.5, 99.5])
        range_plot = (q1 - 0.5*(q99-q1), q99 + 0.5*(q99-q1))
        # Compute scaling factor for PDF to match counts
        # Factor = Total Count * Bin Width
        bin_width = (range_plot[1] - range_plot[0]) / bins
        factor = len(data) * bin_width
        
        # Display counts (density=False)
        ax.hist(data, bins=bins, range=range_plot, density=False, alpha=0.5, color=color, label=label, log=log_scale)
        
        if do_fits and len(data) > 10:
            x_vals = np.linspace(range_plot[0], range_plot[1], 1000)
            
            # Cauchy
            try:
                params_cauchy = stats.cauchy.fit(data)
                pdf_cauchy = stats.cauchy.pdf(x_vals, *params_cauchy)
                # Scale PDF to match counts
                ax.plot(x_vals, pdf_cauchy * factor, 'r--', label='Cauchy', linewidth=1.5)
            except:
                pass

            # IMB
            try:
                guess_imb = [np.median(data), np.std(data), 0.8]
                
                # Careful with bounds for IMB optimization
                res_imb = minimize(
                    imb_cost, 
                    guess_imb, 
                    args=(data, 100), 
                    bounds=[(None, None), (1e-6, None), (0.1, 10.0)]
                )
                if res_imb.success:
                    params_imb = res_imb.x
                    pdf_imb = imb(x_vals, *params_imb)
                    # Scale PDF to match counts
                    ax.plot(x_vals, pdf_imb * factor, 'k-', linewidth=2, label='IMB')
            except:
                pass
        
        ax.set_title(label + (" (Log)" if log_scale else " (Linear)"))
        ax.set_ylabel("Occurrences")
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.3)
        return

    # Prepare Tabs for Kernels
    tabs = st.tabs(["Kernel 1", "Kernel 2", "Kernel 3"])
    
    for k in range(3):
        with tabs[k]:
            # Data
            d_so = data_so[:, k].flatten()
            d_po = data_po[:, k].flatten()
            d_full = data_full[:, k].flatten()
            
            labels = ["Star Only", "Planet Only", "Star + Planet"]
            colors = ["blue", "green", "orange"]
            datasets = [d_so, d_po, d_full]
            
            for data, label, col in zip(datasets, labels, colors):
                st.markdown(f"**{label}**")
                col_lin, col_log = st.columns(2)
                
                # Linear
                with col_lin:
                    fig1, ax1 = plt.subplots(figsize=(6, 3))
                    fit_and_plot(ax1, data, "Linear", col, log_scale=False)
                    st.pyplot(fig1)

                # Log
                with col_log:
                    fig2, ax2 = plt.subplots(figsize=(6, 3))
                    fit_and_plot(ax2, data, "Log", col, log_scale=True)
                    st.pyplot(fig2)

    st.success("Analysis complete. The plots above show the breakdown of signal components and their respective model fits.")

    # 5. Transmission Maps
    st.subheader("Kernel Transmission Maps")
    _, maps = ctx.get_transmission_maps(N=100)
    
    fig_maps, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Planet Position calc
    comp = ctx.target.companions[0]
    r_p = comp.ρ.to(u.mas).value
    th_p = comp.θ.to(u.rad).value
    x_p = r_p * np.cos(th_p)
    y_p = r_p * np.sin(th_p)
    
    for k in range(3):
        ax = axes[k]
        im = ax.imshow(maps[k], extent=[-fov_val/2, fov_val/2, -fov_val/2, fov_val/2], origin='lower', cmap='bwr')
        ax.scatter(0, 0, marker='*', s=100, color='yellow', edgecolors='k')
        ax.scatter(x_p, y_p, marker='o', s=50, color='lime', edgecolors='k')
        ax.set_title(f"Kernel {k+1}")

    st.pyplot(fig_maps)

# --- Transition ---
st.divider()
st.markdown("""
Now that we have a valid analytical model for our distributions (the IMB model), we can construct the optimal test statistic: the **Likelihood Ratio**.

👉 **Next Step:** [Likelihood Ratio Test](Likelihood_Ratio)
""")
