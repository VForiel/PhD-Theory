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
from fitter import Fitter, get_distributions

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

st.header("Overview")

st.markdown(r"""
In order to perform deep statistical analysis, we need an analytical model of the distribution of our test statistic under $\mathcal{H}_0$ and $\mathcal{H}_1$.
This page allows you to explore the distribution of the Kernel Nuller output and fit different analytical models to it.
""")

tab_star, tab_planet = st.tabs(["Star Models", "Planet Models"])

with tab_star:
    st.markdown("**Models for stellar leakage**")
    t_imb, t_cauchy, t_laplace = st.tabs(["IMB", "Cauchy", "Laplace"])
    
    with t_imb:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(r"""
            **Intensity Mismatch Balance (IMB)**
            
            Proposed by Dannert et al. (2025) for nulling interferometry.
            It models the distribution of the output signal when dominated by intensity mismatches between the beams.
            Mathematically, it relates to the Bessel K function, generalizing distributions that arise from the product of Gaussian variables.
            
            **Analytic Form:**  
            $$f(x; \mu, \sigma, \nu) \propto \left| \frac{x-\mu}{\sigma} \right|^{\nu/2} K_{\nu/2}\left( \left| \frac{x-\mu}{\sigma} \right| \right)$$
            *(Approximation of form)*
            
            **Parameters & Constraints:**
            - $\mu \in \mathbb{R}$: Location parameter (mean).
            - $\sigma > 0$: Scale parameter, width of the core.
            - $\nu > 0$: Shape parameter (degrees of freedom). Small $\nu$ $\to$ heavy tails, Large $\nu$ $\to$ Gaussian-like.
            """)
        with c2:
            st.image(str(WEB / "assets/img/models/imb_plot.png"), caption="Effect of ν on tail weight")

    with t_cauchy:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(r"""
            **Cauchy Distribution**
            
            A heavy-tailed distribution that arises as the ratio of two independent standard normal variables.
            It is notable for having undefined mean and variance. In physics, it describes resonance shapes (Lorentzian).
            In our context, it serves as a robust baseline for data with frequent extreme outliers.
            
            **Analytic Form:**
            $$f(x; x_0, \gamma) = \frac{1}{\pi\gamma \left[1 + \left(\frac{x-x_0}{\gamma}\right)^2\right]}$$
            
            **Parameters & Constraints:**
            - $x_0 \in \mathbb{R}$: Location parameter (peak), median.
            - $\gamma > 0$: Scale parameter (HWHM).
            """)
        with c2:
            st.image(str(WEB / "assets/img/models/cauchy_plot.png"), caption="Effect of scale γ")

    with t_laplace:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(r"""
            **Laplace Distribution**
            
            The distribution of the difference of two independent exponential random variables.
            It has fatter tails than a Gaussian but thinner than Cauchy. It is peaked at the mean.
            
            **Analytic Form:**
            $$f(x; \mu, b) = \frac{1}{2b} \exp \left( - \frac{|x-\mu|}{b} \right)$$
            
            **Parameters & Constraints:**
            - $\mu \in \mathbb{R}$: Location parameter.
            - $b > 0$: Scale parameter (diversity). Variance is $2b^2$.
            """)
        with c2:
            st.image(str(WEB / "assets/img/models/laplace_plot.png"), caption="Effect of diversity b")

with tab_planet:
    st.markdown("**Models for companion contribution**")
    t_beta, t_gamma = st.tabs(["Beta", "Gamma"])
    
    with t_beta:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(r"""
            **Beta Distribution**
            
            A flexible distribution defined on the bounded interval $[0, 1]$.
            In the context of nulling, the planet signal intensity fluctuates between zero (destructive) and a maximum (constructive interference), naturally fitting a bounded domain.
            
            **Analytic Form:**
            $$f(x; \alpha, \beta) \propto x^{\alpha-1} (1-x)^{\beta-1}$$
            
            **Parameters & Constraints:**
            - $\alpha > 0$: Shape parameter 1.
            - $\beta > 0$: Shape parameter 2.
            """)
        with c2:
            st.image(str(WEB / "assets/img/models/beta_plot.png"), caption="Various Beta shapes")

    with t_gamma:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(r"""
            **Gamma Distribution**
            
            A generalization of the exponential distribution, often used to model the sum of squared Gaussian variables (Chi-squared like).
            Relevant for intensity distributions where the signal is the sum of field amplitudes squared.
            
            **Analytic Form:**
            $$f(x; k, \theta) = \frac{x^{k-1} e^{-x/\theta}}{\theta^k \Gamma(k)}$$
            
            **Parameters & Constraints:**
            - $k > 0$: Shape parameter.
            - $\theta > 0$: Scale parameter. Mean is $k\theta$, Variance is $k\theta^2$.
            """)
        with c2:
            st.image(str(WEB / "assets/img/models/gamma_plot.png"), caption="Effect of shape k")

st.divider()

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
    c = np.abs((z-μ) / (σ * np.sqrt(ν)))
    
    # Bessel K
    k_val = scipy.special.kv(v, c)
    
    # Combine
    # Handle potential NaNs at 0
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

# Visibility Selector (Global)
st.subheader("Model Display Configuration")

col_sel1, col_sel2 = st.columns(2)

with col_sel1:
    available_models_star = ["IMB", "Cauchy", "Laplace", "Fitter Top 1", "Fitter Top 2", "Fitter Top 3"]
    default_models_star = ["IMB", "Fitter Top 1"]
    selected_models_star = st.multiselect("Star Models", available_models_star, default=default_models_star)

with col_sel2:
    available_models_planet = ["Beta", "Gamma", "Fitter Top 1", "Fitter Top 2", "Fitter Top 3"]
    default_models_planet = ["Beta", "Gamma", "Fitter Top 1"]
    selected_models_planet = st.multiselect("Planet Models", available_models_planet, default=default_models_planet)

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
        # Pre-generate noise to share it
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
    def fit_and_plot(ax, data, label, color, selected_models, do_fits=True, log_scale=True, do_fitter=False, explicit_beta=False, precomputed_models=None):
        # Histogram
        bins = 100
        q1, q99 = np.percentile(data, [0.5, 99.5])
        range_plot = (q1 - 0.5*(q99-q1), q99 + 0.5*(q99-q1))
        
        # Compute scaling factor for PDF to match counts
        bin_width = (range_plot[1] - range_plot[0]) / bins
        factor = len(data) * bin_width
        
        # Display counts (density=False)
        ax.hist(data, bins=bins, range=range_plot, density=False, alpha=0.5, color=color, label=label, log=log_scale)
        
        x_vals = np.linspace(range_plot[0], range_plot[1], 1000)
        
        results_list = []
        fitted_models = {}

        # If precomputed_models is provided, we use it. Otherwise we build it.
        use_precomputed = precomputed_models is not None
        current_models = precomputed_models if use_precomputed else {}

        if (do_fits or explicit_beta) and len(data) > 10:
            
            if do_fits:
                # 1. Cauchy
                if "Cauchy" in current_models:
                     params_cauchy = current_models["Cauchy"]["params"]
                     sse_cauchy = current_models["Cauchy"]["sse"]
                     success = True
                elif not use_precomputed:
                    try:
                        params_cauchy = stats.cauchy.fit(data)
                        hist_vals, bin_edges = np.histogram(data, bins=100, density=True)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        pdf_at_bins = stats.cauchy.pdf(bin_centers, *params_cauchy)
                        sse_cauchy = np.sum((hist_vals - pdf_at_bins) ** 2)
                        fitted_models["Cauchy"] = {"params": params_cauchy, "sse": sse_cauchy}
                        success = True
                    except:
                        success = False
                else: 
                     success = False

                if success:
                    results_list.append({"model_name": "Cauchy", "sumsquare_error": sse_cauchy, "aic": np.nan, "bic": np.nan})
                    if "Cauchy" in selected_models:
                        pdf_cauchy = stats.cauchy.pdf(x_vals, *params_cauchy)
                        ax.plot(x_vals, pdf_cauchy * factor, 'r--', label='Cauchy', linewidth=1.5)

                # 2. Laplace
                if "Laplace" in current_models:
                     params_laplace = current_models["Laplace"]["params"]
                     sse_laplace = current_models["Laplace"]["sse"]
                     success = True
                elif not use_precomputed:
                    try:
                        params_laplace = stats.laplace.fit(data)
                        hist_vals, bin_edges = np.histogram(data, bins=100, density=True)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        pdf_at_bins = stats.laplace.pdf(bin_centers, *params_laplace)
                        sse_laplace = np.sum((hist_vals - pdf_at_bins) ** 2)
                        fitted_models["Laplace"] = {"params": params_laplace, "sse": sse_laplace}
                        success = True
                    except:
                        success = False
                else: success = False

                if success:
                    results_list.append({"model_name": "Laplace", "sumsquare_error": sse_laplace, "aic": np.nan, "bic": np.nan})
                    if "Laplace" in selected_models:
                         pdf_laplace = stats.laplace.pdf(x_vals, *params_laplace)
                         ax.plot(x_vals, pdf_laplace * factor, color='cyan', linestyle=':', linewidth=2, label='Laplace')

                # 3. IMB
                if "IMB" in current_models:
                     params_imb = current_models["IMB"]["params"]
                     sse_imb = current_models["IMB"]["sse"]
                     success = True
                elif not use_precomputed:
                    try:
                        guess_imb = [np.median(data), np.std(data), 0.8]
                        res_imb = minimize(imb_cost, guess_imb, args=(data, 100), bounds=[(None, None), (1e-6, None), (0.1, 10.0)])
                        if res_imb.success:
                            params_imb = res_imb.x
                            sse_imb = imb_cost(params_imb, data, 100)
                            fitted_models["IMB"] = {"params": params_imb, "sse": sse_imb}
                            success = True
                        else: success = False
                    except: success = False
                else: success = False
                
                if success:
                    results_list.append({"model_name": "IMB", "sumsquare_error": sse_imb, "aic": np.nan, "bic": np.nan})
                    if "IMB" in selected_models:
                        pdf_imb = imb(x_vals, *params_imb)
                        ax.plot(x_vals, pdf_imb * factor, 'k-', linewidth=2, label='IMB')
            
            # 3. Explicit Beta & Gamma (for Planet)
            if explicit_beta:
                 # Beta
                 if "Beta" in current_models:
                     params_beta = current_models["Beta"]["params"]
                     sse_beta = current_models["Beta"]["sse"]
                     success = True
                 elif not use_precomputed:
                     try:
                        params_beta = stats.beta.fit(data)
                        hist_vals, bin_edges = np.histogram(data, bins=100, density=True)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        pdf_at_bins = stats.beta.pdf(bin_centers, *params_beta)
                        sse_beta = np.sum((hist_vals - pdf_at_bins) ** 2)
                        fitted_models["Beta"] = {"params": params_beta, "sse": sse_beta}
                        success = True
                     except: success = False
                 else: success = False

                 if success:
                    results_list.append({"model_name": "Beta", "sumsquare_error": sse_beta, "aic": np.nan, "bic": np.nan})
                    if "Beta" in selected_models:
                        pdf_beta = stats.beta.pdf(x_vals, *params_beta)
                        ax.plot(x_vals, pdf_beta * factor, color='purple', linestyle='-.', linewidth=2, label='Beta')
                 
                 # Gamma
                 if "Gamma" in current_models:
                     params_gamma = current_models["Gamma"]["params"]
                     sse_gamma = current_models["Gamma"]["sse"]
                     success = True
                 elif not use_precomputed:
                     try:
                        params_gamma = stats.gamma.fit(data)
                        hist_vals, bin_edges = np.histogram(data, bins=100, density=True)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        pdf_at_bins = stats.gamma.pdf(bin_centers, *params_gamma)
                        sse_gamma = np.sum((hist_vals - pdf_at_bins) ** 2)
                        fitted_models["Gamma"] = {"params": params_gamma, "sse": sse_gamma}
                        success = True
                     except: success = False
                 else: success = False

                 if success:
                    results_list.append({"model_name": "Gamma", "sumsquare_error": sse_gamma, "aic": np.nan, "bic": np.nan})
                    if "Gamma" in selected_models:
                        pdf_gamma = stats.gamma.pdf(x_vals, *params_gamma)
                        ax.plot(x_vals, pdf_gamma * factor, color='darkorange', linestyle='--', linewidth=2, label='Gamma')

        # 4. Fitter (Automated search)
        need_fitter = any("Fitter Top" in s for s in selected_models)
        
        if do_fitter and need_fitter and len(data) > 10:
             
             # If precomputed has Fitter results, use them
            if "Fitter" in current_models:
                f_results = current_models["Fitter"] # list of dicts {name, params, sse, aic, bic}
                
                # Plotting logic
                for rank in [1, 2, 3]:
                     if f"Fitter Top {rank}" in selected_models and len(f_results) >= rank:
                         res = f_results[rank-1]
                         name = res["name"]
                         params = res["params"]
                         dist_obj = getattr(stats, name)
                         pdf = dist_obj.pdf(x_vals, *params)
                         
                         styles = [':', '--', '-.']
                         colors = ['g', 'c', 'm']
                         ax.plot(x_vals, pdf * factor, color=colors[rank-1], linestyle=styles[rank-1], linewidth=2, label=f'Top {rank}: {name}')
                
                # Table Logic
                for res in f_results[:3]:
                     results_list.append({
                         "model_name": res["name"],
                         "sumsquare_error": res["sumsquare_error"],
                         "aic": res["aic"],
                         "bic": res["bic"]
                     })
            
            elif not use_precomputed:
                # Run Fitter
                with st.spinner(f"Fitting all available distributions for {label}... (this may take a moment)"):
                    try:
                        f = Fitter(data, distributions=get_distributions(), timeout=10, bins=bins)
                        f.fit(progress=False)
                        
                        fitter_data = [] # To store in fitted_models
                        
                        if hasattr(f, 'df_errors'):
                             sorted_df = f.df_errors.sort_values('sumsquare_error')
                             top_names = sorted_df.index.tolist()
                             
                             # Store top 5? just in case
                             for name in top_names[:5]:
                                 fitter_data.append({
                                     "name": name,
                                     "params": f.fitted_param[name],
                                     "sumsquare_error": sorted_df.loc[name, "sumsquare_error"],
                                     "aic": sorted_df.loc[name, "aic"],
                                     "bic": sorted_df.loc[name, "bic"]
                                 })
                             
                             fitted_models["Fitter"] = fitter_data
                             
                             # Now plot/add to current results
                             for rank in [1, 2, 3]:
                                 if f"Fitter Top {rank}" in selected_models and len(top_names) >= rank:
                                     name = top_names[rank-1]
                                     params = f.fitted_param[name]
                                     dist_obj = getattr(stats, name)
                                     pdf = dist_obj.pdf(x_vals, *params)
                                     
                                     styles = [':', '--', '-.']
                                     colors = ['g', 'c', 'm']
                                     ax.plot(x_vals, pdf * factor, color=colors[rank-1], linestyle=styles[rank-1], linewidth=2, label=f'Top {rank}: {name}')
                             
                             for name in top_names[:3]:
                                 results_list.append({
                                     "model_name": name,
                                     "sumsquare_error": sorted_df.loc[name, "sumsquare_error"],
                                     "aic": sorted_df.loc[name, "aic"],
                                     "bic": sorted_df.loc[name, "bic"]
                                 })
                    except Exception as e:
                        pass
        
        ax.set_title(label + (" (Log)" if log_scale else " (Linear)"))
        ax.set_ylabel("Occurrences")
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.3)
        
        # Combine and Deduplicate Results
        import pandas as pd
        if results_list:
            df_res = pd.DataFrame(results_list)
            # Sort by SSE
            df_res = df_res.sort_values("sumsquare_error")
            # Drop duplicates (e.g. if Beta was found by Fitter AND explicit)
            df_res = df_res.drop_duplicates(subset=["model_name"])
            return df_res, fitted_models
        return None, fitted_models

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
                
                # Config logic
                is_planet_only = (label == "Planet Only")
                is_combined = (label == "Star + Planet")
                
                # Disable fancy fitting for combined
                should_fit_standard = not is_planet_only and not is_combined
                should_fit_fitter = not is_combined
                should_fit_beta = is_planet_only
                
                # Select appropriate model list
                models_to_use = []
                if is_planet_only:
                    models_to_use = selected_models_planet
                elif not is_combined: # Star Only
                    models_to_use = selected_models_star
                else:
                    models_to_use = [] # Or default?
                
                col_lin, col_log = st.columns(2)
                
                # Linear
                with col_lin:
                    fig1, ax1 = plt.subplots(figsize=(6, 3))
                    results_df, fitted = fit_and_plot(ax1, data, label, col, models_to_use,
                                                  do_fits=should_fit_standard, 
                                                  log_scale=False, 
                                                  do_fitter=should_fit_fitter, 
                                                  explicit_beta=should_fit_beta,
                                                  precomputed_models=None)
                    st.pyplot(fig1)

                # Log
                with col_log:
                    fig2, ax2 = plt.subplots(figsize=(6, 3))
                    # Reuse fitted models!
                    _ , _ = fit_and_plot(ax2, data, label, col, models_to_use, 
                                     do_fits=should_fit_standard, 
                                     log_scale=True, 
                                     do_fitter=should_fit_fitter,
                                     explicit_beta=should_fit_beta,
                                     precomputed_models=fitted)
                    st.pyplot(fig2)
                
                # Display Top Models Table
                if results_df is not None:
                     st.caption(f"Model Performance for {label}")
                     # Format scientific notation
                     st.dataframe(
                         results_df.style.format({
                             "sumsquare_error": "{:.2e}",
                             "aic": "{:.2f}",
                             "bic": "{:.2f}"
                         }),
                         use_container_width=True
                     )
    
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

st.divider()
with st.expander("📚 References"):
    st.markdown("""
    - **Dannert et al. (2025)**: *Consequences of non-Gaussian instrumental noise in perturbed nulling interferometers*.
    """)