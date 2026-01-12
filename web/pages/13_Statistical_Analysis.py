import sys
from pathlib import Path
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from copy import deepcopy as copy

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
import importlib
try:
    importlib.reload(ts)
except Exception:
    pass
from phise.modules.test_statistics import ALL_TESTS

# --- Page Config ---
st.set_page_config(
    page_title="Statistical Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Statistical Analysis")

# --- Overview ---
st.header("Overview")

col1, col2 = st.columns([2,1])

with col1:
    st.markdown(r"""
    Statistical analysis is crucial for determining the presence of a planet signal buried in noise.

    The planet signal creates several potential features in the data distribution that we aim to detect:
    1.  **Shift**: A global displacement of the distribution (if the signal is constant).
    2.  **Flattening**: A spread of the distribution around its center (variance increase).
    3.  **Asymmetry**: A skewness introduced by the signal.
    4.  **Bimodality**: The appearance of two peaks in the distribution (e.g., if the signal modulates).
    """)

with col2:
    img_path = ROOT / "docs" / "img" / "Kernel-dist.png"
    st.image(str(img_path), caption="Effect of a planet signal on the Kernel distribution.")

st.markdown("""
The **Likelihood Ratio** is theoretically the optimal test (Neyman-Pearson lemma). However, in practice, it is often **inapplicable** because it requires knowing the exact analytical form of the data distribution (which is often unknown or changing).

Therefore, we study other **test statistics** that are robust and do not require prior knowledge of the distribution.
""")

# --- Data Ansatz ---
st.subheader("Data Description (Ansatz)")
st.markdown(r"""
Before defining our test statistics, we model our data $\vec{x}$ (a vector of $N$ points, representing the null depth or kernel output) under the two hypotheses.
We assume the data is a combination of two **independent** signals:
1.   **Star Signal:** $\alpha S_* + n_*$ (where $S_*$ is the star signal, $\alpha$ its transmission, and $n_*$ the stellar leakage noise).
2.  **Planet Signal:** $\eta S_p + n_p$ (where $\eta$ is the system's transmission, $S_p$ the planet signal, and $n_p$ the associated noise).

By construction of the Kernel, the stellar transmission is nulled ($\alpha = 0$). Thus, for each observation $i$:

$$ x_i = n_{i,*} + \eta S_p + n_{i,p} $$

This lead to the following hypotheses:
- **$H_0$ (Null Hypothesis)**: No planet ($S_p = n_p = 0$).
  $$ x_i = n_{i,*} $$
- **$H_1$ (Alternative Hypothesis)**: Planet is present.
  $$ x_i = n_{i,*} + \eta S_p + n_{i,p} $$

**Important Properties:**
- In this study, our goal is solely detection (determining if $S_p \neq 0$).
- $\eta \in [-1, 1]$. It can be negative depending on the telescope arrangement. Since it can be negative, the signal shift can be in either direction. We use absolute values (either on the data $x_i$ or directly in the test statistic $T(x)$) to map the quantity to $\mathbb{R}^+$. This allows us to define a single upper detection threshold $\xi$.
- $n_*$ and $n_p$ follow unknown independent distributions.
""")

# --- Standard Tests ---
st.subheader("Studied Test Statistics")
ts_tabs = st.tabs(["Mean", "Median", "Kolmogorov-Smirnov", "Cramer von Mises", "Flattening", "Median of Abs"])

with ts_tabs[0]:
    st.markdown(r"""
    **Principle:** We take the average of the distribution and we compare it to a threshold.
    
    $$
    T(x) = \left|\frac{1}{N}\sum_i x_i \right| \stackrel{H_1}{\underset{H_0}{\gtrless}} \xi
    $$
    
    **Ansatz:**
    $$
    \begin{cases}
    H_0 : d = |\bar{n_*}|\\
    H_1 : d =  |\eta \hat{S_p} + \bar{n_*} + \bar{n_p}|
    \end{cases}
    $$
    Where $\hat{S_p}$ is the estimated planet signal magnitude.
    """)

with ts_tabs[1]:
    st.markdown(r"""
    **Principle:** We take the median of the distribution and compare it to a threshold.
    
    $$
    T(x) = \left| \text{median}(x) \right| \stackrel{H_1}{\underset{H_0}{\gtrless}} \xi
    $$
    
    **Ansatz:**
    $$
    \begin{cases}
    H_0 : d = |\tilde{n_*}|\\
    H_1 : d =  | \eta \hat{S_p} + \tilde{n_*} + \tilde{n_p} |
    \end{cases}
    $$
    Where $\tilde{n_*}$ and $\tilde{n_p}$ are the median noise components.
    """)

with ts_tabs[2]:
    st.markdown(r"""
    **Principle:** We compare the maximum distance on the cumulative distribution functions (CDF) of the two distributions.
    
    $$ T(x) = \sup_t |F_x(t) - F_{H_0}(t)| $$
    
    *Effective for detecting shifts/distortions in the distribution shape.*
    """)

with ts_tabs[3]:
    st.markdown(r"""
    **Principle:** We compare the total quadratic distance on the cumulative distribution functions of the two distributions.
    
    $$ T(x) = \int (F_x(t) - F_{H_0}(t))^2 \, dF_{H_0}(t) $$
    
    *Often more sensitive than KS to deviations across the whole distribution.*
    """)

with ts_tabs[4]:
    st.markdown(r"""
    **Principle:** Measures the dispersion around the median.
    
    $$
    \sum_{i=1}^{N} |x_i - \tilde{x}| \stackrel{H_1}{\underset{H_0}{\gtrless}} \xi
    $$
    with $\tilde{x} = \text{median}(x)$
    
    **Ansatz:**
    $$
    \begin{cases}
    H_0 : d = \sum |n_* - \tilde{n_*}|\\
    H_1 : d = \sum |(n_* + n_p) - (\tilde{n_*} + \tilde{n_p})|
    \end{cases}
    $$
    *Note: The spread increases due to the additional noise term $n_p$ in $H_1$.*
    """)

with ts_tabs[5]:
    st.markdown(r"""
    **Principle:** Robust estimator of scale.
    
    **Ansatz:**
    $$
    \begin{cases}
    H_0 : d = \text{median}(|n_*|)\\
    H_1 : d = \text{median}(|n_* + \eta S_p + n_p|)
    \end{cases}
    $$
    """)

st.divider()

# --- Likelihood Ratio Section ---
st.subheader("Likelihood Ratio (Theoretical Benchmark)")
st.markdown(r"""
If we **assume** a specific distribution, we can derive the optimal likelihood ratio test. We compare these theoretical optimums to our practical tests.

The core of the detection is the Likelihood Ratio:

$$
\Lambda(x) = \frac{p(x|H_0)}{p(x|H_1)}
$$

Where $p(x|H_0)$ and $p(x|H_1)$ are the probability density functions of the data $x$ under the null hypothesis $H_0$ and the alternative hypothesis $H_1$ respectively.

The optimal detection strategy (Neyman-Pearson lemma) consists of comparing this ratio to a threshold.
""")

lr_tabs = st.tabs(["Gaussian Case", "Laplacian Case", "Cauchy Case"])

with lr_tabs[0]:
    st.markdown(r"""    
    If we consider that the data follow a Gaussian distribution:
    $$
    p(x;H_i) = \frac{1}{\sqrt{2\pi}\sigma_i} \exp\left(-\frac{(x - \mu_i)^2}{2\sigma_i^2}\right)
    $$
    The log-likelihood ratio (ignoring constants) becomes:
    $$
    \log(\Lambda(\vec{x})) \propto \sum_{i=1}^{n} \frac{(x_i - \mu_1)^2}{2\sigma_1^2} - \frac{(x_i - \mu_0)^2}{2\sigma_0^2}
    $$
    This corresponds to the standard $\chi^2$ detection if variances are known.
    """)

with lr_tabs[1]:
    st.markdown(r"""
    If data follow a Laplacian distribution:
    $$
    p(x;H_i) = \frac{1}{2b_i} \exp\left(-\frac{|x - \mu_i|}{b_i}\right)
    $$
    Log-likelihood:
    $$
    \log(\Lambda(\vec{x})) \propto \sum_{i=1}^{n} \frac{|x_i - \mu_1|}{b_1} - \frac{|x_i - \mu_0|}{b_0}
    $$
    This is often more robust against outliers than the Gaussian assumption (L1 norm vs L2 norm).
    """)

with lr_tabs[2]:
    st.markdown(r"""
    For a Cauchy distribution (heavy tails):
    $$
    p(x;H_i) = \frac{1}{\pi \gamma_i \left[1 + \left(\frac{x - x_i}{\gamma_i}\right)^2\right]}
    $$
    Log-likelihood:
    $$
    \log(\Lambda(\vec{x})) \propto \sum_{i=1}^{n} \log\left(\frac{1+(\frac{x_i - x_1}{\gamma_1})^2}{1+(\frac{x_i - x_0}{\gamma_0})^2}\right)
    $$
    This is useful for very noisy data with frequent significant outliers.
    """)

st.divider()

# --- Simulation ---
st.header("Simulation & Analysis")

# --- Context ---
ctx = context_widget(
    key_prefix="stats",
    presets={
        "Life (Nulling)": Context.get_LIFE(),
        "VLTI": Context.get_VLTI()
    },
    default_preset="VLTI",
    expanded=False,
    show_advanced=True
)

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("MC settings")
    nmc = st.number_input("Number of samples (NMC)", min_value=10, max_value=2000, value=100, step=10)
    size = st.number_input("Data points per sample", min_value=10, max_value=2000, value=100, step=50)
    
with col2:
    st.subheader("Signal settings")
    contrast = st.number_input("Planet Contrast", min_value=1e-9, max_value=1e-1, value=1e-2, format="%.1e")
    piston_rms = st.number_input("Atmos. Piston RMS (nm)", min_value=0.0, max_value=1000.0, value=100.0, step=0.5)
    
with col3:
    st.subheader("Planet position")
    fov_val = ctx.interferometer.fov.to(u.mas).value
    separation = st.number_input("Separation (mas)", min_value=0.0, max_value=fov_val/2, value=min(2.0, fov_val/2), step=0.1)
    angle = st.number_input("Position Angle (deg)", min_value=0.0, max_value=360.0, value=0.0, step=1.0)
    
run_btn = st.button("Run Analysis", type="primary")

if run_btn:
    
    # Update Context with User Settings
    if len(ctx.target.companions) > 0:
        comp = ctx.target.companions[0]
        comp.c = contrast
        comp.ρ = separation * u.mas
        comp.θ = angle * u.deg
    
    # Set Gamma (Atmospheric Piston RMS)
    ctx.Γ = piston_rms * u.nm 

    # 1. Generate Vectors
    with st.spinner(f"Simulating {nmc} samples with {size} points..."):
        p_bar = st.progress(0, text="Generating vectors...")
        # Get separated kernels (flatten=False) -> shapes (3, nmc, size)
        t0_all, t1_all = ts.get_vectors(ctx=ctx, nmc=nmc, size=size, progress_callback=lambda x: p_bar.progress(x, text=f"Simulating... {int(x*100)}%"), flatten=False, randomize_position=False)
        p_bar.empty()
        st.success("Data generated successfully.")

    # 2. Transmission Maps Generation
    with st.spinner("Computing transmission maps..."):
        resol = 100 
        fov = ctx.interferometer.fov.to(u.mas).value
        # Unpacking explicitly: (Raw, Processed/Kernels)
        _, maps = ctx.get_transmission_maps(N=resol)
        
        # Calculate Planet Position for overlay
        if ctx.target.companions:
            comp = ctx.target.companions[0]
            radius = comp.ρ.to(u.mas).value
            # Angle: usually Defined East of North? Or standard polar?
            # In Context/Simulation, it's typically polar coordinates.
            # Assuming standard projection:
            angle = comp.θ.to(u.rad).value
            x_p = radius * np.cos(angle)
            y_p = radius * np.sin(angle)
        else:
            x_p, y_p = None, None

    # Define number of kernels for loops
    nb_kernels = t0_all.shape[0]

    # 3. Tabs for analysis
    tab_names = ["Global Comparison"] + list(ALL_TESTS.keys())
    tabs = st.tabs(tab_names)

    # Helper function for ROC
    def compute_roc(t0: np.ndarray, t1: np.ndarray, test: callable):
        
        # For 2-sample tests (KS, CvM), we need a reference H0 distribution.
        # For H1 stats: compare t1[i] vs t0[i]
        vals_t1 = np.array([test(t1[i], t0[i]) for i in range(t1.shape[0])])
        
        # For H0 stats: compare t0[i] vs another H0 sample (e.g. t0[i-1]) to estimate null distribution of the statistic
        # We roll t0 to get a different sample for comparison
        t0_ref = np.roll(t0, shift=1, axis=0)
        vals_t0 = np.array([test(t0[i], t0_ref[i]) for i in range(t0.shape[0])])
        
        all_stats = np.concatenate([vals_t0, vals_t1])
        
        if len(all_stats) == 0 or np.min(all_stats) == np.max(all_stats):
                return (np.array([0,1]), np.array([0,1]), 0) # Degenerate case

        thresholds = np.linspace(np.min(all_stats), np.max(all_stats), 200)
        pdet = []
        pfa = []
        for thresh in thresholds:
            tp = np.sum(vals_t1 > thresh) # H1 > thresh
            fn = np.sum(vals_t1 <= thresh)
            fp = np.sum(vals_t0 > thresh) # H0 > thresh
            tn = np.sum(vals_t0 <= thresh)
            pdet.append(tp / (tp + fn) if (tp+fn)>0 else 0)
            pfa.append(fp / (fp + tn) if (fp+tn)>0 else 0)
        
        # Power calculation
        power = 0
        if len(pfa) > 1:
             pfa_arr = np.array(pfa)
             pdet_arr = np.array(pdet)
             sorted_indices = np.argsort(pfa_arr)
             power = np.round(np.abs(np.trapz(pdet_arr[sorted_indices] - pfa_arr[sorted_indices], pfa_arr[sorted_indices])) * 200, 2)

        return (np.array(pfa), np.array(pdet), power)
    
    # --- Global Comparison ---
    with tabs[0]:
        st.subheader("Global Sensitivity (Broken down by Kernel)")
        st.info("Here we compare the performance of ALL test statistics for EACH kernel individually.")
        
        cols = st.columns(3)
        
        for k in range(3):
            with cols[k]:
                st.markdown(f"### Kernel {k+1}")
                
                # 1. ROC Plot
                fig_k, ax_k = plt.subplots(figsize=(5, 4))
                t0_k = t0_all[k] 
                t1_k = t1_all[k]
                for name, test in ALL_TESTS.items():
                    pfa, pdet, power = compute_roc(t0_k, t1_k, test)
                    ax_k.plot(pfa, pdet, label=f'{name} ({power}%)')
                ax_k.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
                ax_k.set_xlabel('False Positive Rate')
                ax_k.set_ylabel('True Positive Rate')
                ax_k.set_title(f"ROC - Kernel {k+1}")
                ax_k.legend(fontsize='small')
                ax_k.grid(True, alpha=0.3)
                st.pyplot(fig_k)
                
                # 2. Transmission Map
                st.markdown("**Transmission Map**")
                fig_map, ax_map = plt.subplots(figsize=(4, 4))
                if k < maps.shape[0]:
                    # Use 'bwr' (Blue-White-Red) as requested
                    im = ax_map.imshow(maps[k], extent=[-fov/2, fov/2, -fov/2, fov/2], origin='lower', cmap='bwr')
                    # Overlay Star
                    ax_map.scatter(0, 0, marker='*', s=150, color='yellow', edgecolors='black', label="Star", zorder=10)
                    # Overlay Planet if present
                    if x_p is not None:
                        ax_map.plot(x_p, y_p, 'o', color='lime', markersize=8, label="Planet", markeredgecolor='k', zorder=11)
                    
                    ax_map.axis('off')
                st.pyplot(fig_map)

                # 3. Distributions
                st.markdown("**Distributions ($\mathcal{H}_0$ vs $\mathcal{H}_1$)**")
                fig_dist, ax_dist = plt.subplots(figsize=(5, 3))
                # Flatten to show global distribution of samples
                ax_dist.hist(t0_k.flatten(), bins=50, density=True, histtype='step', linestyle='--', linewidth=1.5, label=r'$\mathcal{H}_0$ (Noise)', color='C0')
                ax_dist.hist(t1_k.flatten(), bins=50, density=True, alpha=0.5, label=r'$\mathcal{H}_1$ (Signal)', color='C1')
                ax_dist.set_xlabel('Kernel Output Value')
                ax_dist.set_ylabel('PDF')
                ax_dist.legend(fontsize='small')
                ax_dist.grid(True, alpha=0.3)
                st.pyplot(fig_dist)

    # --- Individual Tabs ---
    for i, (name, test) in enumerate(ALL_TESTS.items()):
        with tabs[i+1]:
            st.subheader(f"Analysis: {name}")
            
            col_roc, col_maps = st.columns([1, 1])

            with col_roc:
                st.markdown("**ROC Curves per Kernel**")
                fig_single, ax_single = plt.subplots(figsize=(5, 4))
                
                colors = ['r', 'g', 'b', 'c', 'm', 'y']
                for k in range(nb_kernels):
                    if k < len(colors): col = colors[k] 
                    else: col = 'k'
                    
                    t0_k = t0_all[k] 
                    t1_k = t1_all[k]
                    
                    pfa, pdet, power = compute_roc(t0_k, t1_k, test)
                    ax_single.plot(pfa, pdet, color=col, label=f'Kernel {k+1} (Power: {power}%)')

                ax_single.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
                ax_single.set_xlabel('False Positive Rate')
                ax_single.set_ylabel('True Positive Rate')
                ax_single.grid(True, alpha=0.3)
                ax_single.legend()
                st.pyplot(fig_single)

            with col_maps:
                # Row 1: Transmission Maps
                st.markdown("**Transmission Maps**")
                cols_m = st.columns(nb_kernels)
                
                for k in range(nb_kernels):
                    with cols_m[k]:
                        fig_map, ax = plt.subplots(figsize=(3, 3))
                        if k < maps.shape[0]:
                            im = ax.imshow(maps[k], extent=[-fov/2, fov/2, -fov/2, fov/2], origin='lower', cmap='bwr')
                            ax.scatter(0, 0, marker='*', s=100, color='yellow', edgecolors='black', label="Star", zorder=10)
                            if x_p is not None:
                                ax.plot(x_p, y_p, 'o', color='lime', markersize=6, label="Planet", markeredgecolor='k', zorder=11)
                            ax.axis('off')
                            ax.set_title(f"Kernel {k+1}", fontsize='small')
                        st.pyplot(fig_map)

                # Row 2: Distributions
                st.markdown("**Distributions**")
                cols_d = st.columns(nb_kernels)
                
                for k in range(nb_kernels):
                    with cols_d[k]:
                        fig_dist, ax_dist = plt.subplots(figsize=(3, 2))
                        ax_dist.hist(t0_all[k].flatten(), bins=50, density=True, histtype='step', linestyle='--', linewidth=1.2, label=r'$\mathcal{H}_0$', color='C0')
                        ax_dist.hist(t1_all[k].flatten(), bins=50, density=True, alpha=0.5, label=r'$\mathcal{H}_1$', color='C1')
                        if k == 0:
                            ax_dist.legend(fontsize='x-small')
                        ax_dist.set_title(f"Kernel {k+1}", fontsize='small')
                        ax_dist.tick_params(axis='both', which='major', labelsize=6)
                        plt.tight_layout()
                        st.pyplot(fig_dist)
                
                st.caption("Top: $\eta$ Maps. Bottom: H0/H1 Distributions. Separability correlates with Map intensity at planet location.")