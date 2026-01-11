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
st.markdown(r"""
Statistical analysis is crucial for determining the presence of a planet signal buried in noise.

The **Likelihood Ratio** is theoretically the optimal test (Neyman-Pearson lemma). However, in practice, it is often **inapplicable** because it requires knowing the exact analytical form of the data distribution (which is often unknown or changing).

Therefore, we study other **test statistics** that are robust and do not require prior knowledge of the distribution.
""")

# --- Data Ansatz ---
st.subheader("Data Description (Ansatz)")
st.markdown(r"""
Before defining our test statistics, we model our data $x$ (a vector of $N$ points, representing the null depth or kernel output) under the two hypotheses:

- **$H_0$ (Null Hypothesis)**: The data consists only of noise $n$.
  $$ x = n $$
- **$H_1$ (Alternative Hypothesis)**: The data contains a planetary signal $S_p$ scaled by an intensity $\alpha$, added to the noise.
  $$ x = n + \alpha S_p $$

Our goal is to extract a test statistic $T(x)$ that maximizes the separability between these two cases.
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
    H_0 : d = |\bar{n}|\\
    H_1 : d =  |\alpha \hat{S_p} + \bar{n}|
    \end{cases}
    $$
    Where $\bar{n}$ is the mean noise, and $\hat{S_p}$ is the estimated planet signal.
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
    H_0 : d = |\tilde{n}|\\
    H_1 : d =  | \alpha \hat{S_p} + \tilde{n} |
    \end{cases}
    $$
    Where $\tilde{n}$ is the median noise.
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
    H_0 : d = \sum |n - \tilde{n}|\\
    H_1 : d = \sum \frac{|n - \tilde{n}|}{\beta(\hat{S_p})}
    \end{cases}
    $$
    """)

with ts_tabs[5]:
    st.markdown(r"""
    **Principle:** Robust estimator of scale.
    
    **Ansatz:**
    $$
    \begin{cases}
    H_0 : d = \tilde{\text{abs}(n)}\\
    H_1 : d = \alpha \times \hat{\text{abs}(S_p)} + \frac{\tilde{\text{abs}(n)}}{\beta(\hat{S_p})}
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
    default_preset="Life (Nulling)",
    expanded=False,
    show_advanced=True
)

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Settings")
    nmc = st.number_input("Number of samples (NMC)", min_value=10, max_value=1000, value=100, step=10)
    size = st.number_input("Data points per sample", min_value=10, max_value=2000, value=100, step=50)
    
    run_btn = st.button("Run Analysis", type="primary")

if run_btn:
    
    # 1. Generate Vectors
    with st.spinner(f"Simulating {nmc} samples with {size} points..."):
        p_bar = st.progress(0, text="Generating vectors...")
        t0, t1 = ts.get_vectors(ctx=ctx, nmc=nmc, size=size, progress_callback=lambda x: p_bar.progress(x, text=f"Generating vectors... {int(x*100)}%"))
        p_bar.empty()
        st.success("Data generated successfully.")

    # 2. Tabs for analysis
    tab_names = ["Global Comparison"] + list(ALL_TESTS.keys())
    tabs = st.tabs(tab_names)

    # --- Global Comparison ---
    with tabs[0]:
        st.subheader("Global ROC Comparison")
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        
        # Helper function for ROC
        def compute_roc(t0: np.ndarray, t1: np.ndarray, test: callable):
            t0_stats = np.array([test(t0[i], t0[i + 1]) if i + 1 < t0.shape[0] else test(t0[i], t0[0]) for i in range(t0.shape[0])])
            t1_stats = np.array([test(t1[i], t0[i]) for i in range(t1.shape[0])])
            all_stats = np.concatenate([t0_stats, t1_stats])
            
            if len(all_stats) == 0 or np.min(all_stats) == np.max(all_stats):
                    return (np.array([0,1]), np.array([0,1]), np.array([0,1]), 0) # Degenerate case

            thresholds = np.linspace(np.min(all_stats), np.max(all_stats), 200) # reduced resolution for speed
            pdet = []
            pfa = []
            for thresh in thresholds:
                tp = np.sum(t1_stats > thresh)
                fn = np.sum(t1_stats <= thresh)
                fp = np.sum(t0_stats > thresh)
                tn = np.sum(t0_stats <= thresh)
                pdet.append(tp / (tp + fn) if (tp+fn)>0 else 0)
                pfa.append(fp / (fp + tn) if (fp+tn)>0 else 0)
            
            # Calculate Power (AUC-like metric as per original code)
            power = 0
            if len(pfa) > 1:
                 pfa_arr = np.array(pfa)
                 pdet_arr = np.array(pdet)
                 # Sort by PFA to ensure correct integration
                 sorted_indices = np.argsort(pfa_arr)
                 pfa_sorted = pfa_arr[sorted_indices]
                 pdet_sorted = pdet_arr[sorted_indices]
                 
                 power = np.round(np.abs(np.trapz(pdet_sorted - pfa_sorted, pfa_sorted)) * 200, 2)

            return (np.array(pfa), np.array(pdet), thresholds, power)

        results = {}

        for name, test in ALL_TESTS.items():
            pfa, pdet_vals, _, power = compute_roc(t0, t1, test)
            results[name] = {"pfa": pfa, "pdet": pdet_vals, "power": power}
            ax_roc.plot(pfa, pdet_vals, label=f'{name} (Power: {power}%)')

        ax_roc.plot([0, 1], [0, 1], 'k--', label='Random')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve')
        ax_roc.legend()
        ax_roc.grid(True, alpha=0.3)
        st.pyplot(fig_roc)

    # --- Individual Tabs ---
    for i, (name, test) in enumerate(ALL_TESTS.items()):
        with tabs[i+1]:
            st.subheader(f"Analysis: {name}")
            
            col_res1, col_res2 = st.columns(2)
            
            # ROC for this specific test
            with col_res1:
                st.markdown(f"**ROC Curve** (Power: {results[name]['power']}%)")
                fig_single, ax_single = plt.subplots(figsize=(5, 4))
                ax_single.plot(results[name]["pfa"], results[name]["pdet"], label=name)
                ax_single.plot([0, 1], [0, 1], 'k--', label='Random')
                ax_single.set_xlabel('False Positive Rate')
                ax_single.set_ylabel('True Positive Rate')
                ax_single.grid(True, alpha=0.3)
                ax_single.legend()
                st.pyplot(fig_single)

            # P-Values for this specific test
            with col_res2:
                st.markdown("**P-Value Distribution**")
                
                # Compute p-values logic (re-implemented for single test)
                values = []
                for u_vec, v_vec in zip(t1, t0):
                    values.append(test(u_vec, v_vec))
                
                if len(values) > 0:
                    sup = np.max(values)
                    thresholds_p = np.linspace(0, sup, 200)
                    p_values = np.zeros(len(thresholds_p))
                    for idx, threshold in enumerate(thresholds_p):
                        p_values[idx] = np.sum(np.array(values) > threshold) / len(values)
                    
                    fig_p, ax_p = plt.subplots(figsize=(5, 4))
                    ax_p.plot(thresholds_p, p_values, label=f"P-values")
                    ax_p.hlines(0.05, 0, sup, color='red', linestyle='dashed', label='0.05 Threshold')
                    ax_p.set_xlabel('Test statistic value')
                    ax_p.set_ylabel('P-value')
                    ax_p.grid(True, alpha=0.3)
                    ax_p.legend()
                    st.pyplot(fig_p)

# --- Discussion ---
st.divider()
st.header("Discussion")
st.markdown("""
- **Performance**: Tests with higher power (bowing further to top-left) are better at distinguishing planets from noise.
- **Robustness**: Different tests (Gaussian, Laplacian, etc.) assume different noise distributions. Using the matched likelihood ratio is optimal if the distribution is known.
- **P-Value**: Used to set detection thresholds. Typically a P-value < 0.05 (or much lower for astronomy, e.g. 3-5 sigma) is required for detection.
""")

# --- Footer ---
st.divider()
with st.expander("Technical Technical Details"):
    st.info("Simulation powered by `phise.modules.test_statistics`.")
    st.info("Uses Monte Carlo simulations to estimate PDF and ROCs based on the current Context.")
