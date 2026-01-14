import sys
from pathlib import Path
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Path Setup ---
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
WEB = ROOT / "web"
if str(WEB) not in sys.path:
    sys.path.insert(0, str(WEB))

# --- Page Config ---
st.set_page_config(
    page_title="Likelihood Ratio",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Likelihood Ratio (Theoretical Benchmark)")

st.header("Theory")

st.markdown("""
The **Likelihood Ratio** is theoretically the optimal test (Neyman-Pearson lemma). However, in practice, it is often **inapplicable** because it requires knowing the exact analytical form of the data distribution (which is often unknown or changing).
""")

st.markdown(r"""
If we **assume** a specific distribution, we can derive the optimal likelihood ratio test. We compare these theoretical optimums to our practical tests.

The core of the detection is the Likelihood Ratio:

$$
\Lambda(x) = \frac{p(x|H_0)}{p(x|H_1)}
$$

Where $p(x|H_0)$ and $p(x|H_1)$ are the probability density functions of the data $x$ under the null hypothesis $H_0$ and the alternative hypothesis $H_1$ respectively.

The optimal detection strategy (Neyman-Pearson lemma) consists of comparing this ratio to a threshold.
""")


st.success("""
##### ✨ My contribution

The **Likelihood Ratio** test had **never been applied** to the output distributions of a Kernel Nuller before this work.

This study demonstrates that the **median** is not only the best practical estimator among those considered, but it also performs **close to the theoretical optimum** defined by the Likelihood Ratio.
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
