import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.special
import os

# Output directory
OUTPUT_DIR = r"../assets/img/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- IMB ---
def imb_pdf(z, mu, sigma, nu):
    if sigma <= 0 or nu <= 0: return np.zeros_like(z)
    v = (nu - 1)/2
    a = 2**((1-nu)/2) * np.sqrt(nu)
    b = sigma * np.sqrt(np.pi) * scipy.special.gamma(nu/2)
    c = np.abs((z-mu) / (sigma * np.sqrt(nu)))
    k_val = scipy.special.kv(v, c)
    pdf = (a / b) * c**v * k_val
    return np.nan_to_num(pdf, nan=0.0)

x = np.linspace(-5, 5, 1000)
plt.figure(figsize=(6, 4))
plt.plot(x, imb_pdf(x, 0, 1, 0.5), label=r'$\nu=0.5$')
plt.plot(x, imb_pdf(x, 0, 1, 1.5), label=r'$\nu=1.5$')
plt.plot(x, imb_pdf(x, 0, 1, 5.0), label=r'$\nu=5.0$')
plt.title("IMB Distribution (varying $\\nu$)")
plt.xlabel("Intensity")
plt.ylabel("PDF")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "imb_plot.png"))
plt.close()

# --- Cauchy ---
x = np.linspace(-10, 10, 1000)
plt.figure(figsize=(6, 4))
plt.plot(x, stats.cauchy.pdf(x, loc=0, scale=0.5), label=r'$\gamma=0.5$')
plt.plot(x, stats.cauchy.pdf(x, loc=0, scale=1.0), label=r'$\gamma=1.0$')
plt.plot(x, stats.cauchy.pdf(x, loc=0, scale=2.0), label=r'$\gamma=2.0$')
plt.title("Cauchy Distribution (varying scale $\\gamma$)")
plt.xlabel("x")
plt.ylabel("PDF")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cauchy_plot.png"))
plt.close()

# --- Laplace ---
x = np.linspace(-5, 5, 1000)
plt.figure(figsize=(6, 4))
plt.plot(x, stats.laplace.pdf(x, loc=0, scale=0.5), label=r'$b=0.5$')
plt.plot(x, stats.laplace.pdf(x, loc=0, scale=1.0), label=r'$b=1.0$')
plt.plot(x, stats.laplace.pdf(x, loc=0, scale=2.0), label=r'$b=2.0$')
plt.title("Laplace Distribution (varying scale $b$)")
plt.xlabel("x")
plt.ylabel("PDF")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "laplace_plot.png"))
plt.close()

# --- Beta ---
x = np.linspace(0, 1, 1000)
plt.figure(figsize=(6, 4))
plt.plot(x, stats.beta.pdf(x, 0.5, 0.5), label=r'$\alpha=0.5, \beta=0.5$')
plt.plot(x, stats.beta.pdf(x, 2, 2), label=r'$\alpha=2, \beta=2$')
plt.plot(x, stats.beta.pdf(x, 2, 5), label=r'$\alpha=2, \beta=5$')
plt.title("Beta Distribution")
plt.xlabel("x")
plt.ylabel("PDF")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "beta_plot.png"))
plt.close()

# --- Gamma ---
x = np.linspace(0, 10, 1000)
plt.figure(figsize=(6, 4))
plt.plot(x, stats.gamma.pdf(x, a=1, scale=1), label=r'$k=1$ (Exp)')
plt.plot(x, stats.gamma.pdf(x, a=2, scale=1), label=r'$k=2$')
plt.plot(x, stats.gamma.pdf(x, a=9, scale=1), label=r'$k=9$ (Normal-like)')
plt.title("Gamma Distribution (varying shape $k$)")
plt.xlabel("x")
plt.ylabel("PDF")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gamma_plot.png"))
plt.close()

print("Plots generated successfully.")
