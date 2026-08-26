import numpy as np
import matplotlib.pyplot as plt

def gauss(x, μ, σ):
    """Compute the Gaussian function."""
    return np.exp(-0.5 * ((x - μ) / σ) ** 2) / (σ * np.sqrt(2 * np.pi)) 

def plot_gauss(x, μ, σ):
    """Plot the Gaussian function."""
    y = gauss(x, μ, σ)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('Gaussian')
    plt.title(f'Gaussian with μ={μ}, σ={σ}')
    plt.show()

if __name__ == "__main__":
    x = np.linspace(-10, 10, 400)
    μ = 0
    σ = 2.14
    plot_gauss(x, μ, σ) 