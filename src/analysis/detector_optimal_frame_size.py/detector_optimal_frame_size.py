"""Simple script for optimal Gaussian spot capture window analysis.

This script simulates a Gaussian spot on a pixel grid with additive background noise.
It computes the measured flux inside both square and circular apertures, with window size
expressed in units of sigma. The total spot flux is derived from a target signal-to-noise
ratio (SNR), where SNR = total Gaussian flux / noise RMS over the aperture.

The flux values are therefore expressed in photon-equivalent units or normalized
detector counts, and the SNR refers to the ratio between the total spot signal and
the theoretical noise fluctuation expected in the window.
"""

import numpy as np
import matplotlib.pyplot as plt
import pltedit


def gaussian_spot(image_shape, center, sigma, total_flux):
    """Create a normalized Gaussian spot image with a given total flux."""
    y = np.arange(image_shape[0])
    x = np.arange(image_shape[1])
    xx, yy = np.meshgrid(x, y)
    dx = xx - center[1]
    dy = yy - center[0]
    exponent = -(dx**2 + dy**2) / (2 * sigma**2)
    image = np.exp(exponent)
    image *= total_flux / image.sum()
    return image


def square_window_mask(image_shape, center, half_width):
    """Return a boolean mask for pixels inside a square aperture centered on the spot."""
    y = np.arange(image_shape[0])
    x = np.arange(image_shape[1])
    xx, yy = np.meshgrid(x, y)
    return (
        np.abs(xx - center[1]) <= half_width
    ) & (
        np.abs(yy - center[0]) <= half_width
    )


def circular_window_mask(image_shape, center, radius):
    """Return a boolean mask for pixels inside a circular aperture."""
    y = np.arange(image_shape[0])
    x = np.arange(image_shape[1])
    xx, yy = np.meshgrid(x, y)
    distance = np.sqrt((xx - center[1])**2 + (yy - center[0])**2)
    return distance <= radius


def simulate_capture_response(
    sigma=2.0,
    snr_levels=None,
    noise_std=3.0,
    window_size_sigma=None,
    n_realizations=300,
    image_size=81,
):
    """Simulate measured flux fraction for varying window sizes and target SNR values."""
    if snr_levels is None:
        snr_levels = [2.0, 4.0, 8.0]
    if window_size_sigma is None:
        window_size_sigma = np.linspace(0.5, 15.0, 100)

    center = (image_size // 2, image_size // 2)
    results = {}

    # distance to place neighbors: 7 sigma along x-axis
    neighbor_offset_sigma = 7.0

    for snr in snr_levels:
        square_means = []
        square_stds = []
        circle_means = []
        circle_stds = []
        square_central_true = []
        circle_central_true = []
        square_neighbor_frac = []
        circle_neighbor_frac = []

        for window_sigma in window_size_sigma:
            half_width = window_sigma * sigma / 2.0
            circle_radius = window_sigma * sigma / 2.0

            square_mask = square_window_mask((image_size, image_size), center, half_width)
            circle_mask = circular_window_mask((image_size, image_size), center, circle_radius)

            n_square = square_mask.sum()
            n_circle = circle_mask.sum()

            # Compute total flux for central spot from target SNR and noise RMS of aperture
            central_flux = snr * noise_std * np.sqrt(n_square)  # use square aperture area for flux calc

            # Build images: central + two neighbors at ±7 sigma along x
            central_image = gaussian_spot((image_size, image_size), center, sigma, central_flux)
            off_x = int(round(neighbor_offset_sigma * sigma))
            left_center = (center[0], center[1] - off_x)
            right_center = (center[0], center[1] + off_x)
            neighbor_left = gaussian_spot((image_size, image_size), left_center, sigma, central_flux)
            neighbor_right = gaussian_spot((image_size, image_size), right_center, sigma, central_flux)

            total_image = central_image + neighbor_left + neighbor_right

            # True fractions (no noise) inside apertures, normalized to central flux
            central_in_square = central_image[square_mask].sum() / central_flux
            neighbors_in_square = (neighbor_left + neighbor_right)[square_mask].sum() / central_flux
            central_in_circle = central_image[circle_mask].sum() / central_flux
            neighbors_in_circle = (neighbor_left + neighbor_right)[circle_mask].sum() / central_flux

            # Monte Carlo measurements (noise added) normalized to central_flux
            square_measurements = []
            circle_measurements = []
            for _ in range(n_realizations):
                noise = np.random.normal(loc=0.0, scale=noise_std, size=total_image.shape)
                meas_square = (total_image + noise)[square_mask].sum() / central_flux
                meas_circle = (total_image + noise)[circle_mask].sum() / central_flux
                square_measurements.append(meas_square)
                circle_measurements.append(meas_circle)

            square_measurements = np.array(square_measurements)
            circle_measurements = np.array(circle_measurements)

            square_means.append(square_measurements.mean())
            square_stds.append(square_measurements.std(ddof=1))
            circle_means.append(circle_measurements.mean())
            circle_stds.append(circle_measurements.std(ddof=1))

            square_central_true.append(central_in_square)
            circle_central_true.append(central_in_circle)
            square_neighbor_frac.append(neighbors_in_square)
            circle_neighbor_frac.append(neighbors_in_circle)

        results[snr] = {
            "window_sigma": window_size_sigma,
            "square_mean": np.array(square_means),
            "square_std": np.array(square_stds),
            "circle_mean": np.array(circle_means),
            "circle_std": np.array(circle_stds),
            "square_central_true": np.array(square_central_true),
            "circle_central_true": np.array(circle_central_true),
            "square_neighbor_frac": np.array(square_neighbor_frac),
            "circle_neighbor_frac": np.array(circle_neighbor_frac),
        }

    return results


def plot_response(results):
    """Plot the measured flux response for square and circular windows."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for snr, values in results.items():
        window_sigma = values["window_sigma"]
        line_color = f'C{int(snr)}'

        axes[0].plot(window_sigma, values["square_central_true"], linestyle='--', color=line_color, alpha=0.9)
        axes[0].plot(window_sigma, values["square_mean"], label=f"SNR {snr:.1f}", color=line_color)
        axes[0].fill_between(
            window_sigma,
            values["square_mean"] - values["square_std"],
            values["square_mean"] + values["square_std"],
            alpha=0.18,
            color=line_color,
            linewidth=0,
        )
        contamination_label = 'Neighbor contamination' if snr == list(results.keys())[0] else None
        axes[0].plot(window_sigma, values["square_neighbor_frac"], linestyle=':', color='k', label=contamination_label)

        axes[1].plot(window_sigma, values["circle_central_true"], linestyle='--', color=line_color, alpha=0.9)
        axes[1].plot(window_sigma, values["circle_mean"], label=f"SNR {snr:.1f}", color=line_color)
        axes[1].fill_between(
            window_sigma,
            values["circle_mean"] - values["circle_std"],
            values["circle_mean"] + values["circle_std"],
            alpha=0.18,
            color=line_color,
            linewidth=0,
        )
        axes[1].plot(window_sigma, values["circle_neighbor_frac"], linestyle=':', color='k', label=contamination_label)

    axes[0].set_title("Square aperture")
    axes[0].set_xlabel("Window size (in sigma)")
    axes[0].set_ylabel("Normalized flux (central flux reference)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(title="Target SNR")

    axes[1].set_title("Circular aperture")
    axes[1].set_xlabel("Window size (in sigma)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(title="Target SNR")

    fig.suptitle("Square vs circular comparison — central capture and neighbor contamination")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pltedit.save(fig, "detector_optimal_frame_size.plt")
    plt.show()


if __name__ == "__main__":
    results = simulate_capture_response()
    plot_response(results)
