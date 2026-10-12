
import numpy as np
import matplotlib.pyplot as plt
import phise
import astropy.units as u
from copy import deepcopy as copy

ctx = phise.examples.contexts.get_VLTI()
ctx.chip.σ *= 0
ctx.target.companions[0].c = 1e-6
ctx.Γ = 1 * u.nm

STATS = {
    'Median': np.median,
    'Mean': np.mean,
    # "Argmax500": phise.modules.test_statistics.argmax100,
}

def plot_stats_dispersion(ctx, samples, dropout, stats):

    # Data generation (one serie per stat)
    flux_data = np.empty((len(stats), samples))
    for s in range(len(stats.keys())):
        if s > 0 and True:
            flux_data[s] = flux_data[0]
        for i in range(samples):
            outs = ctx.observe()
            k1 = (outs[1] - outs[2]) / outs[0]
            flux_data[s, i] = k1



    # Get true value
    ctx_ideal = copy(ctx)
    ctx_ideal.Γ *= 0
    outs_ideal = ctx_ideal.observe()
    k1_ideal = (outs_ideal[1] - outs_ideal[2]) / outs_ideal[0]

    # Bootstrap statistics with sub-series
    nmc = 1000 # number of realizations of the test statistics
    stats_data = np.zeros((len(stats), nmc))
    for s, (name, stat_func) in enumerate(stats.items()):
        for i in range(nmc):
            # new_flux_data = np.random.choice(flux_data[s], size=int(samples*(1-dropout)), replace=False)
            new_flux_data = np.random.choice(flux_data[s], size=samples, replace=True)
            stats_data[s, i] = stat_func(new_flux_data)

    # Plotting
    colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink']

    stats_mean = np.mean(stats_data, axis=1)
    stats_std = np.std(stats_data, axis=1)

    # Getting prior
    prior_idx = np.argmin(stats_std)
    prior_std = stats_std[prior_idx]
    prior_mean = stats_mean[prior_idx]
    prior_name = list(stats.keys())[prior_idx]
    print(f"Prior:", list(stats.keys())[prior_idx])

    # Coupling the statistics using a Kalman filter approach
    for j, name in enumerate(stats.keys()):
        if name == prior_name:
            continue

        kalman_gain = prior_std**2 / (stats_std[j]**2 + prior_std**2)
        print("---")
        print(f"Kalman gain for {name}: {kalman_gain}")
        print(f"Difference: {stats_mean[j] - prior_mean}")
        print(f"Prior mean:   {prior_mean}, Prior std: {prior_std}")
        prior_mean = prior_mean + kalman_gain * (stats_mean[j] - prior_mean)
        prior_std = 1/(np.sqrt(1/prior_std**2 + 1/stats_std[j]**2))
        print(f"Updated mean: {prior_mean}, Updated std: {prior_std}")
        print("===")
        
    post_mean = prior_mean
    post_std = prior_std

    print(stats_mean)
    print(post_mean)
    print(stats_std)
    print(post_std)

    min_stat = np.min(stats_data)
    max_stat = np.max(stats_data)

    x = np.linspace(min_stat, max_stat, 1000)

    def gaussian(x, mu, sigma):
        return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma)**2)

    for j, (name, _) in enumerate(stats.items()):
        plt.hist(stats_data[j], bins=30, alpha=0.5, density=True, color=colors[j % len(colors)])
        # Gauss of the stats_data
        mu, std = stats_mean[j], stats_std[j]
        plt.plot(x, gaussian(x, mu, std), label=f'{name}', color=colors[j % len(colors)])

    # Plotting the coupled statistic
    x = np.linspace(post_mean - 4*post_std, post_mean + 4*post_std, 100)
    plt.plot(x, gaussian(x, post_mean, post_std), label='Kalman filter', color='red', linewidth=2)

    plt.axvline(k1_ideal, color='black', linestyle='dashed', linewidth=2, label='True value')

    plt.xlabel('Flux Difference (k1)')
    plt.ylabel('Density')
    plt.title(f'Kernel Combined Statistics (Dropout={dropout})')
    plt.legend()

    print(ctx)

if __name__ == "__main__":
    samples = 10_000
    dropout = 0.2
    plot_stats_dispersion(ctx, samples, dropout, STATS)
    plt.show()