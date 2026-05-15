import numpy as np
import matplotlib.pyplot as plt
try:
    plt.rcParams['image.origin'] = 'lower'
except Exception:
    pass
from copy import deepcopy as copy
import astropy.units as u
from scipy.optimize import minimize
from scipy import stats
from phise import Context
import phise
from phise.modules import utils
from phise.modules.test_statistics import ALL_TESTS
import phise.modules.test_statistics as ts_module
import scipy.special
from phise.modules import utils

def roc(t0: np.ndarray, t1: np.ndarray, test: callable):
    t0_stats = np.array([test(t0[i], t0[i + 1]) if i + 1 < t0.shape[0] else test(t0[i], t0[0]) for i in range(t0.shape[0])])
    t1_stats = np.array([test(t1[i], t0[i]) for i in range(t1.shape[0])])
    all_stats = np.concatenate([t0_stats, t1_stats])
    thresholds = np.linspace(np.min(all_stats), np.max(all_stats), 1000)
    pdet = []
    pfa = []
    for thresh in thresholds:
        tp = np.sum(t1_stats > thresh)
        fn = np.sum(t1_stats <= thresh)
        fp = np.sum(t0_stats > thresh)
        tn = np.sum(t0_stats <= thresh)
        pdet.append(tp / (tp + fn))
        pfa.append(fp / (fp + tn))
    return (np.array(pfa), np.array(pdet), thresholds)

def plot_rocs(t0: np.ndarray, t1: np.ndarray, tests: dict=ALL_TESTS, figsize=(6, 6), save_as=None):
    plt.figure(figsize=figsize, constrained_layout=True)
    for (name, test) in tests.items():
        (pfa, pdet, thresholds) = roc(t0, t1, test)
        plt.plot(pfa, pdet, label=f'{name}')
        power = np.round(np.abs(np.trapezoid(pdet - pfa, pfa)) * 200, 2)
        print(f'Power of {name}: {power}%')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.legend()
    if save_as:
        utils.save_plot(save_as, "roc_curves.png")
    plt.show()

def test_power(ctx=None, tests=ALL_TESTS, nmc=100, bootstrap=10, resolution=10, maxpoints=1000, save_as=None):

    # Build context
    if ctx is None:
        ctx = Context.get_VLTI()
        ctx.interferometer.chip.σ = np.zeros(14) * u.nm
        ctx.target.companions[0].c = 0.0001
    else:
        ctx = copy(ctx)

    
    nb_values = np.logspace(1, np.log10(maxpoints), resolution, endpoint=True).astype(int)
    auc_bootstrap = copy(tests)
    power_bootstrap = copy(tests)
    for (ts_name, test_fn) in tests.items():
        auc_bootstrap[ts_name] = []
        power_bootstrap[ts_name] = []
    for b in range(bootstrap):
        print('=' * 10, f'\nBootstrap {b + 1}/{bootstrap}...\n', '=' * 10, sep='')
        (t0, t1) = ts_module.get_vectors(ctx=ctx, nmc=nmc, size=maxpoints)
        dataset = []
        for nb in nb_values:
            indices = np.random.choice(t0.shape[1], nb, replace=False)
            t0_subset = t0[:, indices]
            t1_subset = t1[:, indices]
            dataset.append((t0_subset, t1_subset))
        print(f'Computing tests power...', end='\r')
        for (ts_name, test_fn) in tests.items():
            auc = np.zeros(len(dataset))
            power = np.zeros(len(dataset))
            for (i, (t0, t1)) in enumerate(dataset):
                (pfa, pdet, _) = roc(t0=t0, t1=t1, test=test_fn)
                (pfa, pdet) = zip(*sorted(zip(pfa, pdet)))
                pfa = np.array(pfa)
                pdet = np.array(pdet)
                auc[i] = np.round(np.abs(np.trapezoid(pdet - pfa, pfa)) * 200, 2)
                power[i] = np.round(pdet[np.argmax(pfa[pfa <= 0.01])] * 100, 2)
            auc_bootstrap[ts_name].append(auc)
            power_bootstrap[ts_name].append(power)
        print('Done computing tests power ✅')
    (_, axs) = plt.subplots(1, 2, figsize=(12, 6))
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(tests)))
    for (i, (ts_name, test_fn)) in enumerate(tests.items()):
        auc_mean = np.mean(auc_bootstrap[ts_name], axis=0)
        power_mean = np.mean(power_bootstrap[ts_name], axis=0)
        auc_std = np.std(auc_bootstrap[ts_name], axis=0)
        power_std = np.std(power_bootstrap[ts_name], axis=0)
        axs[0].plot(nb_values, auc_mean, label=ts_name, color=colors[i])
        axs[0].fill_between(nb_values, auc_mean - auc_std, auc_mean + auc_std, color=colors[i], alpha=0.2)
        axs[1].plot(nb_values, power_mean, label=ts_name, color=colors[i])
        axs[1].fill_between(nb_values, power_mean - power_std, power_mean + power_std, color=colors[i], alpha=0.2)
    for ax in axs:
        ax.set_xscale('log')
        ax.set_xlabel('Number of data points')
    axs[0].set_ylabel('Global power [%]')
    axs[1].set_ylabel('Power [%]')
    axs[0].set_title('AUC (global power)')
    axs[1].set_title('Power ($P_{det}$ at $P_{fa}<1\\%$)')
    plt.legend()
    if save_as:
        utils.save_plot(save_as, "test_power.png")

def plot_p_values(t0, t1, tests=ALL_TESTS, save_as=None):
    col = min(2, len(tests))
    row = int(np.ceil(len(tests) / col))
    (_, axs) = plt.subplots(row, col, figsize=(5 * col, 5 * row))
    axs = axs.flatten()
    for (t, (ts_name, ts)) in enumerate(tests.items()):
        sup = 0
        values = []
        for (u, v) in zip(t1, t0):
            values.append(ts(u, v))
        sup = max(sup, np.max(values))
        thresholds = np.linspace(0, sup, 1000)
        p_values = np.zeros(len(thresholds))
        for (i, threshold) in enumerate(thresholds):
            p_values[i] = np.sum(values > threshold) / len(values)
        axs[t].plot(thresholds, p_values, label=ts_name)
        axs[t].hlines(0.05, 0, sup, color='red', linestyle='dashed')
        axs[t].set_xlabel('Test value')
        axs[t].set_ylabel('P-value')
        axs[t].set_title(f'P-values for {ts_name}')
        axs[t].legend()
        axs[t].set_title(f'P-values for {ts_name}')
        axs[t].legend()
    if save_as:
        utils.save_plot(save_as, "p_values.png")
    plt.show()

#==============================================================================
# Benchmark
#==============================================================================

# Specific models definition ----------------------------------------------

def imb(z, μ, σ, ν):
    # Felix Dannert et al. 2025

    ν=0.8

    v = (ν - 1)/2
    a = 2**((1-ν)/2) * np.sqrt(ν)
    b = σ * np.sqrt(np.pi) * scipy.special.gamma(ν/2)
    c = np.abs((z-μ) / (σ * np.sqrt(ν)))
    k = scipy.special.kv(v, c)

    return (a / b) * c**v * k

def np_benchmark(ctx: Context=None, save_as=None):

    # Build H1 context
    if ctx is None:
        ctx = phise.examples.contexts.get_VLTI()
        ctx.interferometer.chip.σ = np.zeros(14) * u.nm
        ctx.target.companions[0].c = 0.001
        ctx.monochromatic = False
    else:
        ctx = copy(ctx)

    # Build H0 context
    ctx_star_only = copy(ctx)
    ctx_star_only.target.companions = []

    # Generate reference distribution using the numerical model ---------------

    print('⌛ Generating distributions...')
    samples = 10_000
    bins = np.sqrt(samples).astype(int)
    h0_data_kn = np.empty(samples)
    h1_data_kn = np.empty(samples)
    for i in range(samples):
        print(f'{(i + 1) / samples * 100:.2f}% ({i + 1}/{samples})', end='\r')
        outs_h0 = ctx_star_only.observe()
        ker_h0 = ctx_star_only.interferometer.chip.process_outputs(outs_h0)
        h0_data_kn[i] = ker_h0[0]
        outs_h1 = ctx.observe()
        ker_h1 = ctx.interferometer.chip.process_outputs(outs_h1)
        h1_data_kn[i] = ker_h1[0]
    print('✅ Distributions generated.')

    # Cost function for imb fit -----------------------------------------------

    def imb_cost(params, data):
        μ, σ, ν = params
        pdf_vals = imb(data, μ, σ, ν)
        pdf_vals /= np.trapezoid(pdf_vals, data)
        hist_vals, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        model_vals = imb(bin_centers, μ, σ, ν)
        model_vals /= np.trapezoid(model_vals, bin_centers)
        cost = np.sum((hist_vals - model_vals) ** 2)
        return cost

    # Fit distributions -------------------------------------------------------

    (x0, γ0) = stats.cauchy.fit(h0_data_kn)
    (x1, γ1) = stats.cauchy.fit(h1_data_kn)
    (μ0, b0) = stats.laplace.fit(h0_data_kn)
    (μ1, b1) = stats.laplace.fit(h1_data_kn)
    # (β0, m0, s0) = stats.gennorm.fit(h0_data_kn)
    # (β1, m1, s1) = stats.gennorm.fit(h1_data_kn)
    kde_h0 = stats.gaussian_kde(h0_data_kn)
    kde_h1 = stats.gaussian_kde(h1_data_kn)
    # imb fit for h0
    initial_guess_h0 = [np.median(h0_data_kn), np.std(h0_data_kn), 0.8]
    result_h0 = minimize(imb_cost, initial_guess_h0, args=(h0_data_kn,), bounds=[(None, None), (1e-5, None), (2.1, None)])
    μ_imb0, σ_imb0, ν_imb0 = result_h0.x
    # imb fit for h1
    initial_guess_h1 = [np.median(h1_data_kn), np.std(h1_data_kn), 0.8]
    result_h1 = minimize(imb_cost, initial_guess_h1, args=(h1_data_kn,), bounds=[(None, None), (1e-5, None), (2.1, None)])
    μ_imb1, σ_imb1, ν_imb1 = result_h1.x

    # Init plot
    x = np.linspace(min(np.min(h0_data_kn), np.min(h1_data_kn)), max(np.max(h0_data_kn), np.max(h1_data_kn)), 1000)
    plt.figure(figsize=(10, 6))

    # Reference distributions
    plt.hist(h0_data_kn, bins=bins, density=True, alpha=0.5, label='h0 data', color='blue', log=True)
    plt.hist(h1_data_kn, bins=bins, density=True, alpha=0.5, label='h1 data', color='orange', log=True)
    
    # Fitted distributions
    plt.plot(x, stats.cauchy.pdf(x, loc=x0, scale=γ0), 'b--', label='h0 cauchy fit', linewidth=2)
    plt.plot(x, stats.cauchy.pdf(x, loc=x1, scale=γ1), 'r--', label='h1 cauchy fit', linewidth=2)
    # plt.plot(x, stats.laplace.pdf(x, loc=μ0, scale=b0), 'b:', label='h0 laplace fit', linewidth=2)
    # plt.plot(x, stats.laplace.pdf(x, loc=μ1, scale=b1), 'r:', label='h1 laplace fit', linewidth=2)
    # plt.plot(x, stats.gennorm.pdf(x, β0, m0, s0), 'b-.', label='h0 gennorm fit', linewidth=2)
    # plt.plot(x, stats.gennorm.pdf(x, β1, m1, s1), 'r-.', label='h1 gennorm fit', linewidth=2)
    # plt.plot(x, 0.5 * stats.cauchy.pdf(x, loc=x0, scale=γ0) + 0.5 * stats.laplace.pdf(x, loc=μ0, scale=b0), 'b.', label='h0 mix fit', linewidth=2)
    # plt.plot(x, 0.5 * stats.cauchy.pdf(x, loc=x1, scale=γ1) + 0.5 * stats.laplace.pdf(x, loc=μ1, scale=b1), 'r.', label='h1 mix fit', linewidth=2)
    # plt.plot(x, kde_h0(x), 'b-', label='h0 KDE', linewidth=2)
    # plt.plot(x, kde_h1(x), 'r-', label='h1 KDE', linewidth=2)
    plt.plot(x, imb(x, μ_imb0, σ_imb0, ν_imb0), 'b-.', label='h0 IMB fit', linewidth=2)
    plt.plot(x, imb(x, μ_imb1, σ_imb1, ν_imb1), 'r-.', label='h1 IMB fit', linewidth=2)
    
    # Finalize plot
    plt.xlabel('Test Statistic Value')
    plt.ylabel('Density')
    plt.title('Distributions and Fits')
    plt.legend()
    if save_as:
        utils.save_plot(save_as, "distributions.png")
    plt.show()

    # Generate random distributions from the fitted models
    print('⌛ Generating random distributions from the fitted models...')
    nmc = 100
    samples = 1000
    t0_sim = np.empty((nmc, samples))
    t1_sim = np.empty((nmc, samples))
    t0_cauchy = np.empty((nmc, samples))
    t1_cauchy = np.empty((nmc, samples))
    t0_laplace = np.empty((nmc, samples))
    t1_laplace = np.empty((nmc, samples))
    t0_gennorm = np.empty((nmc, samples))
    t1_gennorm = np.empty((nmc, samples))
    t0_imb = np.empty((nmc, samples))
    t1_imb = np.empty((nmc, samples))
    for i in range(nmc):
        print(f'{(i + 1) / nmc * 100:.2f}% ({i + 1}/{nmc})', end='\r')
        t0_cauchy[i] = stats.cauchy.rvs(loc=x0, scale=γ0, size=samples)
        t1_cauchy[i] = stats.cauchy.rvs(loc=x1, scale=γ1, size=samples)
        t0_laplace[i] = stats.laplace.rvs(loc=μ0, scale=b0, size=samples)
        t1_laplace[i] = stats.laplace.rvs(loc=μ1, scale=b1, size=samples)
        # t0_gennorm[i] = stats.gennorm.rvs(beta=β0, loc=m0, scale=s0, size=samples)
        # t1_gennorm[i] = stats.gennorm.rvs(beta=β1, loc=m1, scale=s1, size=samples)
        for j in range(samples):
            outs0 = ctx_star_only.observe()
            k0 = ctx_star_only.interferometer.chip.process_outputs(outs0)
            t0_sim[i, j] = k0[0]
            outs1 = ctx.observe()
            k1 = ctx.interferometer.chip.process_outputs(outs1)
            t1_sim[i, j] = k1[0]

            # random in imb distribution
            def sample_imb_random(mu, sigma, nu, x_min=None, x_max=None, n_grid=2000):
                # Build sampling grid from empirical ranges if not provided
                if x_min is None:
                    x_min = min(np.min(h0_data_kn), np.min(h1_data_kn))
                if x_max is None:
                    x_max = max(np.max(h0_data_kn), np.max(h1_data_kn))
                x = np.linspace(x_min, x_max, n_grid)
                pdf = imb(x, mu, sigma, nu)
                pdf = np.clip(pdf, 0.0, None)
                dx = x[1] - x[0]
                cdf = np.cumsum(pdf) * dx
                if not np.isfinite(cdf[-1]) or cdf[-1] <= 0:
                    # fallback to normal jitter if CDF invalid
                    return np.random.normal(loc=mu, scale=max(sigma, 1e-9))
                cdf /= cdf[-1]
                u = np.random.rand()
                return float(np.interp(u, cdf, x))

            # draw one sample from each fitted IMB for h0 and h1
            t0_imb[i, j] = sample_imb_random(μ_imb0, σ_imb0, ν_imb0)
            t1_imb[i, j] = sample_imb_random(μ_imb1, σ_imb1, ν_imb1)
    print('✅ Random distributions generated.')
    print('⌛ Plotting ROC curves...')

    # Define likelihood ratio tests for fitted distributions
    def lr_cauchy(u, v):
        return np.sum(np.log((1 + ((u - x0) / γ0) ** 2) / (1 + ((u - x1) / γ1) ** 2)))

    def lr_laplace(u, v):
        return np.sum(np.abs(u - μ0) / b0 - np.abs(u - μ1) / b1)

    # def lr_gennorm(u, v):
    #     return np.sum(np.abs((u - m0) / s0) ** β0 - np.abs((u - m1) / s1) ** β1)

    def lr_kde(u, v):
        return np.sum(np.log(kde_h1(u) / kde_h0(u)))
    
    def lr_imb(u, v):
        return np.sum(np.log(imb(u, μ_imb1, σ_imb1, ν_imb1) / imb(u, μ_imb0, σ_imb0, ν_imb0)))
    
    # Plot ROC curves
    tests = copy(ALL_TESTS)
    print('📊 Simulated case:')
    plot_rocs(t0_sim, t1_sim, tests=tests, figsize=(4, 4))
    print('📊 Cauchy case:')
    tests['Likelihood Ratio'] = lr_cauchy
    plot_rocs(t0_cauchy, t1_cauchy, tests=tests, figsize=(4, 4))
    print('📊 Laplace case:')
    tests['Likelihood Ratio'] = lr_laplace
    plot_rocs(t0_laplace, t1_laplace, tests=tests, figsize=(4, 4))
    # print('📊 Gennorm case:')
    # tests['Likelihood Ratio'] = lr_gennorm
    # plot_rocs(t0_gennorm, t1_gennorm, tests=tests, figsize=(4, 4))
    print('📊 KDE case:')
    tests['Likelihood Ratio'] = lr_kde
    plot_rocs(t0_sim, t1_sim, tests=tests, figsize=(4, 4))
    print('📊 IMB case:')
    tests['Likelihood Ratio'] = lr_imb
    plot_rocs(t0_imb, t1_imb, tests=tests, figsize=(4, 4))
    print('✅ ROC curves plotted.')