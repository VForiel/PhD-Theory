import phise
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from copy import deepcopy as copy
import pandas as pd

# Defining all the metrics to compare -----------------------------------------

# Sum(N) /B
def m1(outs):
    outs[outs <= 1] = 1  # Avoid log(0) issues
    return np.sum(outs[1:]) / outs[0]
# Max(N) / B
def m2(outs):
    outs[outs <= 1] = 1  # Avoid log(0) issues
    return np.max(outs[1:]) / outs[0]
# Sum(N)
def m3(outs):
    outs[outs <= 1] = 1  # Avoid log(0) issues
    return np.sum(outs[1:])
# Max(N)
def m4(outs):
    outs[outs <= 1] = 1  # Avoid log(0) issues
    return np.max(outs[1:])
# N3/B
def m5(outs, n=1):
    outs[outs <= 1] = 1  # Avoid log(0) issues
    return outs[n] / outs[0]
# N3
def m6(outs, n=1):
    outs[outs <= 1] = 1  # Avoid log(0) issues
    return outs[n]

metrics = {
    "sum(N) / B": m1,
    "max(N) / B": m2,
    "sum(N)": m3,
    "max(N)": m4,
    # "O1/B": lambda x: m5(x, n=1),
    # "O1": lambda x: m6(x, n=1),
    # "O2/B": lambda x: m5(x, n=2),
    # "O2": lambda x: m6(x, n=2),
    "O3/B": lambda x: m5(x, n=3),
    "O3": lambda x: m6(x, n=3),
}

# Show convergence for each metric on a single run (with random initial σ) ----

def run_once(ctx: phise.Context, name=None):
    ref_ctx = ctx

    if name is not None:
        considered_metrics = {name: metrics[name]}
    else:
        considered_metrics = metrics
    
    for name, metric in considered_metrics.items():

        ctx = copy(ref_ctx)
        ctx.chip.σ = np.abs(np.random.normal(0, 1, len(ctx.chip.σ))) * ctx.interferometer.λ/10

        print(f"Calibrating with metric: {name}")
        history = ctx.chip.calibrate(plot=True, β=0.8, hooke_jeeves_metric=metric)
        depths_history = history["depths"]
        for i in range(depths_history.shape[1]):
            print(f"Mean on the last 100 null depth values for the output {i}: {np.mean(depths_history[-100:, i]):.2e}")
        print(f"Total mean: {np.mean(depths_history[-100:, :]):.2e}")
        ctx.chip.laugiergram(show=True)
        plt.show()

# Compare distributions for each metric ---------------------------------------

def compare(ctx: phise.Context, N=100, demo=False, plot=True):
    ref_ctx = ctx

    # Bootstrap the mean null depth for each metric to compare performances ---

    data = np.zeros((len(metrics), N, 3)) # (metric, sample, output)
    metric_data = np.zeros((len(metrics), N)) # (sample, metric)

    for i in tqdm.tqdm(range(N)):
        for name, metric in metrics.items():

            ctx = copy(ref_ctx)
            ctx.chip.σ = np.abs(np.random.normal(0, 1, len(ctx.chip.σ))) * ctx.interferometer.λ/10

            history = ctx.chip.calibrate(plot=False, β=0.9, hooke_jeeves_metric=metric)
            depths_history = history["depths"]
            mean_depth = np.mean(depths_history[-100:, :], axis=0)
            data[list(metrics.keys()).index(name), i] = mean_depth
            metric_data[list(metrics.keys()).index(name), i] = history["metric"][-1]


    data_means = np.mean(data, axis=1)
    data_stds = np.std(data, axis=1)
    data_mins = np.min(data, axis=1)
    data_maxs = np.max(data, axis=1)

    # Display table with matrics in line and outputs means & stds in columns (as well as the total mean and std)

    table_data = []
    for i, name in enumerate(metrics.keys()):
        row = [name]
        row.extend([f"{data_means[i, j]:.2e} ± {data_stds[i, j]:.2e} | {data_mins[i, j]:.2e} - {data_maxs[i, j]:.2e}" for j in range(data.shape[2])])
        row.append(f"{np.mean(data_means[i]):.2e} ± {np.mean(data_stds[i]):.2e} | {np.min(data_mins[i]):.2e} - {np.max(data_maxs[i]):.2e}")
        table_data.append(row)
    columns = ["Metric", "Output 1", "Output 2", "Output 3", "Total"]
    df = pd.DataFrame(table_data, columns=columns)

    # Plot distribs for each output
    fig = None
    if plot:
        fig, ax = plt.subplots(1,3, figsize=(20, 5))
        for j in range(data.shape[2]):
            for i, name in enumerate(list(metrics.keys())[:4]):
                ax[j].hist(data[i, :, j], bins=int(np.sqrt(N)), alpha=0.5, label=name)
            ax[j].set_title(f"Null depth distrib. for output {j+1}")
            ax[j].set_xlabel("Mean null depth")
            ax[j].set_xscale("log")
            ax[j].set_ylabel("Occurences")
            ax[j].legend()
        plt.show()

        fig2, ax2 = plt.subplots(1, len(metrics), figsize=(len(metrics)*5, 5))
        for i, name in enumerate(metrics.keys()):
            ax2[i].hist(metric_data[i], bins=int(np.sqrt(N)), alpha=0.5)
            ax2[i].set_title(f"Final metric value distrib. for {name}")
            ax2[i].set_xlabel("Final metric value")
            ax2[i].set_xscale("log")
            ax2[i].set_ylabel("Occurences")

    return df, fig, fig2