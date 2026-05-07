import os
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy as copy
import tqdm
import yaml

def _run_single_calibration(task):

    ctx, metric_name, sample_index, metric_index, seed, metrics = task
    task_metric_index = metric_index
    metric_function = metrics[metric_name]
    working_ctx = copy(ctx)

    rng = np.random.default_rng(seed)
    working_ctx.chip.σ = np.abs(rng.normal(0.0, 1.0, len(working_ctx.chip.σ))) * working_ctx.interferometer.λ

    history = working_ctx.chip.calibrate(plot=False, β=0.6, hooke_jeeves_metric=metric_function)

    depths_history = np.asarray(history["depths"], dtype=float)
    metric_history = np.asarray(history["metric"], dtype=float).ravel()

    # Evaluate all the metrics on the final state
    final_matrics = np.empty(len(metrics), dtype=float)
    for eval_metric_index, (metric_name, metric_function) in enumerate(metrics.items()):
        outs = working_ctx.observe()
        final_matrics[eval_metric_index] = metric_function(outs)

    return {
        "metric_index": task_metric_index,
        "sample_index": sample_index,
        "history_length": metric_history.size,
        "depths_history": depths_history,
        "metric_history": metric_history,
        "final_metrics": final_matrics,
    }

def generate_data(ctx, metrics, samples=1000):

    # Check if data already exists in cache
    if os.path.exists("data.npz") and os.path.exists("data.yml"):
        with open("data.yml", "r") as f:
            metadata = yaml.safe_load(f)
        
        if metadata.get("samples") >= samples and metadata.get("metrics") == list(metrics.keys()):
            print("Loading data from cache...")
            data = np.load("data.npz")
            return data["depths_histories"], data["metrics_histories"], data["final_metrics"]

    nb_metrics = len(metrics)
    nb_runs = samples * nb_metrics
    
    worker_count = min(os.cpu_count() or 1, nb_runs)
    base_seed = np.random.SeedSequence().entropy

    tasks = [
        (ctx, metric_name, sample_index, metric_index, int(base_seed + sample_index * nb_metrics + metric_index), metrics)
        for sample_index in range(samples)
        for metric_index, metric_name in enumerate(metrics)
    ]

    executor = ProcessPoolExecutor(max_workers=worker_count)
    task_iterator = executor.map(_run_single_calibration, tasks)

    # Declare storage variables
    depths_histories = None
    metrics_histories = None
    final_metrics = None

    progress_label = f"Hooke and Jeeves metric runs ({samples} samples x {nb_metrics} metrics)"
    for result in tqdm.tqdm(task_iterator, total=nb_runs, desc=progress_label):
        metric_index = result["metric_index"]
        sample_index = result["sample_index"]
        history_length = result["history_length"]
        depths_history = result["depths_history"]
        metric_history = result["metric_history"]
        final_metrics_vector = result["final_metrics"]

        if depths_histories is None:
            depths_histories = np.empty((nb_metrics, nb_runs, history_length), dtype=float)
            metrics_histories = np.empty((nb_metrics, nb_runs, history_length), dtype=float)
            final_metrics = np.empty((nb_metrics, nb_runs, nb_metrics), dtype=float)

        depths_histories[metric_index, sample_index, :] = depths_history
        metrics_histories[metric_index, sample_index, :] = metric_history
        final_metrics[metric_index, sample_index, :] = final_metrics_vector

    executor.shutdown()

    # Save data in a cache file
    np.savez_compressed(
        "data.npz",
        depths_histories=depths_histories,
        metrics_histories=metrics_histories,
        final_metrics=final_metrics,
        metric_names=list(metrics.keys()),
    )

    # Save metadata in a yml file
    metadata = {
        "samples": samples,
        "metrics": list(metrics.keys()),
    }
    with open("data.yml", "w") as f:
        yaml.dump(metadata, f)

    return depths_histories, metrics_histories, final_metrics

