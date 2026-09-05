"""CPU versus stateless and persistent CUDA prediction attribution.

Run this on an NVIDIA CUDA host. Reported GPU time includes device allocation,
host-to-device transfers, kernel execution, synchronization, and result copy.
Model fitting, tree parsing, JIT compilation, and baseline-leaf caching are
warmed before timing both backends.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from treeig import GPUTreeIG, TreeIG
from treeig.core import _compute_y0_per_baseline_tree
from treeig.cuda_backend import attribute_cuda, cuda_available


def _time(call, repeats):
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        samples.append(time.perf_counter() - start)
    return result, float(np.median(samples))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", nargs="+", type=int, default=[100, 1000, 10000])
    parser.add_argument("--baselines", nargs="+", type=int, default=[1, 10, 100])
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--forest-depths", nargs="+", type=int, default=[6])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads-per-block", type=int, default=128)
    args = parser.parse_args()
    if not cuda_available():
        raise SystemExit("No CUDA device is available.")

    rng = np.random.default_rng(20260828)
    max_n = max(args.n)
    X = rng.normal(size=(max_n + 2500, 20))
    y = (
        1.2 * X[:, 0] - 0.8 * X[:, 1] ** 2
        + 0.5 * X[:, 2] * X[:, 3] + np.sin(X[:, 4])
    )
    train_X, train_y = X[:2000], y[:2000]
    evaluation = X[2000:2000 + max_n]
    background = X[-100:]
    models = {
        "boosting_depth3": GradientBoostingRegressor(
            n_estimators=args.trees, max_depth=3, random_state=1
        ).fit(train_X, train_y),
    }
    for depth in args.forest_depths:
        models[f"forest_depth{depth}"] = RandomForestRegressor(
            n_estimators=args.trees, max_depth=depth,
            random_state=2, n_jobs=1,
        ).fit(train_X, train_y)

    for model_name, model in models.items():
        explainer = TreeIG(model)
        arrays = explainer._arrays
        for n in args.n:
            for b in args.baselines:
                data = evaluation[:n]
                baselines = np.ascontiguousarray(background[:b])
                weights = np.arange(1.0, b + 1.0)
                weights /= weights.sum()
                y0 = _compute_y0_per_baseline_tree(arrays, baselines)

                cpu_call = lambda: explainer.attribute(
                    data, baseline=baselines, baseline_weights=weights
                )
                gpu_call = lambda: attribute_cuda(
                    arrays, baselines, weights, data, explainer.time_tol, y0,
                    threads_per_block=args.threads_per_block,
                )[0]
                persistent = GPUTreeIG(
                    model,
                    baseline=baselines,
                    baseline_weights=weights,
                    threads_per_block=args.threads_per_block,
                )
                persistent_call = lambda: persistent.attribute(data)
                cpu_call()
                gpu_call()
                persistent_call()
                cpu, cpu_seconds = _time(cpu_call, args.repeats)
                gpu, gpu_seconds = _time(gpu_call, args.repeats)
                resident, resident_seconds = _time(
                    persistent_call, args.repeats
                )
                max_abs_difference = float(np.max(np.abs(cpu - resident)))
                scale = max(1.0, float(np.max(np.abs(cpu))))
                print(json.dumps({
                    "model": model_name,
                    "n": n,
                    "baselines": b,
                    "trees": args.trees,
                    "tasks": n * b * args.trees,
                    "cpu_ms": 1000.0 * cpu_seconds,
                    "gpu_transfer_inclusive_ms": 1000.0 * gpu_seconds,
                    "gpu_persistent_ms": 1000.0 * resident_seconds,
                    "scratch_widths": persistent._gpu._scratch_widths,
                    "stateless_speedup": cpu_seconds / gpu_seconds,
                    "persistent_speedup": cpu_seconds / resident_seconds,
                    "stateless_vs_persistent": gpu_seconds / resident_seconds,
                    "max_abs_difference": max_abs_difference,
                    "max_scaled_difference": max_abs_difference / scale,
                }))


if __name__ == "__main__":
    main()
