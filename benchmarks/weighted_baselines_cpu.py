"""Benchmark TreeIG attribution over observation and baseline distributions.

Run from the repository root, for example::

    PYTHONPATH=. python benchmarks/weighted_baselines_cpu.py
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from treeig import TreeIG


N_VALUES = (1, 10, 100, 1000)
B_VALUES = (1, 10, 100)


def _data(seed=2026):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(2200, 12))
    y = (
        1.3 * X[:, 0]
        - 0.9 * X[:, 1] ** 2
        + 0.7 * X[:, 2] * X[:, 3]
        + np.sin(X[:, 4])
        + 0.2 * X[:, 5]
    )
    return X, y


def _models(X, y):
    return {
        "boosting_depth2_50": GradientBoostingRegressor(
            n_estimators=50, max_depth=2, random_state=2026
        ).fit(X, y),
        "forest_depth8_25": RandomForestRegressor(
            n_estimators=25, max_depth=8, random_state=2026, n_jobs=1
        ).fit(X, y),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--save-results")
    args = parser.parse_args()

    X, y = _data()
    X_train, y_train = X[:1200], y[:1200]
    X_eval = X[1200:2200]
    baselines = X_train[:100]
    all_outputs = {}

    for model_name, model in _models(X_train, y_train).items():
        explainer = TreeIG(model)
        for n in N_VALUES:
            for b in B_VALUES:
                baseline = baselines[:b]
                weights = np.arange(1.0, b + 1.0)
                weights /= weights.sum()
                data = X_eval[:n]

                expected_delta = model.predict(data) - weights @ model.predict(baseline)
                output = explainer.attribute(
                    data, baseline=baseline, baseline_weights=weights
                )
                max_completeness_error = float(
                    np.max(np.abs(output.sum(axis=1) - expected_delta))
                )

                samples = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    output = explainer.attribute(
                        data, baseline=baseline, baseline_weights=weights
                    )
                    samples.append(time.perf_counter() - start)

                key = f"{model_name}__n{n}__b{b}"
                all_outputs[key] = output
                print(json.dumps({
                    "model": model_name,
                    "n": n,
                    "baselines": b,
                    "median_ms": 1000.0 * float(np.median(samples)),
                    "min_ms": 1000.0 * float(np.min(samples)),
                    "max_completeness_error": max_completeness_error,
                }))

    if args.save_results:
        np.savez_compressed(args.save_results, **all_outputs)


if __name__ == "__main__":
    main()
