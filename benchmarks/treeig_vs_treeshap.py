"""Small reproducible runtime comparison of TreeIG and exact TreeSHAP.

The benchmark deliberately uses one baseline row for both methods. This aligns
their reference prediction and keeps TreeSHAP in exact interventional mode,
although the two methods still answer different attribution questions.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time

import numpy as np
import shap
import sklearn
from sklearn.datasets import make_regression
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor

from treeig import TreeIG


def median_seconds(function, repeats: int) -> float:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    X, y = make_regression(
        n_samples=6_000, n_features=20, n_informative=15, noise=0.1, random_state=0
    )
    X_train, X_eval = X[:4_000], np.ascontiguousarray(X[4_000:])
    y_train = y[:4_000]
    baseline = np.median(X_train, axis=0)

    models = [
        ("scikit-learn gradient boosting", GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)),
        ("Extremely randomized trees (ExtraTrees)", ExtraTreesRegressor(
            n_estimators=200, max_depth=12, random_state=0, n_jobs=1)),
    ]

    print(f"Python {platform.python_version()}; {platform.machine()}; "
          f"NumPy {np.__version__}; scikit-learn {sklearn.__version__}; SHAP {shap.__version__}")
    print("model,n_eval,treeig_ms,treeshap_ms,treeig_over_treeshap,"
          "max_reference_error,max_additivity_error")

    for model_name, model in models:
        model.fit(X_train, y_train)
        treeig = TreeIG(model, baseline=baseline)
        treeshap = shap.TreeExplainer(
            model, data=baseline.reshape(1, -1),
            feature_perturbation="interventional", model_output="raw")

        treeig.warmup(X_eval[:3])
        treeig.attribute(X_eval[:10])
        treeshap.shap_values(X_eval[:10], check_additivity=False)

        treeig_expected = float(treeig.model_output(baseline.reshape(1, -1))[0])
        shap_expected = float(np.asarray(treeshap.expected_value).reshape(-1)[0])
        reference_error = abs(treeig_expected - shap_expected)

        for n_eval in (100, 1_000):
            batch = X_eval[:n_eval]
            treeig_s = median_seconds(lambda: treeig.attribute(batch), args.repeats)
            treeshap_s = median_seconds(
                lambda: treeshap.shap_values(batch, check_additivity=False), args.repeats)
            treeig_values = treeig.attribute(batch)
            treeshap_values = np.asarray(
                treeshap.shap_values(batch, check_additivity=False))
            predictions = np.asarray(model.predict(batch))
            additivity_error = max(
                float(np.max(np.abs(treeig_expected + treeig_values.sum(axis=1) - predictions))),
                float(np.max(np.abs(shap_expected + treeshap_values.sum(axis=1) - predictions))),
            )
            print(f"{model_name},{n_eval},{1e3 * treeig_s:.3f},{1e3 * treeshap_s:.3f},"
                  f"{treeig_s / treeshap_s:.3f},{reference_error:.3e},{additivity_error:.3e}")


if __name__ == "__main__":
    sys.exit(main())
