"""Compare adaptive and fixed-grid TreeIGNumeric on CatBoost raw scores."""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
from sklearn.datasets import make_classification, make_regression

from treeig import TreeIGNumeric


def _models(seed: int):
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError as exc:
        raise SystemExit("Install CatBoost to run this benchmark.") from exc

    X_reg, y_reg = make_regression(
        n_samples=520, n_features=20, noise=0.1, random_state=seed
    )
    regressor = CatBoostRegressor(
        iterations=80,
        depth=8,
        learning_rate=0.08,
        verbose=False,
        allow_writing_files=False,
        random_seed=seed,
        thread_count=1,
    ).fit(X_reg[:500], y_reg[:500])

    X_cls, y_cls = make_classification(
        n_samples=520,
        n_features=20,
        n_informative=10,
        n_redundant=0,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=seed,
    )
    classifier = CatBoostClassifier(
        iterations=80,
        depth=8,
        learning_rate=0.08,
        verbose=False,
        allow_writing_files=False,
        random_seed=seed,
        thread_count=1,
    ).fit(X_cls[:500], y_cls[:500])
    return (
        ("regression", regressor, X_reg[500:504], X_reg[504:508], (None,)),
        (
            "multiclass",
            classifier,
            X_cls[500:504],
            X_cls[504:508],
            tuple(range(4)),
        ),
    )


def _attribute(model, baseline, data, targets, grid, refine):
    values = []
    outputs = []
    start = time.perf_counter()
    for target in targets:
        explainer = TreeIGNumeric(
            model,
            baseline,
            target=target,
            grid_size=grid,
            max_refine=refine,
            warn_residual=False,
        )
        values.append(explainer.attribute(data[None, :])[0])
        outputs.append(explainer.model_output(data[None, :])[0])
    return (
        np.column_stack(values),
        np.asarray(outputs),
        time.perf_counter() - start,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reference-grid", type=int, default=8192)
    args = parser.parse_args()
    configurations = ((128, 0), (128, 4), (1024, 0), (1024, 4))

    print(
        "| model | grid | refine | max relative L1 error | "
        "max completeness error | ms/path |"
    )
    print("|---|---|---|---|---|---|")
    for name, model, baselines, observations, targets in _models(args.seed):
        references = [
            _attribute(
                model, baseline, data, targets, args.reference_grid, 0
            )[0]
            for baseline, data in zip(baselines, observations)
        ]
        for grid, refine in configurations:
            relative_errors = []
            completeness = []
            elapsed = 0.0
            for baseline, data, reference in zip(
                baselines, observations, references
            ):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    actual, endpoint, duration = _attribute(
                        model, baseline, data, targets, grid, refine
                    )
                elapsed += duration
                for output in range(reference.shape[1]):
                    numerator = np.abs(
                        actual[:, output] - reference[:, output]
                    ).sum()
                    denominator = np.abs(reference[:, output]).sum()
                    relative_errors.append(
                        numerator / max(float(denominator), np.finfo(float).eps)
                    )
                baseline_output = np.asarray(
                    [
                        TreeIGNumeric(model, baseline, target=target).model_output(
                            baseline[None, :]
                        )[0]
                        for target in targets
                    ]
                )
                completeness.append(
                    float(
                        np.max(
                            np.abs(
                                actual.sum(axis=0)
                                - endpoint
                                + baseline_output
                            )
                        )
                    )
                )
            print(
                f"| {name} | {grid} | {refine} | "
                f"{max(relative_errors):.2%} | {max(completeness):.3g} | "
                f"{1000.0 * elapsed / len(observations):.1f} |"
            )


if __name__ == "__main__":
    main()
