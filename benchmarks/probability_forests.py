"""Stress TreeIGNumeric on probability-averaging sklearn classifiers.

The benchmark compares numerical path-event attribution with an independent
oracle that traverses each fitted sklearn tree along the one-dimensional
baseline-to-input path.  It is intended for validation and default selection,
not as a stable performance leaderboard.
"""

from __future__ import annotations

import argparse
import time
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from treeig import TreeIGNumeric


@dataclass(frozen=True)
class Scenario:
    name: str
    estimator: str
    n_estimators: int
    max_depth: int | None
    n_features: int
    n_classes: int


QUICK_SCENARIOS = (
    Scenario("decision_tree", "tree", 1, None, 20, 2),
    Scenario("small_forest", "forest", 10, 3, 10, 2),
    Scenario("sparse_forest", "forest", 5, None, 20, 2),
    Scenario("reference", "forest", 50, 6, 20, 2),
    Scenario("large_forest", "forest", 200, 6, 20, 2),
    Scenario("deep_forest", "forest", 50, 12, 20, 2),
    Scenario("wide_forest", "forest", 50, 6, 200, 2),
    Scenario("extra_trees", "extra", 50, 6, 20, 2),
    Scenario("multiclass", "forest", 50, 6, 20, 4),
    Scenario("sparse_multiclass", "forest", 5, None, 20, 4),
)

FULL_SCENARIOS = QUICK_SCENARIOS + (
    Scenario("very_wide", "forest", 100, 8, 500, 2),
    Scenario("very_large", "forest", 500, 8, 50, 2),
    Scenario("deep_multiclass", "forest", 100, None, 50, 5),
)


def _centered_log_scores(probabilities: np.ndarray, floor: float) -> np.ndarray:
    probabilities = np.maximum(np.asarray(probabilities, dtype=float), floor)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    scores = np.log(probabilities)
    return scores - scores.mean(axis=1, keepdims=True)


def _output_scores(model, X: np.ndarray, floor: float) -> np.ndarray:
    scores = _centered_log_scores(model.predict_proba(X), floor)
    if scores.shape[1] == 2:
        return (scores[:, 1] - scores[:, 0])[:, None]
    return scores


def _tree_path_crossings(
    estimator, baseline: np.ndarray, data: np.ndarray
) -> list[tuple[float, int]]:
    """Return every reachable split boundary along one interpolation path."""
    tree = estimator.tree_
    direction = data - baseline
    crossings: list[tuple[float, int]] = []

    def visit(node: int, lower: float, upper: float) -> None:
        feature = int(tree.feature[node])
        if feature < 0:
            return
        threshold = float(tree.threshold[node])
        slope = float(direction[feature])
        if slope == 0.0:
            midpoint = 0.5 * (lower + upper)
            value = baseline[feature] + midpoint * slope
            child = (
                tree.children_left[node]
                if value <= threshold
                else tree.children_right[node]
            )
            visit(int(child), lower, upper)
            return

        crossing = (threshold - baseline[feature]) / slope
        if lower < crossing < upper:
            crossings.append((float(crossing), feature))
            if slope > 0.0:
                before, after = tree.children_left[node], tree.children_right[node]
            else:
                before, after = tree.children_right[node], tree.children_left[node]
            visit(int(before), lower, float(crossing))
            visit(int(after), float(crossing), upper)
            return

        midpoint = 0.5 * (lower + upper)
        value = baseline[feature] + midpoint * slope
        child = (
            tree.children_left[node]
            if value <= threshold
            else tree.children_right[node]
        )
        visit(int(child), lower, upper)

    visit(0, 0.0, 1.0)
    return crossings


def _crossing_map(
    model, baseline: np.ndarray, data: np.ndarray
) -> dict[float, list[tuple[object, int]]]:
    estimators: Iterable[object]
    if hasattr(model, "estimators_"):
        estimators = np.asarray(model.estimators_, dtype=object).ravel()
    else:
        estimators = (model,)
    crossings: dict[float, list[tuple[object, int]]] = {}
    for estimator in estimators:
        for crossing, feature in _tree_path_crossings(estimator, baseline, data):
            crossings.setdefault(crossing, []).append((estimator, feature))
    return crossings


def _effective_leaf_support(
    transitions: Sequence[tuple[object, int]], points: np.ndarray
) -> float:
    """Return mean harmonic leaf support across trees changing at an event."""
    supports = []
    for estimator, _ in transitions:
        leaves = estimator.apply(points)
        counts = estimator.tree_.weighted_n_node_samples[leaves]
        left, right = float(counts[0]), float(counts[1])
        if left > 0.0 and right > 0.0:
            supports.append(2.0 / (1.0 / left + 1.0 / right))
    return float(np.mean(supports)) if supports else 0.0


def _oracle(
    model, baseline: np.ndarray, data: np.ndarray, floor: float
) -> tuple[np.ndarray, int, int, bool, np.ndarray]:
    crossings = _crossing_map(model, baseline, data)
    times = sorted(crossings)
    n_outputs = 1 if len(model.classes_) == 2 else len(model.classes_)
    attribution = np.zeros((data.size, n_outputs))
    active_events = 0
    tied_events = 0
    support_mass = np.zeros(n_outputs)
    jump_mass = np.zeros(n_outputs)

    representatives = [0.0]
    representatives.extend(
        0.5 * (left + right) for left, right in zip(times[:-1], times[1:])
    )
    representatives.append(1.0)
    path_points = baseline + np.asarray(representatives)[:, None] * (data - baseline)
    path_has_zero = bool(np.any(model.predict_proba(path_points) <= 0.0))

    for index, crossing in enumerate(times):
        left = 0.5 * ((times[index - 1] if index else 0.0) + crossing)
        right = 0.5 * (
            crossing + (times[index + 1] if index + 1 < len(times) else 1.0)
        )
        points = baseline + np.array([left, right])[:, None] * (data - baseline)
        jump = np.diff(_output_scores(model, points, floor), axis=0)[0]
        if not np.any(np.abs(jump) > 1e-14):
            continue
        active_events += 1
        transitions = crossings[crossing]
        features = {feature for _, feature in transitions}
        magnitude = np.abs(jump)
        support_mass += magnitude * _effective_leaf_support(transitions, points)
        jump_mass += magnitude
        if len(features) != 1:
            tied_events += 1
            continue
        attribution[next(iter(features))] += jump
    effective_support = np.divide(
        support_mass,
        jump_mass,
        out=np.zeros_like(support_mass),
        where=jump_mass > 0.0,
    )
    return attribution, active_events, tied_events, path_has_zero, effective_support


def _support_weighted_relative_error(
    numerators: Sequence[float],
    denominators: Sequence[float],
    supports: Sequence[float],
) -> float:
    numerators_array = np.asarray(numerators, dtype=float)
    denominators_array = np.asarray(denominators, dtype=float)
    supports_array = np.asarray(supports, dtype=float)
    if numerators_array.size == 0:
        return float("nan")
    numerator = float(np.sum(supports_array * numerators_array))
    denominator = float(np.sum(supports_array * denominators_array))
    return numerator / max(denominator, np.finfo(float).eps)


def _fit(scenario: Scenario, seed: int, n_eval: int):
    n_train = 800
    informative = min(scenario.n_features, max(5, 2 * scenario.n_classes))
    X, y = make_classification(
        n_samples=n_train + 2 * n_eval,
        n_features=scenario.n_features,
        n_informative=informative,
        n_redundant=0,
        n_classes=scenario.n_classes,
        n_clusters_per_class=1,
        random_state=seed,
    )
    options = {"max_depth": scenario.max_depth, "random_state": seed}
    if scenario.estimator == "tree":
        model = DecisionTreeClassifier(**options)
    elif scenario.estimator == "extra":
        model = ExtraTreesClassifier(
            n_estimators=scenario.n_estimators, n_jobs=1, **options
        )
    else:
        model = RandomForestClassifier(
            n_estimators=scenario.n_estimators, n_jobs=1, **options
        )
    model.fit(X[:n_train], y[:n_train])
    return model, X[n_train : n_train + n_eval], X[n_train + n_eval :]


def _numeric(
    model, baseline, data, floor: float, grid_size: int, max_refine: int
):
    targets: Sequence[int | None]
    targets = (
        (None,)
        if len(model.classes_) == 2
        else tuple(range(len(model.classes_)))
    )
    values = []
    infos = []
    start = time.perf_counter()
    for target in targets:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            explainer = TreeIGNumeric(
                model,
                baseline,
                target=target,
                probability_to_score=True,
                probability_floor=floor,
                grid_size=grid_size,
                max_refine=max_refine,
                warn_residual=False,
            )
        phi, target_infos, _ = explainer.explain(data[None, :])
        values.append(phi[0])
        infos.append(target_infos[0])
    elapsed = time.perf_counter() - start
    return np.column_stack(values), infos, elapsed


def benchmark(
    scenario: Scenario,
    grids: Sequence[int],
    n_eval: int,
    floor: float,
    seed: int,
    max_refine: int,
) -> list[dict]:
    model, baselines, observations = _fit(scenario, seed, n_eval)
    oracles = [
        _oracle(model, baseline, data, floor)
        for baseline, data in zip(baselines, observations)
    ]
    rows = []
    for grid_size in grids:
        # Exclude one-time optional-backend imports from steady-state timing.
        _numeric(
            model, baselines[0], observations[0], floor, grid_size, max_refine
        )
        relative_errors = []
        error_numerators = []
        error_denominators = []
        relative_error_supports = []
        completeness = []
        missed_fractions = []
        elapsed = 0.0
        for baseline, data, oracle in zip(baselines, observations, oracles):
            expected, oracle_events, tied_events, _, effective_support = oracle
            actual, infos, duration = _numeric(
                model, baseline, data, floor, grid_size, max_refine
            )
            elapsed += duration
            if tied_events == 0:
                for output in range(expected.shape[1]):
                    denominator = float(np.sum(np.abs(expected[:, output])))
                    numerator = float(
                        np.sum(np.abs(actual[:, output] - expected[:, output]))
                    )
                    relative_errors.append(
                        numerator / max(denominator, np.finfo(float).eps)
                    )
                    error_numerators.append(numerator)
                    error_denominators.append(denominator)
                    relative_error_supports.append(effective_support[output])
            endpoint_delta = (
                _output_scores(model, data[None, :], floor)
                - _output_scores(model, baseline[None, :], floor)
            )[0]
            completeness.append(
                float(np.max(np.abs(actual.sum(axis=0) - endpoint_delta)))
            )
            detected = max(int(info["n_events"]) for info in infos)
            if oracle_events:
                missed_fractions.append(max(0.0, 1.0 - detected / oracle_events))
        rows.append(
            {
                "scenario": scenario.name,
                "grid": grid_size,
                "refine": max_refine,
                "features": scenario.n_features,
                "trees": scenario.n_estimators,
                "depth": (
                    scenario.max_depth
                    if scenario.max_depth is not None
                    else "full"
                ),
                "classes": scenario.n_classes,
                "median_relative_l1_error": (
                    float(np.median(relative_errors))
                    if relative_errors
                    else float("nan")
                ),
                "p95_relative_l1_error": float(
                    np.percentile(relative_errors, 95)
                ) if relative_errors else float("nan"),
                "support_weighted_relative_l1_error": (
                    _support_weighted_relative_error(
                        error_numerators,
                        error_denominators,
                        relative_error_supports,
                    )
                ),
                "median_effective_leaf_support": float(
                    np.median(np.concatenate([item[4] for item in oracles]))
                ),
                "max_completeness_error": max(completeness),
                "mean_missed_event_fraction": float(np.mean(missed_fractions)),
                "zero_path_fraction": float(np.mean([item[3] for item in oracles])),
                "milliseconds_per_path": 1000.0 * elapsed / n_eval,
            }
        )
    return rows


def _print_rows(rows: Sequence[dict]) -> None:
    headings = (
        "scenario",
        "grid",
        "refine",
        "features",
        "trees",
        "depth",
        "classes",
        "median relative L1 error",
        "95th pct relative L1 error",
        "support-weighted relative L1 error",
        "median effective leaf support",
        "max completeness error",
        "missed events",
        "zero paths",
        "ms/path",
    )
    print("| " + " | ".join(headings) + " |")
    print("|" + "|".join("---" for _ in headings) + "|")
    for row in rows:
        print(
            "| {scenario} | {grid} | {refine} | {features} | {trees} | "
            "{depth} | "
            "{classes} | {median_relative_l1_error:.2%} | "
            "{p95_relative_l1_error:.2%} | "
            "{support_weighted_relative_l1_error:.2%} | "
            "{median_effective_leaf_support:.1f} | "
            "{max_completeness_error:.3g} | "
            "{mean_missed_event_fraction:.1%} | {zero_path_fraction:.1%} | "
            "{milliseconds_per_path:.1f} |".format(**row)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--grids", nargs="+", type=int, default=(64, 256, 1024))
    parser.add_argument("--eval", type=int, default=8, dest="n_eval")
    parser.add_argument("--floor", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-refine", type=int, default=4)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        help="run only the named scenarios from the selected profile",
    )
    args = parser.parse_args()
    if (
        args.n_eval < 1
        or args.max_refine < 0
        or any(grid < 1 for grid in args.grids)
    ):
        parser.error("--eval/grids must be positive and --max-refine nonnegative")
    scenarios = QUICK_SCENARIOS if args.profile == "quick" else FULL_SCENARIOS
    if args.scenarios:
        requested = set(args.scenarios)
        available = {scenario.name for scenario in scenarios}
        unknown = requested - available
        if unknown:
            parser.error("unknown scenarios: " + ", ".join(sorted(unknown)))
        scenarios = tuple(
            scenario for scenario in scenarios if scenario.name in requested
        )
    rows = []
    for scenario in scenarios:
        rows.extend(
            benchmark(
                scenario,
                args.grids,
                args.n_eval,
                args.floor,
                args.seed,
                args.max_refine,
            )
        )
    _print_rows(rows)


if __name__ == "__main__":
    main()
