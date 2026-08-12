from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .utils import _resolve_binary_classifier_target


def _pack_sklearn_tree_objects(
    trees: List[Any],
    tree_weight: np.ndarray,
    n_features: int,
    backend: str,
    output_kind: str,
    n_outputs: int = 1,
    target_required: bool = False,
) -> Dict[str, Any]:
    """
    Pack sklearn tree objects into the common TreeIG internal format.

    sklearn tree convention: left branch if x[j] <= threshold.
    """
    n_trees = len(trees)

    if n_trees == 0:
        raise ValueError("model contains no trees.")

    max_nodes = max(t.tree_.node_count for t in trees)

    children_left = -np.ones((n_trees, max_nodes), dtype=np.int64)
    children_right = -np.ones((n_trees, max_nodes), dtype=np.int64)
    feature = -np.ones((n_trees, max_nodes), dtype=np.int64)
    threshold = np.zeros((n_trees, max_nodes), dtype=np.float64)
    value = np.zeros((n_trees, max_nodes), dtype=np.float64)
    left_inclusive = np.ones((n_trees, max_nodes), dtype=np.bool_)
    round_input = np.zeros((n_trees, max_nodes), dtype=np.bool_)

    for m, tree in enumerate(trees):
        tr = tree.tree_
        n = tr.node_count

        if tr.value.shape[1] != 1 or tr.value.shape[2] != 1:
            raise ValueError(
                "TreeIG currently supports scalar tree outputs only. "
                f"Tree {m} has value shape {tr.value.shape}."
            )

        children_left[m, :n] = tr.children_left.astype(np.int64)
        children_right[m, :n] = tr.children_right.astype(np.int64)
        feature[m, :n] = tr.feature.astype(np.int64)
        threshold[m, :n] = tr.threshold.astype(np.float64)
        value[m, :n] = tr.value[:, 0, 0].astype(np.float64)

    tree_weight = np.asarray(tree_weight, dtype=np.float64)

    if tree_weight.ndim != 1 or tree_weight.shape[0] != n_trees:
        raise ValueError(
            f"tree_weight must have shape ({n_trees},), got {tree_weight.shape}."
        )

    return {
        "children_left": children_left,
        "children_right": children_right,
        "feature": feature,
        "threshold": threshold,
        "value": value,
        "left_inclusive": left_inclusive,
        "round_input": round_input,
        "tree_weight": tree_weight,
        "n_features": int(n_features),
        "backend": backend,
        "output_kind": output_kind,
        "n_outputs": int(n_outputs),
        "target_required": bool(target_required),
    }


def _extract_sklearn_gradient_boosting_regressor(model: Any) -> Dict[str, Any]:
    """
    Extract sklearn GradientBoostingRegressor.

    Prediction form: init_prediction + learning_rate * sum_m tree_m(x).
    The init_prediction cancels in f(x) - f(x0).
    """
    trees = list(model.estimators_.ravel())
    n_trees = len(trees)

    return _pack_sklearn_tree_objects(
        trees=trees,
        tree_weight=np.full(n_trees, float(model.learning_rate), dtype=np.float64),
        n_features=model.n_features_in_,
        backend="sklearn_gradient_boosting_regressor",
        output_kind="regression",
    )


def _extract_sklearn_decision_tree_regressor(model: Any) -> Dict[str, Any]:
    """Extract sklearn DecisionTreeRegressor."""
    return _pack_sklearn_tree_objects(
        trees=[model],
        tree_weight=np.ones(1, dtype=np.float64),
        n_features=model.n_features_in_,
        backend="sklearn_decision_tree_regressor",
        output_kind="regression",
    )


def _extract_sklearn_forest_regressor(model: Any, backend: str) -> Dict[str, Any]:
    """Extract sklearn RandomForestRegressor or ExtraTreesRegressor."""
    trees = list(model.estimators_)
    n_trees = len(trees)

    if n_trees == 0:
        raise ValueError("forest model contains no fitted trees.")

    return _pack_sklearn_tree_objects(
        trees=trees,
        tree_weight=np.full(n_trees, 1.0 / float(n_trees), dtype=np.float64),
        n_features=model.n_features_in_,
        backend=backend,
        output_kind="regression",
    )


def _extract_sklearn_gradient_boosting_classifier(
    model: Any,
    target: Optional[int],
) -> Dict[str, Any]:
    """Extract sklearn GradientBoostingClassifier in raw-score space."""
    n_classes = int(len(model.classes_))

    if n_classes == 2:
        resolved = _resolve_binary_classifier_target(target)
        trees = list(model.estimators_[:, 0].ravel())
        n_trees = len(trees)
        arrays = _pack_sklearn_tree_objects(
            trees=trees,
            tree_weight=np.full(
                n_trees,
                resolved.sign * float(model.learning_rate),
                dtype=np.float64,
            ),
            n_features=model.n_features_in_,
            backend="sklearn_gradient_boosting_classifier_binary",
            output_kind="binary_margin",
            n_outputs=2,
        )
        arrays["target"] = resolved.target
        return arrays

    if target is None:
        raise ValueError(
            "Multiclass GradientBoostingClassifier requires target=<class index>."
        )

    k = int(target)
    if k < 0 or k >= n_classes:
        raise ValueError(f"target must be in [0, {n_classes - 1}], got {target}.")

    trees = list(model.estimators_[:, k].ravel())
    n_trees = len(trees)

    arrays = _pack_sklearn_tree_objects(
        trees=trees,
        tree_weight=np.full(n_trees, float(model.learning_rate), dtype=np.float64),
        n_features=model.n_features_in_,
        backend="sklearn_gradient_boosting_classifier_multiclass",
        output_kind="multiclass_margin",
        n_outputs=n_classes,
        target_required=True,
    )
    arrays["target"] = k
    return arrays


def _predict_sklearn_model(model: Any, X: np.ndarray, target: Optional[int]) -> np.ndarray:
    from sklearn.ensemble import GradientBoostingClassifier

    from .utils import _select_prediction_target

    if isinstance(model, GradientBoostingClassifier):
        pred = model.decision_function(X)
        return _select_prediction_target(pred, target)

    return np.asarray(model.predict(X), dtype=np.float64)
