from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils import _resolve_binary_classifier_target, _select_prediction_target


def _lightgbm_model_kind(model: Any) -> Optional[str]:
    try:
        import lightgbm as lgb
    except ImportError:
        return None

    if isinstance(model, lgb.Booster):
        return "booster"

    module = type(model).__module__
    name = type(model).__name__

    if module.startswith("lightgbm.") and hasattr(model, "booster_"):
        if name == "LGBMClassifier":
            return "classifier"
        if name == "LGBMRegressor":
            return "regressor"
        return "sklearn_like"

    return None


def _lightgbm_num_features(model: Any, booster: Any) -> int:
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    if hasattr(booster, "num_feature"):
        return int(booster.num_feature())
    raise ValueError("Could not determine LightGBM number of features.")


def _select_lightgbm_tree_info(
    model: Any,
    model_dict: Dict[str, Any],
    target: Optional[int],
) -> Tuple[np.ndarray, str, int, Optional[int], float]:
    kind = _lightgbm_model_kind(model)
    tree_info = model_dict.get("tree_info", [])
    n_trees = len(tree_info)

    if kind == "classifier":
        n_classes = int(len(model.classes_))
        if n_classes == 2:
            resolved = _resolve_binary_classifier_target(target)
            return (
                np.arange(n_trees, dtype=np.int64),
                "binary_margin",
                2,
                resolved.target,
                resolved.sign,
            )

        if target is None:
            raise ValueError("Multiclass LGBMClassifier requires target=<class index>.")
        k = int(target)
        if k < 0 or k >= n_classes:
            raise ValueError(f"target must be in [0, {n_classes - 1}], got {target}.")
        return (
            np.arange(k, n_trees, n_classes, dtype=np.int64),
            "multiclass_margin",
            n_classes,
            k,
            1.0,
        )

    num_class = int(model_dict.get("num_class", 1) or 1)
    if kind == "booster" and num_class > 1:
        if target is None:
            raise ValueError("Multiclass LightGBM Booster requires target=<class index>.")
        k = int(target)
        if k < 0 or k >= num_class:
            raise ValueError(f"target must be in [0, {num_class - 1}], got {target}.")
        return (
            np.arange(k, n_trees, num_class, dtype=np.int64),
            "multiclass_margin",
            num_class,
            k,
            1.0,
        )

    if target not in (None, 0):
        raise ValueError("Regression target must be None or 0.")
    return np.arange(n_trees, dtype=np.int64), "regression", 1, None, 1.0


def _flatten_lightgbm_tree_iterative(root: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    """Iteratively flatten LightGBM tree structure."""
    nodes: List[Dict[str, Any]] = []
    stack: List[Tuple[Dict[str, Any], Optional[int], Optional[str]]] = [(root, None, None)]

    while stack:
        node, parent_idx, side = stack.pop()
        idx = len(nodes)
        nodes.append({"node": node, "left": -1, "right": -1})

        if parent_idx is not None:
            nodes[parent_idx][side] = idx

        if "leaf_value" not in node:
            stack.append((node["right_child"], idx, "right"))
            stack.append((node["left_child"], idx, "left"))

    return nodes, 0


def _extract_lightgbm_booster(model: Any, target: Optional[int]) -> Dict[str, Any]:
    """Extract lightgbm.Booster, LGBMRegressor, or LGBMClassifier."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("LightGBM support requires lightgbm.") from exc

    kind = _lightgbm_model_kind(model)
    if kind is None:
        raise TypeError(f"Expected LightGBM model, got {type(model).__name__}.")

    if isinstance(model, lgb.Booster):
        booster = model
    else:
        booster = model.booster_

    model_dict = booster.dump_model()
    tree_info = model_dict.get("tree_info", [])
    if len(tree_info) == 0:
        raise ValueError("LightGBM model contains no trees.")

    tree_indices, output_kind, n_outputs, resolved_target, sign = _select_lightgbm_tree_info(
        model, model_dict, target
    )

    flattened: List[List[Dict[str, Any]]] = []
    max_nodes = 0

    for idx in tree_indices:
        tree_struct = tree_info[int(idx)]["tree_structure"]
        nodes, root_idx = _flatten_lightgbm_tree_iterative(tree_struct)
        if root_idx != 0:
            raise RuntimeError("Unexpected LightGBM root index.")
        flattened.append(nodes)
        max_nodes = max(max_nodes, len(nodes))

    n_trees = len(flattened)

    children_left = -np.ones((n_trees, max_nodes), dtype=np.int64)
    children_right = -np.ones((n_trees, max_nodes), dtype=np.int64)
    feature = -np.ones((n_trees, max_nodes), dtype=np.int64)
    threshold = np.zeros((n_trees, max_nodes), dtype=np.float64)
    value = np.zeros((n_trees, max_nodes), dtype=np.float64)
    left_inclusive = np.ones((n_trees, max_nodes), dtype=np.bool_)
    round_input = np.ones((n_trees, max_nodes), dtype=np.bool_)

    for m, nodes in enumerate(flattened):
        for node_idx, rec in enumerate(nodes):
            node = rec["node"]

            if "leaf_value" in node:
                children_left[m, node_idx] = -1
                children_right[m, node_idx] = -1
                value[m, node_idx] = float(node["leaf_value"])
                continue

            decision_type = str(node.get("decision_type", "<="))
            if decision_type != "<=":
                raise NotImplementedError(
                    "TreeIG LightGBM backend currently supports numeric '<=' "
                    f"splits only; got decision_type={decision_type!r}."
                )

            children_left[m, node_idx] = int(rec["left"])
            children_right[m, node_idx] = int(rec["right"])
            feature[m, node_idx] = int(node["split_feature"])
            threshold[m, node_idx] = float(node["threshold"])
            value[m, node_idx] = 0.0
            left_inclusive[m, node_idx] = True
            round_input[m, node_idx] = True

    n_features = _lightgbm_num_features(model, booster)
    backend = "lightgbm_" + (kind if kind != "sklearn_like" else "model")

    return {
        "children_left": children_left,
        "children_right": children_right,
        "feature": feature,
        "threshold": threshold,
        "value": value,
        "left_inclusive": left_inclusive,
        "round_input": round_input,
        "tree_weight": np.full(n_trees, float(sign), dtype=np.float64),
        "n_features": n_features,
        "backend": backend,
        "output_kind": output_kind,
        "n_outputs": n_outputs,
        "target_required": output_kind == "multiclass_margin",
        "target": resolved_target,
    }


def _predict_lightgbm_model(model: Any, X: np.ndarray, target: Optional[int]) -> np.ndarray:
    pred = model.predict(X, raw_score=True)
    return _select_prediction_target(pred, target)
