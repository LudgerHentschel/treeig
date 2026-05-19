from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils import _resolve_binary_classifier_target, _select_prediction_target


def _xgboost_model_kind(model: Any) -> Optional[str]:
    try:
        import xgboost as xgb
    except ImportError:
        return None

    if isinstance(model, xgb.Booster):
        return "booster"

    module = type(model).__module__
    name = type(model).__name__

    if module.startswith("xgboost.") and hasattr(model, "get_booster"):
        if name == "XGBClassifier":
            return "classifier"
        if name == "XGBRegressor":
            return "regressor"
        return "sklearn_like"

    return None


def _xgboost_raw_json(booster: Any) -> Dict[str, Any]:
    """Return XGBoost's internal saved-model JSON."""
    try:
        raw = booster.save_raw(raw_format="json")
    except TypeError:
        import os
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            booster.save_model(path)
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")

    return json.loads(raw)


def _xgboost_internal_trees(model_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract the internal tree list from XGBoost saved-model JSON."""
    try:
        trees = model_json["learner"]["gradient_booster"]["model"]["trees"]
    except KeyError as exc:
        raise ValueError(
            "Could not locate XGBoost trees in saved-model JSON. "
            "The XGBoost JSON schema may have changed."
        ) from exc

    if len(trees) == 0:
        raise ValueError("XGBoost booster contains no trees.")

    return trees


def _xgboost_num_classes_from_config(booster: Any) -> int:
    """Infer the number of classes from XGBoost's config."""
    try:
        cfg = json.loads(booster.save_config())
        learner = cfg.get("learner", {})

        mparam = learner.get("learner_model_param", {})
        if "num_class" in mparam:
            n = int(mparam["num_class"])
            if n > 0:
                return n

        objective = learner.get("objective", {})
        obj_param = objective.get("reg_loss_param", {})
        if "num_class" in obj_param:
            n = int(obj_param["num_class"])
            if n > 0:
                return n

        softmax_param = objective.get("softmax_multiclass_param", {})
        if "num_class" in softmax_param:
            n = int(softmax_param["num_class"])
            if n > 0:
                return n
    except Exception:
        pass

    return 0


def _select_xgboost_tree_indices(
    model: Any,
    n_trees: int,
    target: Optional[int],
) -> Tuple[np.ndarray, str, int, Optional[int], float]:
    """Select XGBoost trees for regression or class-margin attribution."""
    kind = _xgboost_model_kind(model)

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
            raise ValueError("Multiclass XGBClassifier requires target=<class index>.")

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

    if kind == "booster":
        n_classes = _xgboost_num_classes_from_config(model)

        if n_classes and n_classes > 1:
            if target is None:
                raise ValueError(
                    "Multiclass XGBoost Booster requires target=<class index>."
                )

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

    if target not in (None, 0):
        raise ValueError("Regression target must be None or 0.")

    return np.arange(n_trees, dtype=np.int64), "regression", 1, None, 1.0


def _extract_xgboost_booster(model: Any, target: Optional[int]) -> Dict[str, Any]:
    """Extract xgboost.Booster, XGBRegressor, or XGBClassifier."""
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("XGBoost support requires xgboost.") from exc

    kind = _xgboost_model_kind(model)
    if kind is None:
        raise TypeError(f"Expected XGBoost model, got {type(model).__name__}.")

    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        n_features = int(model.n_features_in_)
    elif isinstance(model, xgb.Booster):
        booster = model
        n_features = int(booster.num_features())
    else:
        raise TypeError(f"Expected XGBoost model, got {type(model).__name__}.")

    model_json = _xgboost_raw_json(booster)
    all_trees = _xgboost_internal_trees(model_json)
    all_n_trees = len(all_trees)

    tree_indices, output_kind, n_outputs, resolved_target, sign = (
        _select_xgboost_tree_indices(model, all_n_trees, target)
    )

    selected_trees = [all_trees[int(i)] for i in tree_indices]
    n_trees = len(selected_trees)

    if n_trees == 0:
        raise ValueError("No XGBoost trees selected for the requested target.")

    max_nodes = 0
    for tree in selected_trees:
        try:
            n_nodes = int(tree["tree_param"]["num_nodes"])
        except KeyError:
            n_nodes = len(tree["left_children"])
        max_nodes = max(max_nodes, n_nodes)

    children_left = -np.ones((n_trees, max_nodes), dtype=np.int64)
    children_right = -np.ones((n_trees, max_nodes), dtype=np.int64)
    feature = -np.ones((n_trees, max_nodes), dtype=np.int64)
    threshold = np.zeros((n_trees, max_nodes), dtype=np.float64)
    value = np.zeros((n_trees, max_nodes), dtype=np.float64)

    # XGBoost numeric splits route left if x[j] < threshold.  XGBoost's
    # predictor effectively uses float32 feature values for these comparisons,
    # so thresholds are stored at float32 precision and the core is told to
    # round synthetic path probes through float32 before branch comparisons.
    left_inclusive = np.zeros((n_trees, max_nodes), dtype=np.bool_)
    round_input = np.ones((n_trees, max_nodes), dtype=np.bool_)

    for m, tree in enumerate(selected_trees):
        left_children = np.asarray(tree["left_children"], dtype=np.int64)
        right_children = np.asarray(tree["right_children"], dtype=np.int64)
        split_indices = np.asarray(tree["split_indices"], dtype=np.int64)
        split_conditions = np.asarray(tree["split_conditions"], dtype=np.float64)
        base_weights = np.asarray(tree.get("base_weights", split_conditions), dtype=np.float64)

        n_nodes = left_children.shape[0]

        if right_children.shape[0] != n_nodes:
            raise ValueError("Malformed XGBoost tree: child arrays differ in length.")
        if split_indices.shape[0] != n_nodes:
            raise ValueError("Malformed XGBoost tree: split_indices length mismatch.")
        if split_conditions.shape[0] != n_nodes:
            raise ValueError("Malformed XGBoost tree: split_conditions length mismatch.")
        if base_weights.shape[0] != n_nodes:
            raise ValueError("Malformed XGBoost tree: base_weights length mismatch.")

        if len(tree.get("categories", [])) > 0:
            raise NotImplementedError(
                "TreeIG XGBoost backend currently supports numeric splits only; "
                "categorical splits are not yet supported."
            )

        for node in range(n_nodes):
            lc = int(left_children[node])
            rc = int(right_children[node])

            if lc == -1 and rc == -1:
                children_left[m, node] = -1
                children_right[m, node] = -1

                leaf_val = float(split_conditions[node])
                if not np.isfinite(leaf_val):
                    leaf_val = float(base_weights[node])
                value[m, node] = leaf_val
                continue

            if lc < 0 or rc < 0:
                raise ValueError(
                    "Malformed XGBoost tree: internal node has a missing child."
                )

            children_left[m, node] = lc
            children_right[m, node] = rc
            feature[m, node] = int(split_indices[node])
            threshold[m, node] = float(np.float32(split_conditions[node]))
            value[m, node] = 0.0
            left_inclusive[m, node] = False
            round_input[m, node] = True

    backend = "xgboost_" + (kind if kind != "sklearn_like" else "model")

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


def _predict_xgboost_model(model: Any, X: np.ndarray, target: Optional[int]) -> np.ndarray:
    import xgboost as xgb

    X32 = np.asarray(X, dtype=np.float32)
    if isinstance(model, xgb.Booster):
        pred = model.predict(xgb.DMatrix(X32), output_margin=True)
    else:
        pred = model.predict(X32, output_margin=True)
    return _select_prediction_target(pred, target)
