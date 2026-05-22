"""
TreeIG: Exact Integrated Gradients for Tree Ensembles.
Currently supported backends
----------------------------
Regression:
    sklearn.tree.DecisionTreeRegressor
    sklearn.ensemble.RandomForestRegressor
    sklearn.ensemble.ExtraTreesRegressor
    sklearn.ensemble.GradientBoostingRegressor
    xgboost.XGBRegressor
    xgboost.Booster
    lightgbm.LGBMRegressor
    lightgbm.Booster

Classification, raw margins/logits only:
    sklearn.ensemble.GradientBoostingClassifier
    xgboost.XGBClassifier
    lightgbm.LGBMClassifier

Classification target convention
--------------------------------
For regression models, target must be None or 0.

For additive-score classifiers, TreeIG attributes raw class scores rather
than probabilities, following the standard Integrated Gradients convention
for classification models.

    binary classifiers:
        target=None or target=1 attributes the positive-class margin.
        target=0 attributes the negative margin, implemented as the
        negative of the positive-class margin.

    multiclass classifiers:
        target must select the class-margin output.

Deliberately deferred
---------------------
TreeIG does not currently support probability-output attribution,
missing-value routing, categorical splits, CatBoost, or classifiers that
average probabilities or vote shares directly, such as
DecisionTreeClassifier, RandomForestClassifier, and ExtraTreesClassifier.

Scope
-----
Only finite numeric inputs and baselines are currently supported.
"""
from __future__ import annotations

import time
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .core import (
    _baseline_cache_key,
    _compute_attributions_with_y0,
    _compute_core,
    _compute_event_traces,
    _compute_y0_per_tree,
)
from .dispatch import extract_tree_arrays, model_predict
from .utils import _as_float32_float64, _as_target_key, _check_finite_numeric


class TreeIG:
    """
    Exact Integrated Gradients for tree-based regression and additive-score
    classification models.
    """

    def __init__(
        self,
        model: Any,
        baseline: Optional[np.ndarray] = None,
        time_tol: float = 1e-10,
        tie_policy: str = "first",
        target: Optional[int] = None,
    ):
        if tie_policy != "first":
            raise NotImplementedError(
                "Only tie_policy='first' is currently implemented in the fast "
                "Numba path. The tie_policy argument is reserved for future "
                "allocation rules for coincident active crossings."
            )

        self.model = model
        self.time_tol = float(time_tol)
        self.tie_policy = tie_policy
        self.target = _as_target_key(target)
        self._arrays_by_target: Dict[Optional[int], Dict[str, Any]] = {}
        self._y0_cache: Dict[Tuple[str, Optional[int], bytes], np.ndarray] = {}
        self._baseline_prediction_cache: Dict[
            Tuple[str, Optional[int], bytes], float
        ] = {}

        arrays = extract_tree_arrays(model, self.target)
        self._arrays = arrays
        self._arrays_by_target[self.target] = arrays
        self.n_features_in_ = int(arrays["n_features"])
        self.backend = arrays.get("backend", "unknown")

        if baseline is None:
            self._baseline = None
        else:
            self._baseline = self._prepare_baseline(baseline)

    @staticmethod
    def _round_inputs_for_arrays(arrays: Dict[str, Any]) -> bool:
        round_input = arrays.get("round_input", None)
        return bool(round_input is not None and np.asarray(round_input).any())

    def _prepare_baseline(
        self,
        b: np.ndarray,
        arrays: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        b = np.asarray(b, dtype=np.float64)
        n = self.n_features_in_

        if b.ndim != 1 or b.shape[0] != n:
            raise ValueError(f"baseline must have shape ({n},), got {b.shape}.")

        _check_finite_numeric(b, "baseline")
        arrays = self._arrays if arrays is None else arrays
        if self._round_inputs_for_arrays(arrays):
            return _as_float32_float64(b)
        return np.ascontiguousarray(b, dtype=np.float64)

    def _prepare_X(
        self,
        X: np.ndarray,
        arrays: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n = self.n_features_in_

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")

        if X.shape[1] != n:
            raise ValueError(f"X has {X.shape[1]} features; model expects {n}.")

        _check_finite_numeric(X, "X")
        arrays = self._arrays if arrays is None else arrays
        if self._round_inputs_for_arrays(arrays):
            return _as_float32_float64(X)
        return np.ascontiguousarray(X, dtype=np.float64)

    def _resolve_baseline(
        self,
        baseline: Optional[np.ndarray],
        arrays: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        if baseline is not None:
            return self._prepare_baseline(baseline, arrays)

        if self._baseline is not None:
            return self._baseline

        raise ValueError(
            "A baseline is required. Pass baseline= to this method, or set a "
            "default at construction: TreeIG(model, baseline=x0)."
        )

    def _resolve_arrays_for_target(self, target: Optional[int]) -> Dict[str, Any]:
        target_key = self.target if target is None else _as_target_key(target)

        if target_key in self._arrays_by_target:
            return self._arrays_by_target[target_key]

        arrays = extract_tree_arrays(self.model, target_key)
        self._arrays_by_target[target_key] = arrays
        return arrays

    def _get_y0_per_tree(self, arrays: Dict[str, Any], baseline: np.ndarray) -> np.ndarray:
        key = _baseline_cache_key(arrays, baseline)
        cached = self._y0_cache.get(key)
        if cached is not None:
            return cached

        y0 = _compute_y0_per_tree(arrays, baseline)
        self._y0_cache[key] = y0
        return y0

    def _get_baseline_prediction(self, arrays: Dict[str, Any], baseline: np.ndarray) -> float:
        key = _baseline_cache_key(arrays, baseline)
        cached = self._baseline_prediction_cache.get(key)
        if cached is not None:
            return cached

        resolved_target = arrays.get("target", None)
        pred0 = float(model_predict(self.model, baseline.reshape(1, -1), resolved_target)[0])
        self._baseline_prediction_cache[key] = pred0
        return pred0

    @staticmethod
    def _validate_batch_size(batch_size: Optional[int]) -> Optional[int]:
        if batch_size is None:
            return None
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive or None, got {batch_size}.")
        return batch_size

    def _attribute_prepared(
        self,
        arrays: Dict[str, Any],
        baseline: np.ndarray,
        X: np.ndarray,
        y0_per_tree: np.ndarray,
        batch_size: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = self._validate_batch_size(batch_size)
        n_obs = X.shape[0]

        if batch_size is None or n_obs <= batch_size:
            return _compute_attributions_with_y0(
                arrays,
                baseline,
                X,
                self.time_tol,
                y0_per_tree,
            )

        phis = np.empty((n_obs, X.shape[1]), dtype=np.float64)
        event_counts = np.empty(n_obs, dtype=np.int64)

        for start in range(0, n_obs, batch_size):
            stop = min(start + batch_size, n_obs)
            phi_chunk, event_chunk = _compute_attributions_with_y0(
                arrays,
                baseline,
                X[start:stop],
                self.time_tol,
                y0_per_tree,
            )
            phis[start:stop] = phi_chunk
            event_counts[start:stop] = event_chunk

        return phis, event_counts

    def attribute(
        self,
        X: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        target: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute feature attributions.

        This is the fastest public path: it does not compute residual
        diagnostics or call model.predict().
        """
        arrays = self._resolve_arrays_for_target(target)
        b = self._resolve_baseline(baseline, arrays)
        X_prep = self._prepare_X(X, arrays)
        y0 = self._get_y0_per_tree(arrays, b)

        phis, _ = self._attribute_prepared(arrays, b, X_prep, y0, batch_size)
        return phis

    def explain(
        self,
        X: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        target: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        """Compute attributions with per-observation diagnostics."""
        arrays = self._resolve_arrays_for_target(target)
        b = self._resolve_baseline(baseline, arrays)
        X_prep = self._prepare_X(X, arrays)
        y0 = self._get_y0_per_tree(arrays, b)

        resolved_target = arrays.get("target", None)
        y0_scalar = self._get_baseline_prediction(arrays, b)
        endpoint_delta = model_predict(self.model, X_prep, resolved_target) - y0_scalar

        batch_size = self._validate_batch_size(batch_size)
        if batch_size is None or X_prep.shape[0] <= batch_size:
            return _compute_core(arrays, b, X_prep, self.time_tol, y0, endpoint_delta)

        phis, event_counts = self._attribute_prepared(
            arrays,
            b,
            X_prep,
            y0,
            batch_size,
        )
        residuals = phis.sum(axis=1) - endpoint_delta

        infos = [
            {
                "n_events": int(event_counts[i]),
                "endpoint_delta": float(endpoint_delta[i]),
                "attribution_sum": float(phis[i].sum()),
                "residual": float(residuals[i]),
                "abs_residual": float(abs(residuals[i])),
            }
            for i in range(X_prep.shape[0])
        ]

        summary = {
            "mean_abs_residual": float(np.mean(np.abs(residuals))),
            "median_abs_residual": float(np.median(np.abs(residuals))),
            "max_abs_residual": float(np.max(np.abs(residuals))),
            "mean_events": float(np.mean(event_counts)),
            "median_events": float(np.median(event_counts)),
            "max_events": int(np.max(event_counts)),
        }

        return phis, infos, summary

    def trace(
        self,
        X: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        target: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Return ordered split-crossing prediction events.

        This is a low-level API for downstream scalar-functional attribution,
        including EDEF. It returns per-observation path events, not feature
        attribution sums.

        Returned arrays are padded to a common width. For observation i, only
        the first counts[i] entries are valid.
        """
        arrays = self._resolve_arrays_for_target(target)
        b = self._resolve_baseline(baseline, arrays)
        X_prep = self._prepare_X(X, arrays)
        y0 = self._get_y0_per_tree(arrays, b)

        counts, times, features, jumps = _compute_event_traces(
            arrays,
            b,
            X_prep,
            self.time_tol,
            y0,
        )

        resolved_target = arrays.get("target", None)
        baseline_prediction = self._get_baseline_prediction(arrays, b)
        endpoint_prediction = model_predict(self.model, X_prep, resolved_target)

        return {
            "counts": counts,
            "times": times,
            "features": features,
            "jumps": jumps,
            "baseline_prediction": baseline_prediction,
            "endpoint_prediction": endpoint_prediction,
            "baseline": b,
            "target": resolved_target,
        }
        
    def warmup(
        self,
        X: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        target: Optional[int] = None,
    ):
        """Trigger Numba JIT compilation on a small sample and cache y0."""
        arrays = self._resolve_arrays_for_target(target)
        b = self._resolve_baseline(baseline, arrays)
        X_prep = self._prepare_X(X, arrays)
        y0 = self._get_y0_per_tree(arrays, b)

        _compute_attributions_with_y0(
            arrays,
            b,
            X_prep[: min(2, len(X_prep))],
            self.time_tol,
            y0,
        )
        return self


def compute(
    model: Any,
    baseline: np.ndarray,
    X: np.ndarray,
    time_tol: float = 1e-10,
    tie_policy: str = "first",
    target: Optional[int] = None,
    batch_size: Optional[int] = None,
):
    """Functional interface to TreeIG; returns phis, infos, summary."""
    return TreeIG(
        model,
        baseline=baseline,
        time_tol=time_tol,
        tie_policy=tie_policy,
        target=target,
    ).explain(X, batch_size=batch_size)


def timed_call(fn, *args, **kwargs):
    """Return (result, elapsed_seconds) for any callable."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def exact_gb_ig_batch_fast(
    model: Any,
    x0: np.ndarray,
    X: np.ndarray,
    tol: float = 1e-10,
    boundary_tol: float = 1e-8,
    tie_policy: str = "first",
    target: Optional[int] = None,
):
    """Backward-compatible alias for compute(); boundary_tol is ignored."""
    warnings.warn(
        "exact_gb_ig_batch_fast is deprecated; use TreeIG(...).explain(...) "
        "or treeig.compute(...).",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute(
        model,
        x0,
        X,
        time_tol=tol,
        tie_policy=tie_policy,
        target=target,
    )


def warmup_exact_gb_ig(
    model: Any,
    x0: np.ndarray,
    X: np.ndarray,
    tol: float = 1e-10,
    boundary_tol: float = 1e-8,
    tie_policy: str = "first",
    target: Optional[int] = None,
):
    """
    Backward-compatible warmup alias.

    Returns the same object shape as the old helper: phis, infos, summary for
    up to the first two observations. boundary_tol is ignored.
    """
    warnings.warn(
        "warmup_exact_gb_ig is deprecated; use TreeIG(...).warmup(...).",
        DeprecationWarning,
        stacklevel=2,
    )
    X_arr = np.asarray(X)
    n = min(2, X_arr.shape[0])
    return compute(
        model,
        x0,
        X_arr[:n],
        time_tol=tol,
        tie_policy=tie_policy,
        target=target,
    )


extract_gb_tree_arrays = extract_tree_arrays

