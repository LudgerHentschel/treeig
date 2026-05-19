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
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .core import (
    _baseline_cache_key,
    _compute_attributions_with_y0,
    _compute_core,
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

        arrays = extract_tree_arrays(model, self.target)
        self._arrays = arrays
        self._arrays_by_target[self.target] = arrays
        self.n_features_in_ = int(arrays["n_features"])
        self.backend = arrays.get("backend", "unknown")

        if baseline is None:
            self._baseline = None
        else:
            self._baseline = self._prepare_baseline(baseline)

    def _prepare_baseline(self, b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, dtype=np.float64)
        n = self.n_features_in_

        if b.ndim != 1 or b.shape[0] != n:
            raise ValueError(f"baseline must have shape ({n},), got {b.shape}.")

        _check_finite_numeric(b, "baseline")
        return _as_float32_float64(b)

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n = self.n_features_in_

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")

        if X.shape[1] != n:
            raise ValueError(f"X has {X.shape[1]} features; model expects {n}.")

        _check_finite_numeric(X, "X")
        return _as_float32_float64(X)

    def _resolve_baseline(self, baseline: Optional[np.ndarray]) -> np.ndarray:
        if baseline is not None:
            return self._prepare_baseline(baseline)

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

    def attribute(
        self,
        X: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        target: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute feature attributions.

        This is the fastest public path: it does not compute residual
        diagnostics or call model.predict().
        """
        b = self._resolve_baseline(baseline)
        X_prep = self._prepare_X(X)
        arrays = self._resolve_arrays_for_target(target)
        y0 = self._get_y0_per_tree(arrays, b)

        phis, _ = _compute_attributions_with_y0(arrays, b, X_prep, self.time_tol, y0)
        return phis

    def explain(
        self,
        X: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        target: Optional[int] = None,
    ):
        """Compute attributions with per-observation diagnostics."""
        b = self._resolve_baseline(baseline)
        X_prep = self._prepare_X(X)
        arrays = self._resolve_arrays_for_target(target)
        y0 = self._get_y0_per_tree(arrays, b)

        resolved_target = arrays.get("target", None)
        endpoint_delta = model_predict(self.model, X_prep, resolved_target) - model_predict(
            self.model, b.reshape(1, -1), resolved_target
        )[0]

        return _compute_core(arrays, b, X_prep, self.time_tol, y0, endpoint_delta)

    def warmup(
        self,
        X: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        target: Optional[int] = None,
    ):
        """Trigger Numba JIT compilation on a small sample and cache y0."""
        b = self._resolve_baseline(baseline)
        X_prep = self._prepare_X(X)
        arrays = self._resolve_arrays_for_target(target)
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
):
    """Functional interface to TreeIG; returns phis, infos, summary."""
    return TreeIG(
        model,
        baseline=baseline,
        time_tol=time_tol,
        tie_policy=tie_policy,
        target=target,
    ).explain(X)


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
