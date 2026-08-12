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
    _compute_weighted_attributions,
    _compute_y0_per_tree,
    _compute_y0_per_baseline_tree,
    _loss_reduction_batch,
    _loss_reduction_weighted_batch,
    _loss_reduction_multiclass_from_traces_batch, 
    _loss_reduction_weighted_multiclass_trees,
)
from .dispatch import extract_tree_arrays, model_predict
from .utils import _as_float32_float64, _as_target_key, _check_finite_numeric


def _mean_standard_errors(values: np.ndarray) -> np.ndarray:
    n_obs = values.shape[0]
    if n_obs < 2:
        return np.full(values.shape[1], np.nan, dtype=np.float64)
    return values.std(axis=0, ddof=1) / np.sqrt(n_obs)


class TreeIG:
    """
    Exact integrated-gradient attribution for supported tree-based models.

    TreeIG computes feature attributions for fitted tree ensembles by exactly
    summing the prediction jumps encountered along the straight-line path from
    a baseline input to each evaluation input. For supported tree models, this
    avoids numerical quadrature and produces an additive decomposition of the
    model-output difference.

    For an input row ``x`` and baseline ``x0``, the returned attributions
    satisfy, up to floating-point error,

        attributions.sum() = f(x) - f(x0)

    where ``f`` is the selected scalar model output.

    Regression models use their scalar prediction output. Supported
    classification models are attributed on raw class scores, margins, or
    logits, not probabilities. Binary classifiers use the positive-class
    margin by default; ``target=0`` attributes the negative of that margin.
    Multiclass classifiers require a selected class target.

    Parameters
    ----------
    model : object
        Fitted supported tree-based model. Supported regression backends
        include scikit-learn decision trees, random forests, extra-trees
        regressors, gradient boosting regressors, XGBoost regressors and
        boosters, and LightGBM regressors and boosters. Supported
        classification backends are additive-score classifiers with raw
        margin/logit outputs.

    baseline : array-like of shape (n_features,), optional
        Default baseline input ``x0``. If omitted, a baseline must be supplied
        to methods that compute attributions.

    time_tol : float, default=1e-10
        Tolerance used when ordering path-crossing times along the straight
        line from the baseline to each input row.

    tie_policy : {"first"}, default="first"
        Rule for coincident active split crossings. Only ``"first"`` is
        currently implemented. The argument is reserved for future allocation
        rules.

    target : int or None, default=None
        Scalar model-output target. For regression, use ``None`` or ``0``.
        For binary additive-score classification, ``None`` or ``1`` selects
        the positive-class margin and ``0`` selects the negative margin. For
        multiclass classification, this selects the class-margin output.

    Attributes
    ----------
    model : object
        The fitted model passed at construction.

    n_features_in_ : int
        Number of input features expected by the extracted tree representation.

    backend : str
        Backend identifier inferred from the fitted model.

    Notes
    -----
    TreeIG currently requires finite numeric inputs and finite numeric
    baselines. Missing-value routing, categorical splits, probability-output
    attribution, CatBoost models, and classifiers that average probabilities
    or vote shares directly are not currently supported.
    """

    def __init__(
        self,
        model: Any,
        baseline: Optional[np.ndarray] = None,
        baseline_weights: Optional[np.ndarray] = None,
        time_tol: float = 1e-10,
        tie_policy: str = "first",
        target: Optional[int] = None,
    ):
        """
        Initialize a TreeIG attribution object.
    
        Parameters
        ----------
        model : object
            Fitted supported tree-based model.
    
        baseline : array-like of shape (n_features,), optional
            Default baseline input used for attribution paths. If omitted, a
            baseline must be supplied to attribution methods.
    
        time_tol : float, default=1e-10
            Tolerance used when ordering split-crossing events along the
            attribution path.
    
        tie_policy : {"first"}, default="first"
            Rule for coincident active split crossings. Only ``"first"`` is
            currently implemented.
    
        target : int or None, default=None
            Default scalar model-output target. See ``TreeIG`` for the
            regression and classification target conventions.
        """
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

        self._baseline_weights = baseline_weights
        if baseline is None:
            self._baseline = None
        else:
            self._baseline, self._baseline_weights = self._prepare_baselines(
                baseline, baseline_weights
            )
            if self._baseline.shape[0] == 1:
                self._baseline = self._baseline[0]

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

    def _prepare_baselines(self, baselines, weights=None, arrays=None):
        if hasattr(baselines, "rows") and hasattr(baselines, "weights"):
            if weights is not None:
                raise ValueError(
                    "baseline_weights must be omitted when baseline supplies weights."
                )
            weights = baselines.weights
            baselines = baselines.rows
        b = np.asarray(baselines, dtype=np.float64)
        if b.ndim == 1:
            b = b.reshape(1, -1)
        if b.ndim != 2 or b.shape[1] != self.n_features_in_ or b.shape[0] == 0:
            raise ValueError(
                "baseline must have shape (n_features,) or "
                f"(n_baselines, {self.n_features_in_}), got {b.shape}."
            )
        b = np.stack([self._prepare_baseline(row, arrays) for row in b])
        if weights is None:
            w = np.full(b.shape[0], 1.0 / b.shape[0])
        else:
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
            if w.shape[0] != b.shape[0]:
                raise ValueError("baseline_weights must align with baseline rows.")
            if not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
                raise ValueError(
                    "baseline_weights must be finite, nonnegative, and have "
                    "positive total mass."
                )
            w = w / w.sum()
        return np.ascontiguousarray(b), np.ascontiguousarray(w)

    def _resolve_baselines(self, baseline, baseline_weights, arrays=None):
        if baseline is not None:
            return self._prepare_baselines(baseline, baseline_weights, arrays)
        if self._baseline is None:
            raise ValueError(
                "A baseline is required. Pass baseline= to this method, or "
                "set a default baseline when constructing TreeIG."
            )
        return self._prepare_baselines(
            self._baseline, self._baseline_weights if baseline_weights is None else baseline_weights,
            arrays,
        )

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
        baseline_weights: Optional[np.ndarray] = None,
        target: Optional[int] = None,
        batch_size: Optional[int] = None,
        baseline_batch_size: Optional[int] = None,
        return_by_baseline: bool = False,
    ) -> np.ndarray:
        """
        Compute exact feature attributions for one or more input rows.

        This is the fastest public attribution method. It computes feature
        attributions only and does not call ``model.predict`` to construct
        endpoint diagnostics.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Evaluation inputs. All entries must be finite numeric values.

        baseline : array-like of shape (n_features,), optional
            Baseline input used for the path integration. If omitted, the
            default baseline supplied at construction is used.

        target : int or None, default=None
            Scalar model-output target to attribute. If omitted, the target
            supplied at construction is used. See ``TreeIG`` for the regression
            and classification target conventions.

        batch_size : int or None, default=None
            Number of rows to process per batch. If ``None``, all rows are
            processed in one call. Use a positive integer to reduce peak memory
            use for large ``X``.

        Returns
        -------
        attributions : ndarray of shape (n_samples, n_features)
            Exact integrated-gradient feature attributions. For each row,
            ``attributions[i].sum()`` equals the selected model output at
            ``X[i]`` minus the selected model output at the baseline, up to
            floating-point error.

        Raises
        ------
        ValueError
            If no baseline is available, if ``X`` has an incompatible shape,
            or if ``X`` or the baseline contains non-finite values.

        Examples
        --------
        >>> import numpy as np
        >>> from treeig import TreeIG
        >>>
        >>> x0 = np.mean(X_train, axis=0)
        >>> explainer = TreeIG(model, baseline=x0)
        >>> attributions = explainer.attribute(X_test)
        >>> attributions.shape
        (X_test.shape[0], X_test.shape[1])
        """
        arrays = self._resolve_arrays_for_target(target)
        baselines, weights = self._resolve_baselines(
            baseline, baseline_weights, arrays
        )
        X_prep = self._prepare_X(X, arrays)
        baseline_batch_size = self._validate_batch_size(baseline_batch_size)
        if baseline_batch_size is None:
            baseline_batch_size = baselines.shape[0]
        weighted = np.zeros((X_prep.shape[0], X_prep.shape[1]), dtype=float)
        by_baseline = [] if return_by_baseline else None
        if not return_by_baseline:
            target_batch = self._validate_batch_size(batch_size) or X_prep.shape[0]
            for x_start in range(0, X_prep.shape[0], target_batch):
                x_stop = min(x_start + target_batch, X_prep.shape[0])
                for start in range(0, baselines.shape[0], baseline_batch_size):
                    stop = min(start + baseline_batch_size, baselines.shape[0])
                    y0 = np.stack([
                        self._get_y0_per_tree(arrays, baselines[i])
                        for i in range(start, stop)
                    ])
                    phi, _ = _compute_weighted_attributions(
                        arrays, baselines[start:stop], weights[start:stop],
                        X_prep[x_start:x_stop], self.time_tol, y0,
                    )
                    weighted[x_start:x_stop] += phi
            return weighted
        for start in range(0, baselines.shape[0], baseline_batch_size):
            stop = min(start + baseline_batch_size, baselines.shape[0])
            for index in range(start, stop):
                b = baselines[index]
                y0 = self._get_y0_per_tree(arrays, b)
                phi, _ = self._attribute_prepared(
                    arrays, b, X_prep, y0, batch_size
                )
                weighted += weights[index] * phi
                if return_by_baseline:
                    by_baseline.append(phi)
        if return_by_baseline:
            return weighted, np.stack(by_baseline, axis=0)
        return weighted

    def model_output(
        self,
        X: np.ndarray,
        target: Optional[int] = None,
    ) -> np.ndarray:
        """Return the scalar model output that TreeIG attributes."""
        arrays = self._resolve_arrays_for_target(target)
        X_prep = self._prepare_X(X, arrays)
        return np.asarray(
            model_predict(self.model, X_prep, arrays.get("target", None)),
            dtype=np.float64,
        )

    def explain(
        self,
        X: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        baseline_weights: Optional[np.ndarray] = None,
        target: Optional[int] = None,
        batch_size: Optional[int] = None,
        baseline_batch_size: Optional[int] = None,
    ):
        """
        Compute exact feature attributions with additivity diagnostics.

        This method returns feature attributions together with per-row and
        aggregate diagnostics comparing the attribution sums to endpoint model
        output differences.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Evaluation inputs. All entries must be finite numeric values.

        baseline : array-like of shape (n_features,), optional
            Baseline input used for the path integration. If omitted, the
            default baseline supplied at construction is used.

        target : int or None, default=None
            Scalar model-output target to attribute. If omitted, the target
            supplied at construction is used.

        batch_size : int or None, default=None
            Number of rows to process per batch. If ``None``, all rows are
            processed in one call.

        Returns
        -------
        attributions : ndarray of shape (n_samples, n_features)
            Exact integrated-gradient feature attributions.

        infos : list of dict
            Per-observation diagnostics. Each dictionary contains:

            ``"n_events"``
                Number of split-crossing events on the path.

            ``"endpoint_delta"``
                Selected model output at the input row minus the selected
                model output at the baseline.

            ``"attribution_sum"``
                Sum of feature attributions for the row.

            ``"residual"``
                ``attribution_sum - endpoint_delta``.

            ``"abs_residual"``
                Absolute value of ``residual``.

        summary : dict
            Aggregate diagnostics containing mean, median, and maximum
            absolute residuals, together with mean, median, and maximum event
            counts.

        Notes
        -----
        ``explain`` is useful for validation, testing, and reporting. For
        production attribution calls where diagnostics are not needed,
        ``attribute`` is faster.
        """
        arrays = self._resolve_arrays_for_target(target)
        baselines, weights = self._resolve_baselines(
            baseline, baseline_weights, arrays
        )
        X_prep = self._prepare_X(X, arrays)
        resolved_target = arrays.get("target", None)
        baseline_prediction = sum(
            weights[i] * self._get_baseline_prediction(arrays, baselines[i])
            for i in range(baselines.shape[0])
        )
        endpoint_delta = (
            model_predict(self.model, X_prep, resolved_target)
            - baseline_prediction
        )
        phis = self.attribute(
            X_prep, baseline=baselines, baseline_weights=weights,
            target=target, batch_size=batch_size,
            baseline_batch_size=baseline_batch_size,
        )
        event_counts = np.zeros(X_prep.shape[0], dtype=np.float64)
        for i, b in enumerate(baselines):
            y0 = self._get_y0_per_tree(arrays, b)
            _, counts = self._attribute_prepared(
                arrays, b, X_prep, y0, batch_size
            )
            event_counts += weights[i] * counts
        residuals = phis.sum(axis=1) - endpoint_delta

        infos = [
            {
                "n_events": float(event_counts[i]),
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
            "max_events": float(np.max(event_counts)),
            "baseline_prediction": float(baseline_prediction),
            "n_baselines": int(baselines.shape[0]),
        }

        return phis, infos, summary

    def trace(
        self,
        X: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        target: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Return ordered split-crossing events along each attribution path.

        This advanced method exposes the event representation used internally
        by TreeIG. It is intended for downstream scalar-functional
        attribution, including loss-based decompositions. It returns path
        events rather than feature attribution sums.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Evaluation inputs.

        baseline : array-like of shape (n_features,) or (n_baselines, n_features), optional
            Baseline input or baseline distribution. If omitted, the default
            baseline supplied at construction is used.

        baseline_weights : array-like of shape (n_baselines,), optional
            Nonnegative baseline weights. They are normalized internally.
            Omit for a single baseline or equal weighting.

        target : int or None, default=None
            Scalar model-output target.

        Returns
        -------
        trace : dict
            Dictionary with the following entries:

            ``"counts"`` : ndarray of shape (n_samples,)
                Number of valid events for each observation.

            ``"times"`` : ndarray of shape (n_samples, max_events)
                Path-crossing times. For row ``i``, only the first
                ``counts[i]`` entries are valid.

            ``"features"`` : ndarray of shape (n_samples, max_events)
                Feature index associated with each valid split-crossing event.

            ``"jumps"`` : ndarray of shape (n_samples, max_events)
                Prediction jump associated with each valid event.

            ``"baseline_prediction"`` : float
                Selected model output at the baseline.

            ``"endpoint_prediction"`` : ndarray of shape (n_samples,)
                Selected model output at each input row.

            ``"baseline"`` : ndarray of shape (n_features,)
                Prepared baseline used in the computation.

            ``"target"`` : int or None
                Resolved target used in the computation.

        Notes
        -----
        Returned event arrays are padded to a common width. Padding entries
        beyond ``counts[i]`` are not part of the path for observation ``i``.
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
    
    def loss_attribution(
        self,
        X: np.ndarray,
        y: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        baseline_weights: Optional[np.ndarray] = None,
        loss: str = "squared_error",
        target: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Attribute average loss reduction to input features.

        This method decomposes the improvement in loss from the baseline
        prediction to the model prediction. The resulting feature values sum
        to average baseline loss minus average model loss, up to
        floating-point error.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Evaluation inputs.

        y : array-like of shape (n_samples,)
            Observed outcomes or labels. For ``loss="squared_error"``, values
            are interpreted as numeric outcomes. For ``loss="log_loss"``,
            values must be binary labels in ``{0, 1}``.

        baseline : array-like of shape (n_features,) or (n_baselines, n_features), optional
            Baseline input or distribution. If omitted, the default baseline
            supplied at construction is used.

        baseline_weights : array-like of shape (n_baselines,), optional
            Nonnegative weights, normalized internally. If omitted, multiple
            baseline rows receive equal weight.

        loss : {"squared_error", "log_loss"}, default="squared_error"
            Loss function used to measure improvement. ``"log_loss"`` is
            currently for binary classification.

        target : int or None, default=None
            Scalar model-output target. For binary log-loss, this should
            select the binary margin convention used by the fitted model.

        batch_size : int or None, default=None
            Currently unused. Present for API consistency with 
            attribution methods.

        Returns
        -------
        result : dict
            Dictionary with the following entries:

            ``"observation_values"`` : ndarray of shape (n_samples, n_features)
                Per-observation feature contributions to loss reduction.

            ``"values"`` : ndarray of shape (n_features,)
                Average feature contributions to loss reduction.

            ``"standard_errors"`` : ndarray of shape (n_features,)
                Standard errors of the average feature contributions, computed
                across observations.

            ``"baseline_loss"`` : float
                Average loss at the baseline prediction.

            ``"model_loss"`` : float
                Average loss at the endpoint model predictions.

            ``"total"`` : float
                ``baseline_loss - model_loss``.

            ``"baseline_prediction"`` : float
                Selected model output at the baseline.

            ``"endpoint_prediction"`` : ndarray of shape (n_samples,)
                Selected model output at each input row.

            ``"loss"`` : str
                Name of the loss function.

        Notes
        -----
        Positive values indicate features that reduce loss on average relative
        to the baseline prediction. Negative values indicate features that
        increase loss on average.

        Examples
        --------
        >>> result = explainer.loss_attribution(X_test, y_test)
        >>> result["values"]
        >>> result["total"]
        """
        if loss not in {"squared_error", "log_loss"}:
            raise NotImplementedError(
                "Only squared_error and binary log_loss are implemented initially."
            )

        arrays = self._resolve_arrays_for_target(target)
        baselines, weights = self._resolve_baselines(
            baseline, baseline_weights, arrays
        )
        X_prep = self._prepare_X(X, arrays)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    
        if loss == "log_loss":
            if not np.all((y_arr == 0.0) | (y_arr == 1.0)):
                raise ValueError(
                    "For log_loss, y must contain only binary labels in {0, 1}."
                )

        if y_arr.shape[0] != X_prep.shape[0]:
            raise ValueError("y and X must have the same number of observations.")
    
        resolved_target = arrays.get("target", None)
        baseline_predictions = np.asarray(
            model_predict(self.model, baselines, resolved_target), dtype=float
        ).reshape(-1)
        endpoint_prediction = model_predict(self.model, X_prep, resolved_target)
        if baselines.shape[0] == 1:
            y0 = self._get_y0_per_tree(arrays, baselines[0])
            obs_values, baseline_losses, model_losses = _loss_reduction_batch(
                arrays, baselines[0], X_prep, y_arr, self.time_tol, y0,
                baseline_predictions[0], endpoint_prediction, loss,
            )
        else:
            y0 = _compute_y0_per_baseline_tree(arrays, baselines)
            obs_values, baseline_losses, model_losses = (
                _loss_reduction_weighted_batch(
                    arrays, baselines, weights, X_prep, y_arr, self.time_tol,
                    y0, baseline_predictions, endpoint_prediction, loss,
                )
            )

        baseline_prediction = float(np.dot(weights, baseline_predictions))
    
        return {
            "observation_values": obs_values,
            "values": obs_values.mean(axis=0),
            "standard_errors": _mean_standard_errors(obs_values),
            "baseline_loss": float(np.mean(baseline_losses)),
            "model_loss": float(np.mean(model_losses)),
            "total": float(np.mean(baseline_losses) - np.mean(model_losses)),
            "baseline_prediction": baseline_prediction,
            "baseline_predictions": baseline_predictions,
            "baseline_weights": weights,
            "endpoint_prediction": endpoint_prediction,
            "loss": loss,
        }
        
    def multiclass_loss_attribution(
        self,
        X: np.ndarray,
        y: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        baseline_weights: Optional[np.ndarray] = None,
        n_classes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Attribute multiclass log-loss reduction to input features.

        This method decomposes the improvement in multiclass log loss from
        baseline class scores to endpoint class scores. It computes traces for
        each class-margin output and combines them into a loss-reduction
        attribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Evaluation inputs.

        y : array-like of shape (n_samples,)
            Integer class labels. Labels must be nonnegative. The labels are
            interpreted as class indices.

        baseline : array-like of shape (n_features,) or (n_baselines, n_features), optional
            Baseline input or distribution. If omitted, the default baseline
            supplied at construction is used.

        baseline_weights : array-like of shape (n_baselines,), optional
            Nonnegative weights, normalized internally.

        n_classes : int or None, default=None
            Number of classes. If omitted, ``len(model.classes_)`` is used.
            Provide this explicitly for models that do not expose
            ``classes_``.

        Returns
        -------
        result : dict
            Dictionary with the following entries:

            ``"observation_values"`` : ndarray of shape (n_samples, n_features)
                Per-observation feature contributions to multiclass log-loss
                reduction.

            ``"values"`` : ndarray of shape (n_features,)
                Average feature contributions.

            ``"standard_errors"`` : ndarray of shape (n_features,)
                Standard errors of the average feature contributions.

            ``"baseline_loss"`` : float
                Average multiclass log loss at the baseline class scores.

            ``"model_loss"`` : float
                Average multiclass log loss at the endpoint class scores.

            ``"total"`` : float
                ``baseline_loss - model_loss``.

            ``"loss"`` : str
                Equal to ``"multiclass_log_loss"``.

        Notes
        -----
        This method uses raw class-score traces, not probability-output
        attributions. Positive values indicate features that reduce multiclass
        log loss on average relative to the baseline class scores.
        """
        X_prep = self._prepare_X(X)
        y_arr = np.asarray(y, dtype=np.int64).reshape(-1)

        if y_arr.shape[0] != X_prep.shape[0]:
            raise ValueError("y and X must have the same number of observations.")

        if np.any(y_arr < 0):
            raise ValueError("y must contain nonnegative class labels.")

        if n_classes is None:
            classes = getattr(self.model, "classes_", None)

            if classes is None:
                raise ValueError(
                    "n_classes must be provided if model has no classes_."
                )

            n_classes = int(len(classes))

        baselines, weights = self._resolve_baselines(
            baseline, baseline_weights, self._arrays
        )
        if baselines.shape[0] > 1:
            arrays_by_class = [
                self._resolve_arrays_for_target(k) for k in range(n_classes)
            ]
            y0 = np.stack([
                _compute_y0_per_baseline_tree(arrays_k, baselines)
                for arrays_k in arrays_by_class
            ], axis=1)
            baseline_scores = np.column_stack([
                model_predict(self.model, baselines, k)
                for k in range(n_classes)
            ])
            endpoint_scores = np.column_stack([
                model_predict(self.model, X_prep, k)
                for k in range(n_classes)
            ])
            observation_values, baseline_losses, model_losses = (
                _loss_reduction_weighted_multiclass_trees(
                    arrays_by_class, baselines, weights, X_prep, y_arr, y0,
                    baseline_scores, endpoint_scores, self.time_tol,
                )
            )
            baseline_loss = float(np.mean(baseline_losses))
            model_loss = float(np.mean(model_losses))
            return {
                "observation_values": observation_values,
                "values": observation_values.mean(axis=0),
                "standard_errors": _mean_standard_errors(observation_values),
                "baseline_loss": baseline_loss,
                "model_loss": model_loss,
                "total": float(baseline_loss - model_loss),
                "baseline_scores": baseline_scores,
                "baseline_weights": weights,
                "loss": "multiclass_log_loss",
            }
        baseline = baselines[0]

        baseline_scores = np.empty(n_classes, dtype=np.float64)
        endpoint_scores = np.empty(
            (X_prep.shape[0], n_classes),
            dtype=np.float64,
        )

        trace_list = []

        for k in range(n_classes):
            arrays_k = self._resolve_arrays_for_target(k)
            b_k = self._resolve_baseline(baseline, arrays_k)
            y0_k = self._get_y0_per_tree(arrays_k, b_k)

            counts_k, times_k, features_k, jumps_k = _compute_event_traces(
                arrays_k,
                b_k,
                X_prep,
                self.time_tol,
                y0_k,
            )

            baseline_scores[k] = self._get_baseline_prediction(arrays_k, b_k)

            endpoint_scores[:, k] = model_predict(
                self.model,
                X_prep,
                k,
            )

            trace_list.append(
                (
                    counts_k,
                    times_k,
                    features_k,
                    jumps_k,
                )
            )

        counts = np.stack([x[0] for x in trace_list], axis=0)
        times = np.stack([x[1] for x in trace_list], axis=0)
        features = np.stack([x[2] for x in trace_list], axis=0)
        jumps = np.stack([x[3] for x in trace_list], axis=0)

        (
            observation_values,
            baseline_losses,
            model_losses,
        ) = _loss_reduction_multiclass_from_traces_batch(
            counts,
            times,
            features,
            jumps,
            y_arr,
            baseline_scores,
            endpoint_scores,
            X_prep.shape[1],
        )

        values = observation_values.mean(axis=0)

        return {
            "observation_values": observation_values,
            "values": values,
            "standard_errors": _mean_standard_errors(observation_values),
            "baseline_loss": float(np.mean(baseline_losses)),
            "model_loss": float(np.mean(model_losses)),
            "total": float(
                np.mean(baseline_losses) - np.mean(model_losses)
            ),
            "loss": "multiclass_log_loss",
        }


    def warmup(
        self,
        X: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        target: Optional[int] = None,
    ):
        """
        Trigger JIT compilation and cache baseline tree outputs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Sample inputs used to trigger compilation. Only up to the first
            two rows are used.

        baseline : array-like of shape (n_features,), optional
            Baseline input. If omitted, the default baseline supplied at
            construction is used.

        target : int or None, default=None
            Scalar model-output target.

        Returns
        -------
        self : TreeIG
            The fitted TreeIG attribution object.
        """
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
    """
    Convenience function for computing TreeIG attributions with diagnostics.

    This is a functional wrapper around ``TreeIG(...).explain(...)``. It
    constructs a temporary ``TreeIG`` object and returns the output of
    ``explain``.

    Parameters
    ----------
    model : object
        Fitted supported tree-based model.

    baseline : array-like of shape (n_features,)
        Baseline input.

    X : array-like of shape (n_samples, n_features)
        Evaluation inputs.

    time_tol : float, default=1e-10
        Tolerance used when ordering path-crossing times.

    tie_policy : {"first"}, default="first"
        Rule for coincident active split crossings. Only ``"first"`` is
        currently implemented.

    target : int or None, default=None
        Scalar model-output target.

    batch_size : int or None, default=None
        Number of rows to process per batch.

    Returns
    -------
    attributions : ndarray of shape (n_samples, n_features)
        Exact integrated-gradient feature attributions.

    infos : list of dict
        Per-observation additivity diagnostics.

    summary : dict
        Aggregate additivity and event-count diagnostics.

    See Also
    --------
    TreeIG : Object-oriented interface.
    TreeIG.attribute : Fast attribution method without diagnostics.
    TreeIG.explain : Attribution method with diagnostics.
    """
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
