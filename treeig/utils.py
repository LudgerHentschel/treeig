from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class _ResolvedTarget:
    target: Optional[int]
    sign: float


def _as_target_key(target: Optional[int]) -> Optional[int]:
    if target is None:
        return None
    return int(target)


def _check_finite_numeric(X: np.ndarray, name: str) -> None:
    if not np.isfinite(X).all():
        raise ValueError(
            f"{name} contains non-finite values (NaN or Inf). "
            "Missing-value routing is not currently supported."
        )


def _as_float32_float64(X: np.ndarray) -> np.ndarray:
    """Round through float32 while returning a float64 array for Numba kernels."""
    return np.asarray(X, dtype=np.float64).astype(np.float32).astype(np.float64)


def _resolve_binary_classifier_target(target: Optional[int]) -> _ResolvedTarget:
    """
    Binary additive-score classifiers expose one positive-class margin.

    target=None or target=1 attributes the positive-class margin.
    target=0 attributes the negative margin as -positive_margin.
    """
    if target is None or int(target) == 1:
        return _ResolvedTarget(target=1, sign=1.0)
    if int(target) == 0:
        return _ResolvedTarget(target=0, sign=-1.0)
    raise ValueError("Binary classification target must be None, 0, or 1.")


def _select_prediction_target(pred: np.ndarray, target: Optional[int]) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)

    if pred.ndim == 1:
        if target is None or int(target) == 1:
            return pred
        if int(target) == 0:
            return -pred
        raise ValueError("Binary classification target must be None, 0, or 1.")

    if pred.ndim == 2:
        if pred.shape[1] == 1:
            out = pred[:, 0]
            if target is None or int(target) == 1:
                return out
            if int(target) == 0:
                return -out
            raise ValueError("Binary classification target must be None, 0, or 1.")

        if target is None:
            raise ValueError("Multiclass prediction requires target=<class index>.")
        k = int(target)
        if k < 0 or k >= pred.shape[1]:
            raise ValueError(f"target must be in [0, {pred.shape[1] - 1}], got {target}.")
        return pred[:, k]

    raise ValueError(f"Unsupported prediction shape {pred.shape}.")
