"""Explicit, persistent GPU API for TreeIG prediction attribution."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .api import TreeIG


class GPUTreeIG(TreeIG):
    """Exact TreeIG prediction attribution with fixed state resident on CUDA.

    The model and weighted baseline distribution are prepared and uploaded at
    construction. Repeated :meth:`attribute` calls reuse that fixed state and
    device buffers. GPU selection is explicit; the existing
    :class:`TreeIG` CPU implementation and dispatch are unchanged.
    """

    def __init__(
        self,
        model: Any,
        baseline: np.ndarray,
        baseline_weights: Optional[np.ndarray] = None,
        time_tol: float = 1e-10,
        tie_policy: str = "first",
        target: Optional[int] = None,
        *,
        threads_per_block: int = 128,
    ):
        from .cuda_backend import CUDAAttributor

        if baseline is None:
            raise ValueError(
                "GPUTreeIG requires baseline= at construction so the fixed "
                "baseline distribution can remain resident on the GPU."
            )
        super().__init__(
            model,
            baseline=baseline,
            baseline_weights=baseline_weights,
            time_tol=time_tol,
            tie_policy=tie_policy,
            target=target,
        )
        baselines, weights = self._resolve_baselines(None, None, self._arrays)
        self._gpu_baselines = baselines
        self._gpu_weights = weights
        y0 = np.stack([
            self._get_y0_per_tree(self._arrays, row) for row in baselines
        ])
        self._gpu = CUDAAttributor(
            self._arrays,
            baselines,
            weights,
            self.time_tol,
            y0,
            threads_per_block=threads_per_block,
        )

    @property
    def device_capacity(self) -> int:
        """Rows accommodated by the currently allocated reusable GPU buffers."""
        return self._gpu.capacity

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
        """Compute exact prediction attributions on the selected CUDA device."""
        arrays = self._resolve_arrays_for_target(target)
        if arrays is not self._arrays:
            raise ValueError(
                "GPUTreeIG target is fixed at construction because its model "
                "state is resident on the GPU."
            )
        if baseline is not None:
            requested_baselines, requested_weights = self._prepare_baselines(
                baseline, baseline_weights, arrays
            )
            if not (
                np.array_equal(requested_baselines, self._gpu_baselines)
                and np.array_equal(requested_weights, self._gpu_weights)
            ):
                raise ValueError(
                    "GPUTreeIG baseline state is fixed at construction."
                )
        elif baseline_weights is not None:
            raise ValueError(
                "GPUTreeIG baseline weights are fixed at construction."
            )
        if baseline_batch_size is not None:
            raise ValueError(
                "GPUTreeIG keeps the complete baseline distribution resident; "
                "baseline_batch_size is not supported."
            )
        if return_by_baseline:
            raise NotImplementedError(
                "GPUTreeIG does not currently return per-baseline attributions."
            )
        X_prep = self._prepare_X(X, self._arrays)
        batch_size = self._validate_batch_size(batch_size) or X_prep.shape[0]
        if X_prep.shape[0] == 0:
            return np.empty((0, self.n_features_in_), dtype=np.float64)
        result = np.empty_like(X_prep)
        for start in range(0, X_prep.shape[0], batch_size):
            stop = min(start + batch_size, X_prep.shape[0])
            result[start:stop] = self._gpu.attribute(X_prep[start:stop])[0]
        return result

    @staticmethod
    def _unsupported(method: str):
        raise NotImplementedError(
            f"GPUTreeIG.{method} is not implemented. GPUTreeIG currently "
            "supports prediction attribute() and explain() only."
        )

    def diagnostics(self, *args, **kwargs):
        self._unsupported("diagnostics")

    def trace(self, *args, **kwargs):
        self._unsupported("trace")

    def loss_attribution(self, *args, **kwargs):
        self._unsupported("loss_attribution")

    def multiclass_loss_attribution(self, *args, **kwargs):
        self._unsupported("multiclass_loss_attribution")

    def warmup(self, *args, **kwargs):
        self._unsupported("warmup")
