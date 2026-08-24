"""A small, plotting-library-independent explanation container."""

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Explanation:
    """Parallel arrays describing feature attributions.

    The contract matches :class:`unifiedig.Explanation` and mirrors the useful
    subset of :class:`shap.Explanation`. Arrays use a leading sample dimension.
    """

    values: NDArray[np.floating]
    base_values: NDArray[np.floating]
    data: NDArray[Any]
    feature_names: Optional[Sequence[str]] = None
    output_names: Optional[Sequence[str]] = None
    completeness_error: Optional[NDArray[np.floating]] = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        data = np.asarray(self.data)
        base_values = np.asarray(self.base_values)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "base_values", base_values)
        if data.ndim < 2 or data.shape[0] == 0:
            raise ValueError("data must have a non-empty leading sample dimension")
        scalar_output = values.shape == data.shape
        multi_output = values.ndim == data.ndim + 1 and values.shape[:-1] == data.shape
        if not scalar_output and not multi_output:
            raise ValueError(
                "values must match data or add one trailing output dimension"
            )
        expected_base_shape = (
            (data.shape[0],)
            if scalar_output
            else (data.shape[0], values.shape[-1])
        )
        if base_values.shape != expected_base_shape:
            raise ValueError(f"base_values must have shape {expected_base_shape}")
        if self.output_names is not None:
            output_names = [str(name) for name in self.output_names]
            object.__setattr__(self, "output_names", output_names)
            expected_outputs = 1 if scalar_output else values.shape[-1]
            if len(output_names) != expected_outputs:
                raise ValueError(
                    f"output_names must contain {expected_outputs} names"
                )
        if self.feature_names is not None:
            feature_names = [str(name) for name in self.feature_names]
            object.__setattr__(self, "feature_names", feature_names)
            if data.ndim == 2 and len(feature_names) != data.shape[1]:
                raise ValueError(
                    f"feature_names must contain {data.shape[1]} names"
                )
        if self.completeness_error is not None:
            completeness_error = np.asarray(self.completeness_error)
            object.__setattr__(self, "completeness_error", completeness_error)
            if completeness_error.shape != base_values.shape:
                raise ValueError(
                    "completeness_error must have the same shape as base_values"
                )

    def __len__(self) -> int:
        return len(self.data)

    @property
    def max_abs_completeness_error(self) -> Optional[float]:
        """Largest absolute completeness residual, or ``None`` if unavailable."""

        if self.completeness_error is None:
            return None
        return float(np.max(np.abs(self.completeness_error)))

    def contrast(self, first: Any, second: Any) -> "Explanation":
        """Return the scalar output contrast ``first - second``."""

        if self.values.ndim != self.data.ndim + 1:
            raise ValueError("contrast requires a multi-output explanation")
        first_index = self._output_index(first)
        second_index = self._output_index(second)
        if first_index == second_index:
            raise ValueError("contrast outputs must be different")
        error = None
        if self.completeness_error is not None:
            error = (
                self.completeness_error[..., first_index]
                - self.completeness_error[..., second_index]
            )
        first_name = self._output_name(first_index)
        second_name = self._output_name(second_index)
        return Explanation(
            values=self.values[..., first_index] - self.values[..., second_index],
            base_values=(
                self.base_values[..., first_index]
                - self.base_values[..., second_index]
            ),
            data=self.data,
            feature_names=self.feature_names,
            output_names=[f"{first_name} - {second_name}"],
            completeness_error=error,
        )

    def _output_index(self, output: Any) -> int:
        n_outputs = self.values.shape[-1]
        if isinstance(output, Integral) and not isinstance(output, bool):
            index = int(output)
            if index < 0 or index >= n_outputs:
                raise IndexError(f"output index must be in [0, {n_outputs - 1}]")
            return index
        if isinstance(output, str):
            if self.output_names is None:
                raise ValueError("named contrasts require output_names")
            matches = [
                index for index, name in enumerate(self.output_names)
                if name == output
            ]
            if len(matches) != 1:
                raise ValueError(f"unknown or ambiguous output name: {output!r}")
            return matches[0]
        raise TypeError("contrast outputs must be integer indices or names")

    def _output_name(self, index: int) -> str:
        if self.output_names is None:
            return str(index)
        return self.output_names[index]

    def to_shap(self) -> Any:
        """Return an equivalent ``shap.Explanation`` when SHAP is installed."""

        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "SHAP is optional. Install it with `pip install treeig[shap]`."
            ) from exc
        output_names: Any = self.output_names
        if self.values.shape == self.data.shape and output_names is not None:
            output_names = output_names[0]
        return shap.Explanation(
            values=self.values,
            base_values=self.base_values,
            data=self.data,
            feature_names=self.feature_names,
            output_names=output_names,
        )
