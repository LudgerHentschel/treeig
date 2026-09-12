# Changelog

## 0.2.2

- Change `TreeIGNumeric` and `make_scalar_fn` to derive classification scores
  by default when no native margin exists: binary log odds or centered
  multiclass log probabilities. `compute_numeric` inherits this default.
  Callers requiring the former class-probability output must explicitly set
  `probability_to_score=False`. Zero probabilities require an explicit
  `probability_floor` for finite scores; no floor is chosen silently.

- Add an agent documentation index, sitemap, canonical URLs, page descriptions,
  and build checks for documentation discovery. Clarify exact versus numerical
  model support, baseline guarantees, and related-project links.
- Lead the README and documentation landing page with the piecewise-constant
  gradient argument, and move the derivative-impulse figure above the fold.
- State completeness precision relative to the fitted model's own arithmetic
  rather than unqualified floating-point precision.

## 0.2.1

- Separate numerical jump-detection tolerance from absolute and relative completeness-warning tolerances; retain raw residual diagnostics.
- Clarify jump-based fallback behavior, bundled-event completeness, and XGBoost prediction precision.
- Check numeric regression tests with unexpected runtime warnings treated as errors in CI.

## 0.2.0

- Add optional `treeig.GPUTreeIG` prediction attribution with persistent CUDA model and weighted-baseline state and reusable observation buffers.
- Keep CPU `TreeIG` as the default; load CUDA only when constructing `GPUTreeIG`.
- Add the `cuda` installation extra, simulator equivalence tests, and GPU usage and benchmark documentation.
- Correct GitHub project links and restrict release publishing to version tags.

GPUTreeIG is part of `treeig`, not a separate distribution. GPU performance depends on the workload; existing T4 measurements are examples rather than guarantees.
