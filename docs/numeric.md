---
myst:
  html_meta:
    description: "Use TreeIGNumeric for explicit numerical jump detection and understand probability outputs, score conversion, and resolution limits."
---

# TreeIGNumeric

TreeIGNumeric is a model-agnostic fallback that recovers the crossing-sum
attribution by numerically detecting prediction discontinuities along the
integration path. It requires no access to model internals — only repeated
evaluations of the prediction function — so it applies to many
piecewise-constant models the exact parser does not support. Whenever a
supported backend is available, exact TreeIG should be preferred. Select
`TreeIGNumeric` explicitly for an unsupported model; `TreeIG` does not silently
switch engines. This fallback sums prediction jumps, rather than estimating
gradients and applying numerical quadrature to a piecewise-constant function.

TreeIGNumeric scans a numerical grid along the integration path to locate
changes in the prediction. It then bisects only the changed intervals, four
adaptive levels by default, before using local axis-aligned probes to attribute
each step to a feature. It preserves completeness for the detected changes and
typically produces attributions very similar to exact TreeIG. Because it
locates crossings numerically, multiple nearby crossings may occasionally be
merged into a single event; exact TreeIG avoids this by enumerating crossings
directly from the tree structure.

The defaults `grid_size=1024` and `max_refine=4` are a practical balance, not
an accuracy guarantee. Refinement concentrates additional evaluations around
detected changes without increasing the global grid. Completeness can remain
exact when offsetting events are hidden inside one coarse interval because the
detected jumps still telescope. For allocation-sensitive work, rerun a
representative subset at a larger grid and compare the feature attributions
themselves. The reproducible
[probability-forest stress benchmark](https://github.com/LudgerHentschel/treeig/blob/main/benchmarks/README.md) performs this
resolution check against an independent structural crossing oracle.

Two caveats on coverage:

- **CatBoost and other encoded models.** TreeIGNumeric removes the *parsing*
  barrier, but not the modeling one: interpolating a *native* categorical
  feature along the straight-line path is not meaningful, which is a property of
  Integrated Gradients itself, not of the implementation. TreeIGNumeric works on
  CatBoost (and similar) models with numeric or one-hot-encoded inputs.
- **Probability-averaging classifiers.** By default, TreeIGNumeric retains its
  original behavior and explains one class probability. With
  `probability_to_score=True`, it instead explains binary log odds or a
  centered multiclass log score. Zero probabilities require an explicit
  `probability_floor`; TreeIGNumeric never clips them silently.

```python
import treeig as tig

ig = tig.TreeIGNumeric(model, baseline=x0)
result = ig.explain(X_eval)
infos, summary = ig.diagnostics(X_eval)
output = ig.model_output(X_eval)

print(result.max_abs_completeness_error)
```

For a probability-only classifier:

```python
ig = tig.TreeIGNumeric(
    model,
    baseline=x0,
    target=2,                    # omit for binary positive-class log odds
    probability_to_score=True,
    probability_floor=1e-6,     # explicit because tree probabilities may be 0
)
phi = ig.attribute(X_eval)
score = ig.model_output(X_eval)
```

For binary probabilities, the explained score is
`log(p1) - log(p0)`. For `K` classes, target `k` selects
`log(p_k) - mean(log(p))`. Pairwise differences are therefore invariant log
odds. The floor is applied to every class probability and the probabilities
are renormalized before taking logarithms.

These are the canonical scores implied by the complete probability vector:
softmax of the centered log scores recovers the original probabilities. They
do not reconstruct an unavailable training-time margin. TreeIGNumeric treats
the derived score itself as the explicitly defined scalar model output and
attributes its jumps after the ensemble probabilities have been aggregated.

## Jump detection and completeness warnings

`tol=0.0` detects every nonzero prediction difference observed by the search.
It also controls local feature-probe comparisons. It is independent of
`residual_atol=1e-12` and `residual_rtol=1e-10`, which control warnings only:

```text
abs(attribution_sum - endpoint_delta) >
    residual_atol + residual_rtol * abs(endpoint_delta)
```

Diagnostics retain the actual residual even when it is below this threshold.
Set both residual tolerances to zero to warn on any nonzero residual, or use
`warn_residual=False` to disable that warning. A positive jump-detection `tol`
can discard small changes, so check it as well as the grid and refinement
settings when investigating a warning.

Isolating every jump is not required for completeness: a bundled interval can
contribute its full net prediction change through the cumulative feature sweep.
Bundling may change the feature allocation, and offsetting jumps inside a cell
can remain invisible even with a zero residual. These are allocation limitations,
not reasons to require exhaustive refinement or replace event detection with
generic numerical Integrated Gradients.
