# Reading and plotting results

## Explanation objects and SHAP plots

Like UnifiedIG, direct TreeIG returns a plotting-library-independent
`Explanation` containing parallel attribution arrays and completeness
diagnostics:

```python
ig = tig.TreeIG(model, baseline=x0)
result = ig.explain(X_eval)

result.values
result.base_values
result.data
result.feature_names
result.output_names
result.max_abs_completeness_error
```

Convert it when you want to use SHAP's plotting ecosystem:

```python
import shap

shap_values = result.to_shap()
shap.plots.beeswarm(shap_values)
shap.plots.waterfall(shap_values[0])
shap.plots.bar(shap_values)
```

SHAP is an optional plotting dependency; install it with
`pip install treeig[shap]`. The conversion changes only the container, not the
TreeIG attribution semantics.

For detailed split-crossing statistics, call `diagnostics`:

```python
infos, summary = ig.diagnostics(X_eval)
```

Each entry in `infos` describes one observation:

```python
{
    "n_events":        ...,   # number of split-crossing events
    "endpoint_delta":  ...,   # F(x) - F(x0)
    "attribution_sum": ...,   # sum_j phi_j
    "residual":        ...,   # attribution_sum - endpoint_delta
    "abs_residual":    ...,
}
```

TreeIGNumeric returns the same fields plus `n_coincident_events`, the number of
events that remained unresolved and were allocated by the fallback rule. It
also reports `n_refined_intervals`, `max_refinement_depth`, and
`n_unresolved_intervals`. The `summary` dictionary aggregates these refinement,
residual, and event-count diagnostics.

## Classification targets

For binary additive-score classifiers, `target=None` and `target=1` both
attribute the positive-class margin. `target=0` attributes the negative margin,
implemented as the negative of the positive-class margin.

```python
ig = tig.TreeIG(model, baseline=x0, target=1)
phi_pos = ig.attribute(X_eval)

ig = tig.TreeIG(model, baseline=x0, target=0)
phi_neg = ig.attribute(X_eval)
```

For multiclass classifiers, pass the class index explicitly.

```python
ig = tig.TreeIG(model, baseline=x0, target=2)
phi_class_2 = ig.attribute(X_eval)
```

Exact TreeIG attributes raw class margins. TreeIGNumeric can use the explicit
probability-derived score convention above when no native margin exists.

## Functional interface

TreeIG also provides a direct functional interface.

```python
result = tig.compute(
    model,
    baseline=x0,
    X=X_eval,
)
```
