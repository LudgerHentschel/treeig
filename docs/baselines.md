# Choosing baselines and batching

The baseline determines the question an attribution answers. With one baseline,
TreeIG explains $F(x)-F(x_0)$. With baseline rows $b_k$ and normalized weights
$w_k$, it averages their path attributions:

$$\phi_j(x)=\sum_k w_k\phi_j(x;b_k),\qquad
\sum_j\phi_j(x)=F(x)-\sum_k w_kF(b_k).$$

This is generally different from using the mean baseline row: a nonlinear
model need not satisfy $F(\sum_k w_k b_k)=\sum_k w_k F(b_k)$.

## Choosing the reference

For Integrated Gradients, the baseline determines the prediction contrast
being explained. **[CBaseline](https://github.com/LudgerHentschel/cbaseline) is the
preferred way to construct TreeIG baselines.** CBaseline produces empirical,
prediction-neutral baseline *distributions* whose weighted mean model output is
the chosen reference prediction. TreeIG then explains the model prediction
relative to that reference level rather than relative to an arbitrary feature
vector such as the feature-wise mean.

TreeIG accepts a CBaseline `Background` directly and evaluates its weighted
baseline paths efficiently. See CBaseline for construction choices and the
interpretation of the reference prediction `f0`.

A single representative observation, domain-specific neutral input, or fixed
benchmark case is also supported. A sample mean is convenient for a first
example, but it need not describe a plausible observation or a neutral model
prediction. State the baseline choice when reporting attributions.

## Weighted distributions

Assume `model` is fitted and `X_train` and `X_eval` have matching feature columns:

```python
import numpy as np
from treeig import TreeIG

baselines = X_train[:20]
weights = np.ones(len(baselines))
ig = TreeIG(model, baseline=baselines, baseline_weights=weights)
result = ig.explain(X_eval)

expected = ig.model_output(X_eval) - np.mean(ig.model_output(baselines))
np.testing.assert_allclose(result.values.sum(axis=1), expected, atol=1e-8)
```

Weights must be finite and nonnegative, have positive total mass, and match the
number of baseline rows. TreeIG normalizes them internally. Omitted weights
mean equal weights. A CBaseline `Background` can be passed directly as
`baseline=background`; TreeIG reads its rows and weights.

## Batching and per-baseline results

```python
phi = ig.attribute(X_eval, batch_size=100, baseline_batch_size=10)
weighted, by_baseline = ig.attribute(X_eval, return_by_baseline=True)
```

`phi` and `weighted` have shape `(n_observations, n_features)`.
`by_baseline` has shape `(n_baselines, n_observations, n_features)` and retains
unweighted attributions for each baseline. Request it only when needed: its
memory use grows with all three dimensions.

Observation batching limits rows processed together. Baseline batching limits
baseline work processed together. They preserve the weighted interpretation,
subject to floating-point summation differences. Pass the whole distribution
through one call rather than maintaining your own Python attribution loop.

A baseline supplied to a CPU attribution call overrides the constructor default
for that call. GPUTreeIG has a separate fixed-state contract described in its
specialized documentation.
