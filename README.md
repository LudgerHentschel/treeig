# TreeIG

TreeIG computes exact Integrated Gradients for tree ensembles. It decomposes the change in a fitted tree model's scalar output between a baseline input $x_0$ and an observation $x$ into additive feature contributions.

For each observation, TreeIG returns feature attributions $\phi_j$ satisfying

```text
sum_j phi_j = F(x) - F(x0)
```

where $F$ is the scalar model output being explained. For regression models, $F$ is the prediction. For supported classifiers, $F$ is the raw margin/logit, not the predicted probability.

TreeIG extends the Integrated Gradients framework of Sundararajan, Taly, and Yan (2017) to tree ensembles by exploiting the piecewise-constant structure of tree models.

TreeIG uses distributional gradients to extend Integrated Gradients to tree-based models. The integrals of the distributional gradients are exactly equal to the sum of the prediction steps along the input path. TreeIG uses this equivalence to efficiently compute Integrated Gradients for tree models. 

## Using TreeIG

TreeIG follows a familiar explainer pattern:

```python
ig = treeig.TreeIG(model, baseline=x0)
phi = ig.attribute(X)
```

TreeIG computes exact Integrated Gradients for supported tree models. It does not rely on Monte Carlo sampling, numerical integration, or approximation parameters controlling attribution accuracy.

## References

TreeIG:

- Hentschel, Ludger. 2026. 
  "TreeIG: Exact Integrated Gradients for Tree-Based Models."
  *https://www.ludgerhentschel.com/Research.html*

Integrated Gradients:

- Sundararajan, Mukund, Ankur Taly, and Qiqi Yan. 2017.
  "Axiomatic Attribution for Deep Networks."
  *International Conference on Machine Learning (ICML)*.

SHAP and TreeSHAP:

- Lundberg, Scott M., and Su-In Lee. 2017.
  "A Unified Approach to Interpreting Model Predictions."
  *Advances in Neural Information Processing Systems (NeurIPS)*.

- Lundberg, Scott M., Gabriel Erion, and Su-In Lee. 2020.
  "From Local Explanations to Global Understanding with Explainable AI for Trees."
  *Nature Machine Intelligence*.

Popular implementations of Integrated Gradients for smooth models include:

- Captum for PyTorch:
  https://captum.ai/

- TensorFlow Integrated Gradients tutorials:
  https://www.tensorflow.org/tutorials/interpretability/integrated_gradients

## Why TreeIG?

Standard Integrated Gradients defines feature contributions by integrating
model gradients along a path from a baseline input to the observation.
Tree models are piecewise constant, so ordinary gradients are zero almost
everywhere and undefined at split boundaries.

TreeIG uses the tree structure directly. Along the straight-line path

```text
x(t) = x0 + t * (x - x0),    0 <= t <= 1,
```

a tree prediction changes only when the path crosses a split threshold.
TreeIG finds those crossings exactly and assigns each jump in prediction
to the feature responsible for the crossing. For ensembles, contributions
are summed across trees.

This gives an exact additive decomposition for tree models without
numerical quadrature.

For smooth models, TreeIG reduces to ordinary Integrated Gradients. For tree models, TreeIG computes the exact path decomposition implied by split crossings.

### Distributional intuition

For tree models, the prediction along the interpolation path is piecewise constant. The prediction changes only when the path crosses a split threshold.

TreeIG interprets these jumps using generalized (distributional) derivatives. A split crossing produces a localized impulse whose integral is exactly equal to the prediction jump.

<p align="center">
  <img src="docs/Figure_TreeGradient.svg" width="700">
</p>

The top panel shows a step in the tree prediction along the interpolation path. The middle panel shows the corresponding distributional derivative: zero everywhere except at the split crossing. The bottom panel shows that the path integral localizes exactly at the crossing and recovers the prediction jump.

Standard numerical Integrated Gradients methods approximate these localized impulses using dense interpolation grids and numerical gradient approximations. TreeIG instead computes the split-crossing contributions analytically from the fitted tree structure, yielding exact additive attributions for tree models.

## Relation to SHAP and TreeSHAP

TreeIG and TreeSHAP answer different attribution questions and generally
produce different decompositions. Neither dominates the other.

**TreeIG** answers: *"How much does feature j contribute to the change in
prediction as we move continuously from baseline x₀ to observation x?"*
Attribution is the integral of partial derivatives along the path from `x0`
to `x`. For piecewise-constant trees this integral reduces exactly to a sum of
prediction jumps at split boundaries crossed along the path.

**TreeSHAP** answers: *"How much does feature j individually shift the
expected prediction, averaged over all possible subsets of the other
features?"* Attribution is an average of discrete inclusion effects, where
absent features are marginalized out over a background dataset. There is no
path and no baseline input; the reference point is the average prediction
over the background distribution.

The methods differ in two fundamental ways. First, TreeIG takes a specific
baseline input `x0` as its reference; TreeSHAP takes a background
distribution. Second, TreeIG measures contributions through calculus —
integrating how the prediction changes as each feature moves continuously
from its baseline value — while TreeSHAP measures contributions through
discrete feature inclusion, asking how much each feature shifts the
expected prediction when it enters a coalition.

## Supported models

TreeIG currently supports tree models with finite numeric feature inputs.

### Regression

- `sklearn.tree.DecisionTreeRegressor`
- `sklearn.ensemble.RandomForestRegressor`
- `sklearn.ensemble.ExtraTreesRegressor`
- `sklearn.ensemble.GradientBoostingRegressor`
- `xgboost.XGBRegressor`
- `xgboost.Booster`
- `lightgbm.LGBMRegressor`
- `lightgbm.Booster`

### Classification (raw margins/logits only)

- `sklearn.ensemble.GradientBoostingClassifier`
- `xgboost.XGBClassifier`
- `lightgbm.LGBMClassifier`

For classification models, TreeIG attributes raw margins or logits. It does not attribute predicted probabilities because these are not additive.

TreeIG computes exact path decompositions directly from the fitted tree structure. Since tree representations differ substantially across implementations, each model family requires customized parsing and routing logic.

## Not currently supported

TreeIG deliberately does not yet support:

- probability-output attribution;
- missing-value routing;
- categorical splits;
- CatBoost;
- probability-averaging or vote-share classifiers such as
  `DecisionTreeClassifier`, `RandomForestClassifier`, and
  `ExtraTreesClassifier`.

## Installation

```bash
pip install treeig
```

Or locally:

```bash
pip install -e .
```

## Basic usage

```python
import numpy as np
import treeig as tig

# model is a fitted supported tree model
x0 = X_train.mean(axis=0)
X_eval = X_test[:100]

ig = tig.TreeIG(model, baseline=x0)
phi = ig.attribute(X_eval)
```

`phi` has the same shape as `X_eval`. Row `i`, column `j`
is the contribution of feature `j` to the model-output change from
`x0` to `X_eval[i]`.

For regression models:

```python
np.testing.assert_allclose(
    phi.sum(axis=1),
    model.predict(X_eval) - model.predict(x0.reshape(1, -1))[0],
)
```

## Diagnostics

Use `explain` when you want attributions together with completeness
diagnostics.

```python
ig = tig.TreeIG(model, baseline=x0)
phi, infos, summary = ig.explain(X_eval)

print(summary)
```

Each entry in `infos` contains diagnostics for one observation:

```python
{
    "n_events": ...,          # number of split-crossing events
    "endpoint_delta": ...,    # F(x) - F(x0)
    "attribution_sum": ...,   # sum_j phi_j
    "residual": ...,          # attribution_sum - endpoint_delta
    "abs_residual": ...,
}
```

The `summary` dictionary reports aggregate residual and event-count
statistics.

## Classification targets

For binary additive-score classifiers, `target=None` and `target=1`
both attribute the positive-class margin. `target=0` attributes the
negative margin, implemented as the negative of the positive-class
margin.

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

TreeIG attributes raw class margins. If probability-space explanations
are needed, users should transform or interpret the margin-level
contributions separately.

## Warmup

TreeIG uses Numba for fast attribution kernels. The first call may
include compilation time. You can compile the kernels in advance with
`warmup`.

```python
ig = tig.TreeIG(model, baseline=x0).warmup(X_eval[:3])
phi = ig.attribute(X_eval)
```

## Functional interface

TreeIG also provides a direct functional interface.

```python
phi, infos, summary = tig.compute(
    model,
    baseline=x0,
    X=X_eval,
)
```

## Numerical conventions

TreeIG follows each backend's split-routing convention as closely as
possible.

- scikit-learn trees route left when `x[j] <= threshold`;
- LightGBM numeric splits route left when `x[j] <= threshold`;
- XGBoost numeric splits route left when `x[j] < threshold`
  using float32-style comparisons.

Inputs must be finite numeric arrays. Missing-value routing is not
currently implemented, so `NaN` and `Inf` values raise errors.

## Baselines

The baseline `x0` defines the reference point for the decomposition.
Common choices include:

- the training-sample mean;
- a median or representative observation;
- a domain-specific neutral input;
- a fixed benchmark case.

The attribution always explains the difference between the model output
at the observation and the model output at the chosen baseline.
Different baselines answer different questions.

## Interpretation

For an observation `x`, TreeIG reports how much each feature contributes
to moving the model output from `F(x0)` to `F(x)` along the straight-line
path from `x0` to `x`.

Positive contributions increase the scalar output relative to the
baseline. Negative contributions decrease it. The contributions are
additive by construction.

## Example: XGBoost regression

```python
import numpy as np
import xgboost as xgb
import treeig as tig

model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    objective="reg:squarederror",
    random_state=0,
)

model.fit(X_train, y_train)

x0 = X_train.mean(axis=0)
X_eval = X_test[:100]

ig = tig.TreeIG(model, baseline=x0).warmup(X_eval[:3])

phi, infos, summary = ig.explain(X_eval)

print(phi.shape)
print(summary["max_abs_residual"])
```

## Example: multiclass classification margins

```python
import lightgbm as lgb
import treeig as tig

model = lgb.LGBMClassifier(...)
model.fit(X_train, y_train)

x0 = X_train.mean(axis=0)
X_eval = X_test[:100]

# Attribute class-2 raw margin
ig = tig.TreeIG(model, baseline=x0, target=2)

phi = ig.attribute(X_eval)
```

## Project status

TreeIG is intended for exact additive attribution of fitted tree models
in raw-output space. The current implementation focuses on correctness,
backend-specific routing consistency, and a compact API.

Future extensions may include:

- CatBoost support, which requires customized analysis of oblivious trees
  and categorical split structure;
- alternative allocation rules for simultaneous multi-feature effects at
  coincident split crossings.