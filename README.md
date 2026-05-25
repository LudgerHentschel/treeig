# TreeIG

TreeIG computes exact Integrated Gradients for tree ensembles. It decomposes the change in a fitted tree model's scalar output between a baseline input $x_0$ and an observation $x$ into additive feature contributions.

For each observation, TreeIG returns feature attributions $\phi_j$ satisfying

$$\sum_j \phi_j = F(x) - F(x_0),$$

where $F$ is the scalar model output being explained. For regression models, $F$ is the prediction. For supported classifiers, $F$ is the raw margin/logit, not the predicted probability.

Integrated Gradients (Sundararajan, Taly, and Yan, 2017) defines feature attributions by integrating model gradients along a straight-line path from a baseline $x_0$ to the observation $x$.

At first glance, Integrated Gradients appears mismatched with piecewise-constant tree models: gradients vanish almost everywhere and are undefined at split boundaries. The path-integral formulation resolves this. Rather than introducing numerical approximation error through quadrature, the tree structure permits an exact finite decomposition in which the attribution reduces to the sum of prediction jumps at split boundaries crossed along the integration path. The result is exact — no Monte Carlo sampling, no numerical quadrature, no approximation parameters.

Because TreeIG replaces numerical quadrature and sampling with a finite sum over split crossings, it is fast in practice. For many real-world models — hundreds of trees, hundreds of features, thousands of observations — attribution completes in under a millisecond on a modern laptop. (See the example notebook for timings.) For many typical use cases TreeIG is faster than TreeSHAP, which is itself considered fast.

## Installation

Requires Python ≥ 3.9, NumPy, and Numba.

```bash
pip install treeig
```

## Using TreeIG

TreeIG follows a familiar explainer pattern:

```python
ig = treeig.TreeIG(model, baseline=x0)
phi = ig.attribute(X)
```

## Why TreeIG?

Standard Integrated Gradients defines feature contributions by integrating
model gradients along a straight-line path from a baseline input to the
observation. Tree models are piecewise constant, so ordinary gradients are
zero almost everywhere and undefined at split boundaries.

TreeIG uses the tree structure directly. Along the interpolation path

$$ x(t) = x0 + t \cdot (x - x0),\qquad    0 \le t \le 1, $$

a tree prediction changes only when the path crosses a split threshold.
TreeIG finds those crossings exactly and assigns each prediction jump to the
feature responsible for the crossing. For ensembles, contributions are summed
across trees. The result is an exact additive decomposition without numerical
quadrature.

The distributional-derivative perspective makes this precise. Along the
interpolation path the prediction is piecewise constant, and its generalized
derivative is a sum of localized impulses at split crossings. The path integral
of each impulse is exactly the prediction jump at that crossing.

<p align="center">
  <img src="docs/Figure_TreeGradient.svg" width="700">
</p>

The top panel shows a step in the tree prediction along the interpolation path. The middle panel shows the corresponding distributional derivative: zero everywhere except at the split crossing. (Here, $\delta(t - t^\ast)$ is the Dirac delta distribution centered at $t^\ast$.) The bottom panel shows that the path integral localizes exactly at the crossing and recovers the prediction jump.

Standard numerical Integrated Gradients methods try to approximate these impulses using dense interpolation grids. TreeIG instead computes the split-crossing contributions analytically from the fitted tree structure.

## Relation to SHAP and TreeSHAP

TreeIG and TreeSHAP answer different attribution questions and generally produce different decompositions. Neither dominates the other.

**TreeIG** answers: "How much does feature $j$ contribute to the change in prediction as we move continuously from baseline $x_0$ to observation $x$?"

- Attribution is the integral of partial derivatives along the path from $x_0$ to $x$. (For piecewise-constant trees this integral reduces exactly to a sum of prediction jumps at split boundaries crossed along the path.)

**TreeSHAP** answers: "How much does feature $j$ shift the expected prediction, averaged over all possible subsets of the other features?"

- Attribution is an average of discrete inclusion effects, where absent features are marginalized out over a background dataset. There is no path; the reference point is the expected prediction over the background distribution.

The methods differ in two fundamental ways.

First, TreeIG takes a specific baseline input $x_0$ as its reference, while TreeSHAP uses a background distribution.

Second, TreeIG measures contributions through calculus -- integrating how the prediction changes as features move continuously from their baseline values -- while TreeSHAP measures contributions through discrete feature inclusion, asking how much each feature changes the expected prediction when it enters a coalition.

SHAP's coalition construction is deliberately indifferent to the prediction surface between the background and the observation. A feature is either in the coalition or out — there is no interpolation, no path, no attention to what happens as the feature value moves from its background value to its observed value. The attribution is built entirely from discrete switches. This means SHAP explores a wide neighborhood of hybrid inputs, many of which may be far from any natural path between real observations, and measures how the model responds to that exploration.

IG by contrast follows a single specific path and pays close attention to everything that happens along it. The attribution accumulates exactly the prediction changes that occur as all features move continuously from their baseline values to their observed values, holding the model fixed throughout. Nothing synthetic is introduced — the model is only ever evaluated at convex combinations of two real inputs.

The practical implication: SHAP's breadth gives it sensitivity to how the model behaves across a wide range of feature combinations, including combinations that sit away from the natural data distribution. IG's specificity gives it a precise account of what the model does on a particular trajectory through input space. SHAP explores a neighborhood; IG traces a path.

For a linear model with $x_0$ equal to the background mean, TreeIG and TreeSHAP produce identical attributions. As the model becomes more nonlinear or the baseline $x_0$ diverges from the background distribution, the two methods increasingly disagree — reflecting genuine differences in the questions they answer rather than errors in either method.
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

For classification models, TreeIG attributes raw margins or logits. It does not
attribute predicted probabilities because these are not additive across trees.

TreeIG computes exact path decompositions directly from the fitted tree
structure. Since tree representations differ substantially across
implementations, each model family requires customized parsing and routing
logic.

## Not currently supported

TreeIG deliberately does not yet support:

- CatBoost;
- categorical splits;
- missing-value routing (use feature augmentation for missingness);
- probability-output attribution (because probability attribution is not additive);
- probability-averaging or vote-share classifiers such as
  `DecisionTreeClassifier`, `RandomForestClassifier`, and
  `ExtraTreesClassifier` (because the produce probabilities, not scores).

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

`phi` has the same shape as `X_eval`. Row `i`, column `j` is the contribution
of feature `j` to the model-output change from `x0` to `X_eval[i]`.

For regression models, the completeness property holds exactly:

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
    "n_events":        ...,   # number of split-crossing events
    "endpoint_delta":  ...,   # F(x) - F(x0)
    "attribution_sum": ...,   # sum_j phi_j
    "residual":        ...,   # attribution_sum - endpoint_delta
    "abs_residual":    ...,
}
```

The `summary` dictionary reports aggregate residual and event-count statistics.

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

TreeIG attributes raw class margins. If probability-space explanations are
needed, users should transform or interpret the margin-level contributions
separately.

## Functional interface

TreeIG also provides a direct functional interface.

```python
phi, infos, summary = tig.compute(
    model,
    baseline=x0,
    X=X_eval,
)
```

## Warmup

TreeIG uses Numba for fast parallel attribution kernels. The first call
includes JIT compilation. You can compile in advance with `warmup`:

```python
ig = tig.TreeIG(model, baseline=x0).warmup(X_eval[:3])
phi = ig.attribute(X_eval)
```

Subsequent calls on the same model are fast. Attribution for thousands of
observations on a typical ensemble completes in well under a second after
warmup.

## Numerical conventions

TreeIG follows each backend's split-routing convention as closely as possible.

- scikit-learn trees route left when `x[j] <= threshold`;
- LightGBM numeric splits route left when `x[j] <= threshold`;
- XGBoost numeric splits route left when `x[j] < threshold`
  using float32-style comparisons.

Inputs must be finite numeric arrays. Missing-value routing is not currently
implemented, so `NaN` and `Inf` values raise errors.

## Baselines

The baseline $x_0$ defines the reference point for the decomposition. Common
choices include the training-sample mean, a median or representative
observation, a domain-specific neutral input, or a fixed benchmark case.

The attribution always explains the difference between the model output at the
observation and the model output at the chosen baseline. Different baselines
answer different questions.

## Interpretation

For an observation $x$, TreeIG reports how much each feature contributes to
moving the model output from $F(x_0)$ to $F(x)$ along the straight-line path
from $x_0$ to $x$. Positive contributions increase the scalar output relative
to the baseline; negative contributions decrease it. The contributions are
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

TreeIG is production-ready for exact attribution of fitted tree models in
raw-output space. The current release covers the dominant tree ensemble
backends in the Python ecosystem.

Future extensions may include:

- CatBoost support, which requires customized analysis of oblivious trees
  and categorical split structure;
- alternative allocation rules for simultaneous multi-feature effects at
  coincident split crossings.

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

Popular implementations of Integrated Gradients for smooth models:

- Captum for PyTorch: https://captum.ai/
- TensorFlow Integrated Gradients: https://www.tensorflow.org/tutorials/interpretability/integrated_gradients