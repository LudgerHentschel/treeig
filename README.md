# TreeIG

[![PyPI version](https://img.shields.io/pypi/v/treeig.svg)](https://pypi.org/project/treeig/)

TreeIG computes exact Integrated Gradients for tree-based models. It decomposes the change in a fitted tree model's scalar output between a baseline input $x_0$ and an observation $x$ into additive feature contributions.

For each observation, TreeIG returns feature attributions $\phi_j$ satisfying

$$\sum_j \phi_j = F(x) - F(x_0),$$

where $F$ is the scalar model output being explained. For regression models,
$F$ is the prediction. Exact classification backends use native raw margins.
TreeIGNumeric can additionally transform probability-only classifiers to
binary log odds or centered multiclass log scores.

Integrated Gradients (Sundararajan, Taly, and Yan, 2017) defines feature attributions by integrating model gradients along a straight-line path from a baseline $x_0$ to the observation $x$.

At first glance, Integrated Gradients appears mismatched with piecewise-constant tree models: gradients vanish almost everywhere and are undefined at split boundaries. [Hentschel (2026)](https://www.ludgerhentschel.com/PDFs/Hentschel%20'26g.pdf) shows that, for tree-based models, the path-integral of the gradients reduces to the sum of prediction jumps at split boundaries crossed along the integration path. The resulting attribution is exact — no Monte Carlo sampling, no numerical quadrature, no approximation parameters.

Because TreeIG replaces numerical quadrature and sampling with a finite sum over split crossings, it is fast in practice. For many real-world models — hundreds of trees, hundreds of features — attribution over thousands of observations completes in a few milliseconds on a modern laptop. (See the [example notebook](examples/) for timings.) For many typical use cases TreeIG is competitive with, and often faster than, TreeSHAP, which is itself considered fast.

TreeIG also includes [TreeIGNumeric](#treeignumeric), a model-agnostic fallback that recovers the same kind of crossing-sum attribution through numerical event detection when exact structural support is unavailable.

## Recommended baseline distribution

For Integrated Gradients, the baseline determines the prediction contrast
being explained. **[CBaseline](https://github.com/lhentschel/cbaseline) is the
preferred way to construct TreeIG baselines.** CBaseline produces empirical,
prediction-neutral baseline *distributions* whose weighted mean model output is
the chosen reference prediction. TreeIG then explains the model prediction
relative to that reference level rather than relative to an arbitrary feature
vector such as the feature-wise mean.

TreeIG accepts a CBaseline `Background` directly and evaluates its weighted
baseline paths efficiently. See CBaseline for construction choices and the
interpretation of the reference prediction `f0`.

## Installation

```bash
pip install treeig
pip install cbaseline  # recommended baseline construction
pip install "treeig[catboost]"  # optional numerical CatBoost support
```

Requires Python ≥ 3.9, NumPy, and Numba. Model backends (scikit-learn,
XGBoost, LightGBM, and CatBoost) are not installed automatically; install
whichever you use.

## Quickstart

```python
import numpy as np
import treeig as tig

# model is a fitted supported tree model
x0 = X_train.mean(axis=0)
X_eval = X_test[:100]

ig = tig.TreeIG(model, baseline=x0)
result = ig.explain(X_eval)
phi = result.values
```

For libraries integrating TreeIG, the public adapter surface also includes:

```python
tig.supports(model)                       # exact backend availability
ig.model_output(X_eval)                   # scalar output being attributed
result.to_shap()                          # optional SHAP plotting adapter

numeric = tig.TreeIGNumeric(model, baseline=x0)
numeric.model_output(X_eval)              # numeric backend output scale
```

The single-vector example above is the minimal API. For substantive
attribution, prefer a prediction-neutral distribution constructed with
[CBaseline](https://github.com/lhentschel/cbaseline).

Weighted baseline distributions are first-class baselines. Pass either a
matrix and aligned weights or a CBaseline `Background` directly:

```python
ig = tig.TreeIG(model, baseline=background)  # uses .rows and .weights
phi = ig.attribute(X_eval)

# Equivalent explicit form; weights are normalized internally.
phi = ig.attribute(
    X_eval,
    baseline=background.rows,
    baseline_weights=background.weights,
    baseline_batch_size=25,
)
```

TreeIG preserves each baseline-specific path and performs the weighted
aggregation inside a compiled loop. With `return_by_baseline=True`,
`attribute` returns `(weighted, by_baseline)` for diagnostics.

Compiled baseline traversal is also used by `loss_attribution` and
`multiclass_loss_attribution`. For multiclass log
loss, TreeIG merges class-score events in chronological path order before
applying each softmax-loss change. Pass the complete baseline distribution in
one call instead of invoking the explainer once per baseline: this amortizes
tree parsing and model dispatch and keeps baseline aggregation inside compiled
code.

`phi` has the same shape as `X_eval`. Row `i`, column `j` is the contribution
of feature `j` to the model-output change from `x_0` to `X_eval[i]`.

For regression models, the completeness property holds exactly:

```python
np.testing.assert_allclose(
    phi.sum(axis=1),
    model.predict(X_eval) - model.predict(x0.reshape(1, -1))[0],
)
```

## Why TreeIG?

Standard Integrated Gradients defines feature contributions by integrating
model gradients along a straight-line path from a baseline input to the
observation. Tree models are piecewise constant, so ordinary gradients are
zero almost everywhere and undefined at split boundaries.

TreeIG uses the tree structure directly. Along the interpolation path

$$ x(t) = x_0 + t\,(x - x_0),\qquad 0 \le t \le 1, $$

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

The top panel shows a step in the tree prediction along the interpolation path. The middle panel shows the corresponding distributional derivative: zero everywhere except at the split crossing. (Here, $\delta(t - t^\ast)$ is the Dirac delta distribution centered at $t^\ast$.) The bottom panel shows that the path integral localizes exactly at the crossing and recovers the prediction jump. TreeIG exploits the fact that integrated gradients applied to trees requires neither numerical differentiation nor numerical integration; it reduces to a simple sum of prediction steps along the integration path $x(t)$.

Standard numerical Integrated Gradients methods try to approximate these impulses using dense interpolation grids. TreeIG instead computes the split-crossing contributions analytically from the fitted tree structure. In this sense, TreeIG plays a role analogous to automatic differentiation for smooth models: rather than numerically searching for discontinuities, it uses the model's computational structure to evaluate the attribution integral exactly and efficiently. (The analogy understates the gain. Automatic differentiation removes derivative approximation but not the numerical quadrature used by Integrated Gradients. TreeIG exploits tree structure to evaluate the attribution integral itself exactly.)

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

For classification models, exact structural TreeIG attributes raw margins or
logits. Its current exact engine does not transform ensemble probabilities.

TreeIG computes exact path decompositions directly from the fitted tree
structure. Since tree representations differ substantially across
implementations, each model family requires customized parsing and routing
logic.

### Exact support not currently available

The exact TreeIG parser does not currently support:

- CatBoost;
- categorical splits;
- missing-value routing (use feature augmentation for missingness);
- structurally exact transformed-probability attribution;
- probability-averaging or vote-share classifiers such as
  `DecisionTreeClassifier`, `RandomForestClassifier`, and
  `ExtraTreesClassifier` (because they produce probabilities, not scores).

Many of these can still be attributed with the model-agnostic
[TreeIGNumeric](#treeignumeric), described below.

## TreeIGNumeric

TreeIGNumeric is a model-agnostic fallback that recovers the crossing-sum
attribution by numerically detecting prediction discontinuities along the
integration path. It requires no access to model internals — only repeated
evaluations of the prediction function — so it applies to many
piecewise-constant models the exact parser does not support. Whenever a
supported backend is available, exact TreeIG should be preferred.

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
[probability-forest stress benchmark](benchmarks/README.md) performs this
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

## Relation to SHAP and TreeSHAP

TreeIG and TreeSHAP answer different attribution questions and generally produce
different decompositions. Neither dominates the other.

**TreeIG** answers: "How much does feature $j$ contribute to the change in
prediction as we move continuously from baseline $x_0$ to observation $x$?" The
attribution is the integral of partial derivatives along the path from $x_0$ to
$x$, which for piecewise-constant trees reduces exactly to a sum of prediction
jumps at the split boundaries crossed along the path.

**TreeSHAP** answers: "How much does feature $j$ shift the expected prediction,
averaged over all possible subsets of the other features?" The attribution is an
average of discrete inclusion effects, where absent features are marginalized
out over a background dataset. There is no path; the reference is the expected
prediction over the background distribution.

The two differ in two ways. First, TreeIG takes a specific baseline input $x_0$
as its reference, while TreeSHAP uses a background distribution. Second, TreeIG
measures contributions through calculus — integrating how the prediction changes
as features move continuously from their baseline values — while TreeSHAP
measures them through discrete feature inclusion, asking how much each feature
changes the expected prediction when it enters a coalition.

The practical consequence is one of scope. SHAP's coalition construction is
indifferent to the prediction surface between the background and the
observation: a feature is either in the coalition or out, so the attribution is
built from discrete switches and explores a wide neighborhood of hybrid inputs,
many far from any natural path between real observations. IG instead follows a
single path and accumulates exactly the prediction changes along it, evaluating
the model only at convex combinations of two real inputs. SHAP explores a
neighborhood; IG traces a path. SHAP's breadth gives sensitivity to model
behavior across many feature combinations; IG's specificity gives a precise
account of one trajectory through input space.

For a linear model with independent features and $x_0$ equal to the background
mean, TreeIG and interventional SHAP coincide. (A linear model is not a tree, so
the comparison is to SHAP generally rather than to TreeSHAP.) As the model
becomes more nonlinear or the baseline $x_0$ diverges from the background
distribution, the two increasingly disagree — reflecting genuine differences in
the questions they answer rather than errors in either method.

### Small runtime comparison

TreeSHAP is widely recognized to be fast, even for complex trees and large data
sets. TreeIG can match or exceed that speed because it solves a smaller
computational problem: it sums the prediction changes at the boundaries crossed
by one path.

The following deliberately narrow benchmark uses two standard scikit-learn
regressors: a 200-tree depth-3 gradient-boosting ensemble and a 200-tree
depth-12 extremely randomized trees (ExtraTrees) ensemble. Both explainers use
the same single median training row as their reference and attribute raw model
output. TreeSHAP is
`shap.TreeExplainer(..., feature_perturbation="interventional")`, which is exact
for this configuration. The common reference predictions agreed within
$7.2\times10^{-15}$, and both methods reconstructed predictions within
$1.3\times10^{-5}$.

| Model | Rows explained | TreeIG | Exact TreeSHAP | TreeIG speedup |
|---|---:|---:|---:|---:|
| scikit-learn gradient boosting | 100 | 0.69 ms | 0.86 ms | 1.2x |
| scikit-learn gradient boosting | 1,000 | 4.53 ms | 8.84 ms | 1.9x |
| Extremely randomized trees (ExtraTrees) | 100 | 3.11 ms | 29.30 ms | 9.4x |
| Extremely randomized trees (ExtraTrees) | 1,000 | 25.36 ms | 273.41 ms | 10.8x |

These are medians of seven warmed attribution calls; model fitting, explainer
construction, TreeIG's Numba compilation, and validation are outside the timed
region. Results were measured on an Apple M5 (10 cores) with Python 3.13.11,
TreeIG 0.1.11, scikit-learn 1.9.0, and SHAP 0.52.0, using each library's default
runtime threading. The seeded 20-feature data, both model definitions, checks,
and timing code are in
[`benchmarks/treeig_vs_treeshap.py`](benchmarks/treeig_vs_treeshap.py); rerun
with `python -m benchmarks.treeig_vs_treeshap`.

[XGBoost](https://xgboost.readthedocs.io/en/stable/prediction.html) and
[LightGBM](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html)
also provide tightly integrated native TreeSHAP routines. These can bring
TreeSHAP much closer to TreeIG's runtime; in a companion check, native XGBoost
TreeSHAP and TreeIG were essentially tied for 100 rows when the XGBoost
`DMatrix` was prepared before timing. That is a useful operational comparison,
but not quite like-for-like. Native `tree_path_dependent` TreeSHAP normally uses
training-sample counts stored along the tree paths as an implicit background
distribution, while the interventional TreeSHAP timings above use the same
explicit reference row as TreeIG. More generally, interventional TreeSHAP's
runtime scales with its explicit background sample, while TreeIG traverses and
averages the paths from its supplied baseline distribution.

This is a runtime comparison, not an equivalence claim: sharing a reference
aligns the output scale and reference prediction, but TreeIG and TreeSHAP still
compute the different quantities described above. Nor does the table claim that
TreeIG is always faster: TreeSHAP can have lower overhead for very small models
or batches. The point is that TreeIG remains competitive—and can be materially
faster—on the larger, slower attribution workloads where runtime matters most.
Exact timings depend on model shape, batch size, hardware, versions, and thread
settings.

## Examples

### XGBoost regression

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
result = ig.explain(X_eval)

print(result.values.shape)
print(result.max_abs_completeness_error)
```

### Multiclass classification margins

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

### Model-agnostic attribution

```python
import treeig as tig

ig = tig.TreeIGNumeric(model, baseline=x0)
result = ig.explain(X_eval)

print(result.max_abs_completeness_error)
```

## Project status

TreeIG is production-ready for exact attribution of supported tree models in raw-output space. The current release covers the dominant tree ensemble backends in the Python ecosystem. TreeIGNumeric provides a model-agnostic fallback for unsupported piecewise-constant models.

Future extensions may include:

- exact structural support for CatBoost and other currently unsupported tree implementations;
- customized handling of categorical split structures and missing-value routing;
- alternative allocation rules for simultaneous multi-feature effects at coincident crossings.

## Citation

If you use TreeIG in your work, please cite:

```bibtex
@misc{hentschel2026treeig,
  author = {Hentschel, Ludger},
  title  = {{TreeIG}: Exact Integrated Gradients for Tree-Based Models},
  year   = {2026},
  url    = {https://www.ludgerhentschel.com/PDFs/Hentschel%20'26g.pdf},
}
```

## License

TreeIG is released under the terms in [LICENSE](LICENSE).

## References

TreeIG:

- Hentschel, Ludger. 2026.
  ["TreeIG: Exact Integrated Gradients for Tree-Based Models."](https://www.ludgerhentschel.com/PDFs/Hentschel%20'26g.pdf)
  *https://www.ludgerhentschel.com/Research.html* and *https://www.ludgerhentschel.com/Programs.html*

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
