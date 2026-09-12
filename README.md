# TreeIG

[![PyPI version](https://img.shields.io/pypi/v/treeig.svg)](https://pypi.org/project/treeig/)
[![Documentation](https://img.shields.io/badge/docs-user%20guide-blue)](https://ludgerhentschel.github.io/treeig/)

**TreeIG computes exact Integrated Gradients for supported numeric tree models.
A tree's gradient is not well defined; its integrated gradient is.**

Tree ensembles are piecewise constant, so $\nabla F = 0$ except on a
measure-zero set of split boundaries. Numerical Integrated Gradients therefore
recovers approximately nothing, which is why IG has largely been confined to
differentiable models.

The pointwise gradient is not the full derivative. In the distributional sense,
$F'$ carries an impulse at each split boundary whose integral equals the
prediction jump there.

![A prediction step, its derivative impulse, and its integrated contribution](https://raw.githubusercontent.com/LudgerHentschel/treeig/main/docs/Figure_TreeGradient.svg)

The top panel shows a single prediction step; the middle shows its derivative
as an impulse at the split; the bottom shows the accumulated contribution.
Integrating across the split recovers the prediction change.

TreeIG enumerates the boundaries crossed by the straight-line path from baseline
to observation, assigns each jump to its split feature, and sums across trees.
No quadrature and no sampling are involved, and completeness

$$\sum_j \phi_j = F(x) - F(x_0)$$

holds to the floating-point precision of the fitted model's own arithmetic.
Weighted baseline distributions are supported directly.

The method is developed in Ludger Hentschel's
[**TreeIG: Exact Integrated Gradients for Tree-Based Models**](https://www.ludgerhentschel.com/PDFs/Hentschel%20'26g.pdf).
It builds on Integrated Gradients introduced by Sundararajan, Taly, and Yan in
[**Axiomatic Attribution for Deep Networks** (ICML 2017)](https://proceedings.mlr.press/v70/sundararajan17a.html).

## Installation

```bash
pip install "treeig[sklearn]"
```

Requires Python 3.9 or later, NumPy, and Numba. Install the model library you use;
extras include `sklearn`, `xgboost`, `lightgbm`, and `catboost`. SHAP is optional
for plotting. The first attribution call includes Numba compilation.

## Quickstart

This snippet assumes a fitted supported model and numeric evaluation data.
For a standalone example that creates data, fits a model, and checks prediction
reconstruction, start with [the complete quickstart](https://ludgerhentschel.github.io/treeig/getting-started.html).

```python
from treeig import TreeIG

# A representative training row provides a simple reference.
ig = TreeIG(model, baseline=X_train[0])
result = ig.explain(X_eval)
phi = result.values
print(result.max_abs_completeness_error)
```

`phi` has one row per observation and one column per feature. Positive values
increase the explained output relative to the baseline; negative values decrease
it. Use `ig.attribute(X_eval)` when only the attribution array is needed.

The baseline defines the comparison. For substantive attribution,
[CBaseline](https://github.com/LudgerHentschel/cbaseline) is the recommended way
to construct a prediction-neutral baseline distribution. TreeIG accepts its
`Background` directly as `baseline=background`, or a matrix of rows with
`baseline_weights`. See the [baseline guide](https://ludgerhentschel.github.io/treeig/baselines.html).

## Model support and interpretation

Exact backends cover selected scikit-learn tree regressors and gradient boosting,
XGBoost, and LightGBM. Regression explains predictions; classification explains
raw margins, not probabilities. Inputs must be finite and numeric; categorical
splits and missing-value routing are not supported by the exact parser.

`TreeIGNumeric` provides a numerical fallback for other piecewise-constant models,
including numeric-input CatBoost and probability-only classifiers. Its resolution
requires care. For probability-only classifiers, it defaults to binary log odds
or centered multiclass log scores; class probabilities require explicit
`probability_to_score=False`. Zero probabilities require an explicit
`probability_floor` for score conversion. A small completeness
residual alone does not establish accurate individual feature allocations. See [supported models](https://ludgerhentschel.github.io/treeig/models.html)
and [the numerical guide](https://ludgerhentschel.github.io/treeig/numeric.html).

TreeIG and TreeSHAP answer different attribution questions. TreeIG can be fast
on substantial attribution workloads, but relative speed depends on the model,
baselines, and batch size. The [comparison and benchmarks](https://ludgerhentschel.github.io/treeig/comparison.html)
explain the distinction and report measured examples.

## Documentation

For automated readers, [llms.txt](https://ludgerhentschel.github.io/treeig/llms.txt)
maps the guides, complete examples, and rendered API reference.

The [user guide](https://ludgerhentschel.github.io/treeig/)
covers a complete runnable example, baseline distributions, classification,
plots, loss attribution, numerical conventions, and performance. The Sphinx
sources also build into searchable HTML with an API reference; see
[building the documentation](https://ludgerhentschel.github.io/treeig/building.html).

## Optional GPU support

`TreeIG` is already fast enough for most applications and remains the default.
When attribution speed matters and an NVIDIA GPU is available, `GPUTreeIG` can
be materially faster; recorded T4 comparisons show roughly 9–20× speedups on
the reported workloads. Performance depends on the problem. See
[GPU documentation](https://ludgerhentschel.github.io/treeig/gpu.html)
for installation and limitations.

## Related projects

| Package | When to use it |
|---|---|
| [UnifiedIG](https://ludgerhentschel.github.io/unifiedig/) (`unifiedig`) | A common Integrated Gradients interface across supported tree and smooth model families. |
| [CBaseline](https://ludgerhentschel.github.io/cbaseline/) (`cbaseline`) | Construct empirical reference distributions; TreeIG accepts its backgrounds with their weights directly. |
| [skgrad](https://ludgerhentschel.github.io/skgrad/) (`skgrad`) | Obtain analytic input gradients and Jacobians for supported smooth scikit-learn models. |

Use TreeIG directly when you need its tree-specific attribution interface.
See [the Integrated Gradients stack](https://ludgerhentschel.github.io/treeig/ig-stack.html)
for how the packages compose and why output scales must agree.

## Citation and license

If you use TreeIG in your work, please cite the
[TreeIG paper](https://www.ludgerhentschel.com/PDFs/Hentschel%20'26g.pdf):

```bibtex
@misc{hentschel2026treeig,
  author = {Hentschel, Ludger},
  title  = {{TreeIG}: Exact Integrated Gradients for Tree-Based Models},
  year   = {2026},
  url    = {https://www.ludgerhentschel.com/PDFs/Hentschel%20'26g.pdf},
}
```

Released under the [BSD-3-Clause license](https://github.com/LudgerHentschel/treeig/blob/main/LICENSE).

Release maintainers: see [Publishing releases](docs/publishing.md).
