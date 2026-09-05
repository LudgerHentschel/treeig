# TreeIG

[![PyPI version](https://img.shields.io/pypi/v/treeig.svg)](https://pypi.org/project/treeig/)

TreeIG computes exact Integrated Gradients for tree-based models. It attributes
the change in a model's scalar output from a baseline to an observation to the
input features. For one baseline $x_0$,

$$\sum_j \phi_j = F(x) - F(x_0).$$

TreeIG finds the split boundaries crossed along the straight-line path and sums
their prediction jumps. For supported models, this avoids numerical integration
and sampling. Weighted baseline distributions are supported as well.

## Why Integrated Gradients works for trees

A tree prediction is constant between splits, so its ordinary gradient is zero
almost everywhere. But the prediction jumps at split boundaries. Those jumps
are the contribution that an ordinary pointwise gradient misses: in the
distributional interpretation, each jump is an impulse whose integral equals
the jump's height.

![A prediction step, its derivative impulse, and its integrated contribution](https://raw.githubusercontent.com/LudgerHentschel/treeig/main/docs/Figure_TreeGradient.svg)

The top panel shows a single prediction step; the middle shows its derivative
as an impulse at the split; the bottom shows the accumulated contribution.
Integrating across the split recovers the prediction change. TreeIG applies
this idea along the path from a baseline to an observation, assigning each
crossing's jump to its split feature and summing across trees.

## Installation

```bash
pip install "treeig[sklearn]"
```

Requires Python 3.9 or later, NumPy, and Numba. Install the model library you use;
extras include `sklearn`, `xgboost`, `lightgbm`, and `catboost`. SHAP is optional
for plotting. The first attribution call includes Numba compilation.

## Quickstart

With a fitted supported model and numeric evaluation data:

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
`baseline_weights`. See the [baseline guide](https://github.com/LudgerHentschel/treeig/blob/main/docs/baselines.md).

## Model support and interpretation

Exact backends cover selected scikit-learn tree regressors and gradient boosting,
XGBoost, and LightGBM. Regression explains predictions; classification explains
raw margins, not probabilities. Inputs must be finite and numeric; categorical
splits and missing-value routing are not supported by the exact parser.

`TreeIGNumeric` provides a numerical fallback for other piecewise-constant models,
including numeric-input CatBoost and probability-only classifiers. Its resolution
requires care. See [supported models](https://github.com/LudgerHentschel/treeig/blob/main/docs/models.md)
and [the numerical guide](https://github.com/LudgerHentschel/treeig/blob/main/docs/numeric.md).

TreeIG and TreeSHAP answer different attribution questions. TreeIG can be fast
on substantial attribution workloads, but relative speed depends on the model,
baselines, and batch size. The [comparison and benchmarks](https://github.com/LudgerHentschel/treeig/blob/main/docs/comparison.md)
explain the distinction and report measured examples.

## Documentation

The [user guide](https://github.com/LudgerHentschel/treeig/blob/main/docs/index.md)
covers a complete runnable example, baseline distributions, classification,
plots, loss attribution, numerical conventions, and performance. The Sphinx
sources also build into searchable HTML with an API reference; see
[building the documentation](https://github.com/LudgerHentschel/treeig/blob/main/docs/building.md).

## Optional GPU support

`GPUTreeIG` is an optional CUDA backend within this package. CPU `TreeIG` remains
the default; GPU performance depends on the workload. See
[GPU documentation](https://github.com/LudgerHentschel/treeig/blob/main/docs/gpu.md)
for installation and limitations.

## Citation and license

If you use TreeIG in your work, please cite:

```bibtex
@misc{hentschel2026treeig,
  author = {Hentschel, Ludger},
  title  = {{TreeIG}: Exact Integrated Gradients for Tree-Based Models},
  year   = {2026},
  url    = {https://www.ludgerhentschel.com/PDFs/Hentschel%20'26g.pdf},
}
```

Released under the [BSD-3-Clause license](https://github.com/LudgerHentschel/treeig/blob/main/LICENSE).
