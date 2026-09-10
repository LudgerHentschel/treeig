# TreeIG documentation

**TreeIG computes exact Integrated Gradients for tree models. A tree's gradient
is zero almost everywhere; its integrated gradient is not.**

Tree ensembles are piecewise constant, so $\nabla F = 0$ except on a
measure-zero set of split boundaries. Numerical Integrated Gradients therefore
recovers approximately nothing, which is why IG has largely been confined to
differentiable models.

The pointwise gradient is not the full derivative. In the distributional sense,
$F'$ carries an impulse at each split boundary whose integral equals the
prediction jump there.

![A prediction step, its derivative impulse, and its integrated contribution](Figure_TreeGradient.svg)

The top panel shows a single prediction step; the middle shows its derivative
as an impulse at the split; the bottom shows the accumulated contribution.
Integrating across the split recovers the prediction change.

TreeIG enumerates the boundaries crossed by the straight-line path from baseline
to observation, assigns each jump to its split feature, and sums across trees.
No quadrature and no sampling are involved, and completeness

$$\sum_j \phi_j = F(x) - F(x_0)$$

holds to the floating-point precision of the fitted model's own arithmetic.
Weighted baseline distributions are supported directly.

The CPU `TreeIG` class is the main interface. Start with a runnable example,
then choose the baseline distribution and output scale that express the
comparison you want to explain.

The method is developed in Ludger Hentschel's
[**TreeIG: Exact Integrated Gradients for Tree-Based Models**](https://www.ludgerhentschel.com/PDFs/Hentschel%20'26g.pdf).
It builds on Integrated Gradients introduced by Sundararajan, Taly, and Yan in
[**Axiomatic Attribution for Deep Networks** (ICML 2017)](https://proceedings.mlr.press/v70/sundararajan17a.html).

## Explore the guide

Read [getting started](getting-started.md), [baselines](baselines.md),
[supported models](models.md), and [worked examples](examples.md) first.
For more detail, see [results and plotting](explanations.md),
[loss attribution](loss.md), [numerical conventions](concepts.md),
[TreeIGNumeric](numeric.md), and [performance](performance.md).

```{toctree}
:maxdepth: 2
:caption: User guide

getting-started
examples
baselines
models
explanations
loss
concepts
numeric
performance
comparison
gpu
ig-stack
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
references
building
publishing
```
