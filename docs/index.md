---
myst:
  html_meta:
    description: "TreeIG computes exact Integrated Gradients for supported numeric tree models, with weighted baselines and a separate numerical fallback."
---

# TreeIG documentation

TreeIG is a Python package for Integrated Gradients feature attribution on
supported numeric tree models. Install and import it as `treeig`. Given a fitted
model, a baseline point or weighted background, and evaluation rows, `TreeIG`
returns feature contributions and completeness diagnostics.

**TreeIG computes exact Integrated Gradients for supported numeric tree models.
A tree's gradient is zero almost everywhere; its integrated gradient is not.**

Check [supported models](models.md) before choosing an interface.
`TreeIG` uses exact structural split crossings; [TreeIGNumeric](numeric.md)
is a separately selected numerical fallback. Exact classification explains raw
margins or logits. Exact parsing requires finite numeric inputs and does not
support categorical splits or missing-value routing. Installing the CatBoost
extra does not add an exact CatBoost backend.

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

For automated readers, [llms.txt](https://ludgerhentschel.github.io/treeig/llms.txt)
maps the guides, complete examples, and rendered API reference.

Read [getting started](getting-started.md), [baselines](baselines.md),
[supported models](models.md), and [worked examples](examples.md) first.
For more detail, see [results and plotting](explanations.md),
[loss attribution](loss.md), [numerical conventions](concepts.md),
[TreeIGNumeric](numeric.md), and [performance](performance.md).

## Related projects

| Package | When to use it |
|---|---|
| [UnifiedIG](https://ludgerhentschel.github.io/unifiedig/) (`unifiedig`) | A common Integrated Gradients interface across supported tree and smooth model families. |
| [CBaseline](https://ludgerhentschel.github.io/cbaseline/) (`cbaseline`) | Construct empirical reference distributions; TreeIG accepts its backgrounds with their weights directly. |
| [skgrad](https://ludgerhentschel.github.io/skgrad/) (`skgrad`) | Obtain analytic input gradients and Jacobians for supported smooth scikit-learn models. |

Use TreeIG directly when you need its tree-specific attribution interface.
See [the Integrated Gradients stack](https://ludgerhentschel.github.io/treeig/ig-stack.html)
for how the packages compose and why output scales must agree.

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
