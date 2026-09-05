# TreeIG documentation

TreeIG computes exact Integrated Gradients for supported tree models by summing
prediction jumps along paths from a baseline to each observation. The CPU
`TreeIG` class is the main interface.

Start with a runnable example, then choose the baseline distribution and output
scale that express the comparison you want to explain.

The method is developed in Ludger Hentschel's
[**TreeIG: Exact Integrated Gradients for Tree-Based Models**](https://www.ludgerhentschel.com/PDFs/Hentschel%20'26g.pdf).
It builds on Integrated Gradients introduced by Sundararajan, Taly, and Yan in
[**Axiomatic Attribution for Deep Networks** (ICML 2017)](https://proceedings.mlr.press/v70/sundararajan17a.html).

## Why Integrated Gradients works for trees

A tree prediction is constant between splits, so its ordinary gradient is zero
almost everywhere. But the prediction jumps at split boundaries. Those jumps
are the contribution that an ordinary pointwise gradient misses: in the
distributional interpretation, each jump is an impulse whose integral equals
the jump's height.

![A prediction step, its derivative impulse, and its integrated contribution](Figure_TreeGradient.svg)

The top panel shows a single prediction step; the middle shows its derivative
as an impulse at the split; the bottom shows the accumulated contribution.
Integrating across the split recovers the prediction change. TreeIG applies
this idea along the path from a baseline to an observation, assigning each
crossing's jump to its split feature and summing across trees.

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
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
references
building
```
