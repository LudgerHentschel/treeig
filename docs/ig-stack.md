# The Integrated Gradients Stack

| Package | Responsibility |
|---|---|
| [CBaseline](https://github.com/LudgerHentschel/cbaseline) | Construct empirical reference baseline distributions |
| [skgrad](https://github.com/LudgerHentschel/skgrad) | Evaluate analytic input derivatives of supported sklearn functions |
| [TreeIG](https://github.com/LudgerHentschel/treeig) | Account for prediction jumps along supported tree paths |
| [UnifiedIG](https://github.com/LudgerHentschel/unifiedig) | Choose backends and compute feature attributions |

An input gradient describes local sensitivity. Integrated Gradients also needs
a baseline and integration along a path. skgrad supplies the derivative needed
inside that integration; its metadata lets consumers recognize constant
Jacobians or known polynomial quadrature orders.

The [polynomial integration example](https://ludgerhentschel.github.io/skgrad/examples.html#integrating-a-polynomial-gradient)
shows this composition explicitly and checks that feature contributions sum to
the prediction difference. It is an educational example for a single baseline.
Use UnifiedIG for the complete attribution interface and baseline distributions.

Keep output scales aligned across packages. Differentiating decision scores,
logits, and probabilities describes different contrasts. In particular, do not
combine a score gradient with a probability-valued baseline prediction.

## Explore the packages

- [CBaseline documentation](https://ludgerhentschel.github.io/cbaseline/):
  construct the reference distribution for the prediction contrast.
- [skgrad documentation](https://ludgerhentschel.github.io/skgrad/):
  evaluate analytic gradients, including supported preprocessing pipelines.
- [TreeIG documentation](https://ludgerhentschel.github.io/treeig/):
  compute Integrated Gradients for supported tree models.
- [UnifiedIG guide](https://github.com/LudgerHentschel/unifiedig/blob/main/docs/how-it-works.md):
  select the attribution backend through a common interface.
