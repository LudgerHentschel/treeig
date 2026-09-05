# Explaining loss reduction

Prediction attribution explains a model output. Loss attribution instead explains
how features reduce loss relative to the baseline prediction, using the observed
outcome for each row. These methods are available on CPU `TreeIG`.

## Regression

Given a fitted regression model, evaluation rows `X_eval`, outcomes `y_eval`,
and a baseline `x0`:

```python
from treeig import TreeIG

ig = TreeIG(model, baseline=x0)
loss = ig.loss_attribution(X_eval, y_eval, loss="squared_error")
print(loss["values"])
print(loss["total"])
```

`observation_values` has one row of feature contributions per observation.
`values` averages those rows; `standard_errors` describes uncertainty of that
mean across the supplied observations. It does not include model-fitting or
baseline-selection uncertainty. `total` is `baseline_loss - model_loss`.
Positive contributions reduce loss; negative contributions increase it.

## Classification

Use `loss="log_loss"` for a binary margin model, with labels encoded as 0 and 1
and the positive-class margin (`target=None` or `target=1`). Do not substitute
probabilities for margins.

```python
ig = TreeIG(binary_model, baseline=x0, target=1)
loss = ig.loss_attribution(X_eval, y_binary, loss="log_loss")
```

For multiclass models, use the dedicated method so that changes across all
class margins are combined before the softmax loss is evaluated:

```python
ig = TreeIG(multiclass_model, baseline=x0, target=0)
loss = ig.multiclass_loss_attribution(X_eval, y_class_indices)
```

Labels are class indices, not arbitrary class names. Supply `n_classes` for a
model without `classes_`. The returned aggregate fields follow the same
loss-reduction convention as regression.

## Baseline distributions

Both methods accept baseline rows and weights, including a constructor default.
They average the loss reductions from the individual baseline paths. This is
not generally the loss reduction from the mean baseline prediction. Loss
attribution operates on the sequence of prediction changes; applying a loss
function to already aggregated prediction attributions is not equivalent.
