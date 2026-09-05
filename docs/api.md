# API reference

The primary interfaces are `TreeIG` for exact structural attribution,
`TreeIGNumeric` for numerical fallback, and `Explanation` for results.
See the user guide for weighted-baseline shapes and runnable examples.

```{eval-rst}
.. autoclass:: treeig.TreeIG
   :members: attribute, explain, model_output, diagnostics, trace, loss_attribution, multiclass_loss_attribution, warmup

.. autoclass:: treeig.TreeIGNumeric
   :members: attribute, explain, model_output, diagnostics

.. autoclass:: treeig.Explanation
   :members: to_shap, max_abs_completeness_error

.. autofunction:: treeig.compute

.. autofunction:: treeig.compute_numeric

.. autofunction:: treeig.supports
```

The optional GPU interface is described separately in [GPU support](gpu.md).
