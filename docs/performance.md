# Performance and repeated calls

TreeIG uses Numba for fast parallel attribution kernels. The first call
includes JIT compilation. You can compile in advance with `warmup`:

```python
ig = tig.TreeIG(model, baseline=x0).warmup(X_eval[:3])
phi = ig.attribute(X_eval)
```

Subsequent calls on the same model are fast. Attribution for thousands of
observations on a typical ensemble completes in well under a second after
warmup.

## Measuring your workload

Separate model fitting, explainer construction, and the first compilation call
from repeated attribution timing. Reuse the explainer with the same fitted model.
Use several warmed repetitions and report the batch size, baseline count, model
shape, package versions, hardware, and thread settings alongside timings.

Use `attribute()` when only the values are needed. `explain()` also evaluates
model outputs to provide baseline values and completeness errors. More baselines
mean more paths; use batching when memory, rather than latency, is the constraint.

See [TreeIG and TreeSHAP](comparison.md) for the recorded CPU comparison and the
[benchmark notes](https://github.com/LudgerHentschel/treeig/blob/main/benchmarks/README.md)
for additional measured workloads. Timings are examples, not performance guarantees.
