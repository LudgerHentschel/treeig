# GPUTreeIG: optional CUDA execution

`TreeIG` on the CPU is the primary tool and recommended starting point. The
`treeig` package also provides `GPUTreeIG`, an optional CUDA implementation for
large, repeated prediction-attribution workloads with a fixed model and baseline
distribution. It uses the same attribution semantics as `TreeIG`, up to
floating-point accumulation differences. It is not a separate package.

On a host with an NVIDIA CUDA GPU, install the optional CUDA support:

```bash
pip install "treeig[cuda]"
```

This installs `numba-cuda`; a compatible NVIDIA driver and CUDA toolkit are also
required. Follow the [Numba CUDA installation instructions](https://nvidia.github.io/numba-cuda/user/installing.html)
for your platform. CUDA is loaded only when constructing `GPUTreeIG`; ordinary
`TreeIG` use requires no CUDA installation. Apple Metal and AMD ROCm are not
supported by this backend. If CUDA is unavailable, `GPUTreeIG` raises an error;
choose `TreeIG` for CPU execution.

```python
import treeig as tig

gpu_ig = tig.GPUTreeIG(
    model,
    baseline=background.rows,
    baseline_weights=background.weights,
)
phi = gpu_ig.attribute(large_batch, batch_size=1000)
```

Construction uploads the model, baseline distribution, and weights. Reuse the
instance across calls to retain that state and reuse device buffers. Each call
transfers input rows, initializes output buffers, and retrieves results. Let the
instance go out of scope or use `del gpu_ig` when it is no longer needed; device
memory may remain in the CUDA allocator cache.

`GPUTreeIG` supports `attribute()` and `explain()` for prediction attribution.
The baseline distribution, weights, and target are fixed at construction.
Observation batching is supported; baseline batching, per-baseline output,
loss attribution, traces, diagnostics, and `warmup()` are not. Models requiring
stack or leaf-segment scratch widths above 1,024 entries are rejected.

GPU selection is explicit. Recorded T4 benchmarks show substantial gains for
some large workloads, but a GPU may be slower for a particular problem. Start
with `TreeIG` and compare on your own workload if GPU execution is useful.
The existing validation includes CPU/GPU simulator comparisons and recorded T4
runs; it is not an exhaustive hardware compatibility study. See the
[CUDA benchmark notes](https://github.com/LudgerHentschel/treeig/blob/main/benchmarks/README.md#cuda-prediction-attribution)
for measured workloads, numerical agreement, and limitations.

