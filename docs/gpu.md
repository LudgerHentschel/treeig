# GPUTreeIG: optional CUDA execution

`TreeIG` on the CPU is already fast enough for most applications and remains
the default. When attribution speed matters and an NVIDIA GPU is available,
`GPUTreeIG` can be materially faster: the [T4 comparisons below](#t4-versus-cpu-benchmarks)
show roughly 9–20× speedups on the reported workloads. The gain depends on the
problem, and small workloads can still favor the CPU.

The
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

## T4 versus CPU benchmarks

The following recorded Google Colab Tesla T4 comparisons illustrate workloads
where GPU execution helped. They are examples, not a promise of speedup on other
hardware or problems. CPU `TreeIG` remains the default.

Here `n` is the number of observations and `B` is the number of weighted baseline
rows. The persistent explainer retains its model and baseline state on the GPU
and reuses its work buffers. Timings are medians of three warmed calls; model
fitting, parsing, baseline preparation, and JIT compilation are outside the timed
region. GPU calls include observation transfers and result retrieval.

### Shallow gradient boosting

The recorded persistent run used a 100-tree, depth-3 boosting ensemble:

| Observations (n) | Baselines (B) | CPU (ms) | T4 (ms) | CPU time / T4 time |
|---:|---:|---:|---:|---:|
| 10 | 100 | 26.00 | 2.69 | 9.68× |
| 100 | 10 | 24.95 | 2.71 | 9.22× |
| 100 | 100 | 252.24 | 15.22 | 16.58× |
| 1,000 | 10 | 259.69 | 15.22 | 17.06× |
| 1,000 | 100 | 2,415.75 | 137.41 | 17.58× |

Maximum absolute CPU/GPU attribution difference was below `2.0e-14` in the
recorded persistent comparison.

### Depth-6 random forest

A later run used the current, smaller per-thread scratch buffers. The recorded
comparisons for the 100-tree forest were:

| Observations (n) | Baselines (B) | CPU (ms) | T4 (ms) | CPU time / T4 time |
|---:|---:|---:|---:|---:|
| 10 | 100 | ≈41.78 | 3.74 | 11.17× |
| 100 | 10 | ≈41.72 | 3.53 | 11.82× |
| 100 | 100 | ≈385.16 | 22.63 | 17.02× |
| 1,000 | 10 | ≈445.14 | 23.64 | 18.83× |
| 1,000 | 100 | ≈4,092.65 | 207.96 | 19.68× |

CPU times marked ≈ are reconstructed as T4 time × recorded speedup. The
notes retain these two rounded quantities rather than raw CPU timings for this
later run, so the reconstructed CPU times are approximate. Across the scratch-specialization runs, which also included
depth-8 and depth-10 forests, the maximum absolute CPU/GPU difference was
`2.62e-14`.

Small workloads can favor the CPU because GPU launch and transfer costs dominate.
Compare warmed calls on your own model and baseline distribution when deciding
whether to use the optional backend. CUDA simulator checks validate semantics;
simulator timings are not GPU performance evidence.

The [full CUDA benchmark notes](gpu-benchmarks.md) preserve the earlier stateless
and persistent measurements and the scratch-buffer comparison. The
[benchmark script](https://github.com/LudgerHentschel/treeig/blob/main/benchmarks/cuda_prediction.py)
is available in the repository. These are historical measurements, not new runs
performed for this documentation update.

```{toctree}
:hidden:

gpu-benchmarks
```
