# Probability-forest stress benchmark

`probability_forests.py` validates `TreeIGNumeric` on probability-averaging
scikit-learn classifiers. It varies forest size, depth, feature count, class
count, and path-grid resolution.

The benchmark does more than check endpoint completeness. An independent
oracle traverses every fitted tree along each straight-line path, enumerates
the reachable split crossings, and evaluates the aggregate forest probability
on both sides. The reported allocation error therefore detects merged or
misassigned events even when attribution remains complete overall.

Run the practical matrix from the repository root:

```bash
python -m benchmarks.probability_forests
```

For larger forests, wider inputs, and five-class models:

```bash
python -m benchmarks.probability_forests --profile full --eval 16 \
    --grids 64 256 1024 2048
```

The primary accuracy summaries are median, 95th-percentile, and pooled
effective-leaf-support-weighted relative L1 allocation error. For each
structural output event, effective leaf support is the harmonic mean of the
weighted sample counts in the leaves immediately before and after the crossing.
Path support then averages event support using absolute output-jump size as
weight. This emphasizes attribution accuracy where both the fitted probability
estimates and the attributed effects have the strongest data support, without
hiding the upper tail. The output also reports maximum completeness error, the
fraction of structural output events merged by the numerical grid, the fraction
of paths encountering an exact zero class probability, and runtime per
baseline-to-input path.
Individual scenarios can be selected with `--scenarios reference multiclass`.
Set `--max-refine 0` to reproduce fixed-grid behavior without adaptive
subdivision.
Results depend on the fitted models and hardware; rerun the benchmark when
changing numerical-event logic or defaults.

With CatBoost installed, compare fixed and adaptive grids against a
high-resolution raw-score reference:

```bash
python -m benchmarks.catboost_adaptive
```

## Representative findings

The quick and targeted matrices were run on August 3--4, 2026 with Python 3.13.11,
NumPy 2.4.6, and scikit-learn 1.9.0 on Apple silicon. Three random seeds and
eight paths per ordinary scenario were used; the largest full-profile cases
used four paths.

- Completeness errors stayed near machine precision in every run, including
  runs with materially incorrect feature allocations. Completeness is therefore
  necessary but cannot diagnose merged crossings by itself.
- With the 1,024-point adaptive default, the reference binary and moderate
  four-class forests had support-weighted error no greater than 0.05% across
  three seeds. For the 200-tree forest, support-weighted error was at most
  0.35% and 95th-percentile error was at most 0.84%.
- The deep binary forest had median error between 0% and 1.72%,
  support-weighted error between 0.39% and 2.57%, and 95th-percentile error
  between 2.31% and 8.86%. Its median effective leaf support was only 11--21,
  compared with roughly 33--47 in the moderate forests.
- The sparse four-class forest had zero median error in all three seeds.
  Support-weighted error ranged from 0% to 3.62%, while 95th-percentile error
  reached 16.86% in one seed. Median effective leaf support was only 5--11.
  This is precisely the setting in which a worst-case statistic overstates the
  practical importance of allocation error.
- Exact zero probabilities occurred on 88--100% of paths for single trees and
  sparse unpruned ensembles in the tested seeds. They were uncommon in larger
  probability-averaging forests, though some deep-forest runs still encountered
  them. Requiring an explicit `probability_floor` is justified.
- Higher grid resolution was not uniformly slower: separating events reduced
  the number of costly coincident-event sweeps. Adaptive 1,024-grid costs were
  roughly 9--12 ms per path for 50-tree binary forests, 45--48 ms for
  four-class forests, and 50--63 ms for the tested 200-tree forests.

Earlier fixed-grid stress results motivated the adaptive subdivision introduced
in TreeIG 0.1.11. With four refinement levels and the same 1,024-point global
grid, the reference, wide, ExtraTrees, and four-class scenarios fell to zero or
near-zero oracle error in the representative seed. Errors caused by exactly
offsetting events inside an unchanged coarse interval remain undetectable.

Across three CatBoost seeds, the 1,024-point adaptive method matched an
8,192-point raw-score reference to displayed precision for both regression and
four-class classification. A 128-point adaptive scan was already exact in two
seeds and retained a small upper-tail discrepancy in the third. Current
benchmark output emphasizes median and 95th-percentile error instead of
treating a single extreme as the primary measure. Adaptive evaluation was
generally faster than fixed-grid evaluation because it avoided expensive
unresolved-event sweeps.

These results support `grid_size=1024` and `max_refine=4` as practical defaults
with an explicit convergence check for important probability-forest work.

## Weighted-baseline CPU experiment

`weighted_baselines_cpu.py` measures distributional attribution over
`n = {1, 10, 100, 1000}` observations and `B = {1, 10, 100}` baselines for a
shallow boosting ensemble and a deeper forest. It also checks completeness and
can save every attribution array for numerical comparison between revisions.

An August 2026 experiment removed the per-baseline dense attribution reduction
and added a second kernel parallelized over observation-baseline pairs. On a
10-thread CPU, the pair kernel improved the intended `n=1, B=100` workload by
about 1.35x for shallow boosting and 1.58x for the deeper forest. The remaining
matrix was largely flat, noisy, or slightly slower. Before/after attributions
agreed to a maximum absolute difference of `1.51e-14` across all 24 workloads.

The production change was rejected: the isolated small-batch gain did not
justify a second compiled kernel, thread-dependent dispatch, per-pair scratch
storage, and an additional reduction. Future weighted-baseline optimization
should target traversal and scratch traffic while preserving the endpoint and
boundary behavior covered by the focused regression tests.

## CUDA prediction attribution

`cuda_prediction.py` compares the unchanged CPU backend with both the stateless
CUDA prototype and public persistent `GPUTreeIG`. The CUDA path parallelizes
the complete observation x baseline x tree product and retains the CPU interval
traversal, segment ordering, and endpoint ownership policy. The stateless
measurement includes device allocation and all fixed-state transfers. The
persistent measurement keeps the model, baseline distribution, weights, and
baseline-tree cache resident, reuses device work buffers, and transfers
each observation batch, output-buffer initialization, and results. JIT compilation, model parsing, and
baseline-leaf preparation are warmed outside all timings.

GPU use remains an explicit choice through `GPUTreeIG`; ordinary `TreeIG` never
automatically dispatches to it. The local Apple-silicon development host has no
CUDA device, so it validates semantics through `NUMBA_ENABLE_CUDASIM=1`;
simulator timings are not performance evidence.

Each CUDA thread owns model-specialized local DFS and segment scratch. The DFS
stack is bounded by maximum tree depth plus one; the segment buffer is bounded
by maximum leaf count. Models requiring either power-of-two width above 1,024
entries are rejected explicitly.

### Colab T4 results

The prototype was run on a free Google Colab Tesla T4 on August 28, 2026.
The matrix used 100 trees, 12 features, weighted baselines, one warmed timing
sample per cell, and transfer-inclusive GPU timings. CPU and GPU attributions
agreed to maximum absolute error below `1.7e-14` in every cell.

| Model | n | B | CPU ms | GPU ms | Speedup |
|---|---:|---:|---:|---:|---:|
| boosting, depth 3 | 10 | 100 | 54.30 | 6.97 | 7.79x |
| boosting, depth 3 | 100 | 10 | 69.84 | 7.56 | 9.24x |
| boosting, depth 3 | 100 | 100 | 467.03 | 19.36 | 24.12x |
| boosting, depth 3 | 1000 | 10 | 410.63 | 28.40 | 14.46x |
| boosting, depth 3 | 1000 | 100 | 3242.77 | 140.73 | 23.04x |
| forest, depth 6 | 10 | 100 | 44.26 | 12.62 | 3.51x |
| forest, depth 6 | 100 | 10 | 42.61 | 12.70 | 3.35x |
| forest, depth 6 | 100 | 100 | 437.08 | 97.67 | 4.47x |
| forest, depth 6 | 1000 | 10 | 435.94 | 97.29 | 4.48x |
| forest, depth 6 | 1000 | 100 | 5503.32 | 943.98 | 5.83x |

Small workloads remain CPU-favorable because launch, allocation, and transfer
costs dominate. The shallow model demonstrates 10x-class gains once the task
grid is large enough. The deeper forest's lower ceiling supports the expected
local-scratch and occupancy concern; reducing per-thread state or grouping
work cooperatively is the next optimization target.

A follow-up production-style run kept the model, baseline distribution, and
baseline-tree cache resident through `GPUTreeIG`, reused its device buffers,
and reported the median of three warmed calls:

| Model | n | B | CPU ms | Persistent GPU ms | Speedup |
|---|---:|---:|---:|---:|---:|
| boosting, depth 3 | 10 | 100 | 26.00 | 2.69 | 9.68x |
| boosting, depth 3 | 100 | 10 | 24.95 | 2.71 | 9.22x |
| boosting, depth 3 | 100 | 100 | 252.24 | 15.22 | 16.58x |
| boosting, depth 3 | 1000 | 10 | 259.69 | 15.22 | 17.06x |
| boosting, depth 3 | 1000 | 100 | 2415.75 | 137.41 | 17.58x |
| forest, depth 6 | 10 | 100 | 46.65 | 10.97 | 4.25x |
| forest, depth 6 | 100 | 10 | 41.06 | 10.75 | 3.82x |
| forest, depth 6 | 100 | 100 | 419.78 | 95.58 | 4.39x |
| forest, depth 6 | 1000 | 10 | 425.98 | 95.70 | 4.45x |
| forest, depth 6 | 1000 | 100 | 4447.77 | 940.57 | 4.73x |

These timings used a fresh random dataset and repeated measurements, so they
should not be compared cell-for-cell with the earlier single-sample stateless
matrix. They establish the intended deployment result: persistent execution
makes 10x-class shallow-ensemble gains available at smaller batches, while it
does not remove the deeper-tree kernel bottleneck. Maximum CPU/GPU attribution
difference was below `2.0e-14` throughout.

### Deep-tree scratch specialization

The initial kernel gave all seven local arrays the same power-of-two width
based on twice the packed node count. A full depth-6 tree therefore used width
256 for every stack and segment array. The revised kernel derives independent
bounds without changing traversal or event semantics: a DFS has at most
`maximum_depth + 1` pending entries, and a leaf's unique root path emits at
most one segment. The depth-6 specialization is consequently `(8, 64)` for
stack and segment widths.

On the same Colab T4, dataset, model seed, and three-repeat persistent workload
used for the earlier depth-6 run, GPU latency changed as follows:

| n | B | Before ms | Specialized ms | Kernel improvement | CPU speedup |
|---:|---:|---:|---:|---:|---:|
| 10 | 100 | 10.97 | 3.74 | 2.94x | 11.17x |
| 100 | 10 | 10.75 | 3.53 | 3.04x | 11.82x |
| 100 | 100 | 95.58 | 22.63 | 4.22x | 17.02x |
| 1000 | 10 | 95.70 | 23.64 | 4.05x | 18.83x |
| 1000 | 100 | 940.57 | 207.96 | 4.52x | 19.68x |

The same implementation reached 8.76--12.84x CPU speedup for a depth-8 forest
with `(16, 128)` scratch widths. A depth-10 forest with `(16, 256)` widths
reached 7.99x at `n=10, B=100`, 9.00x at `n=100, B=100`, and 10.46x at
`n=1000, B=100`. Maximum absolute CPU/GPU difference across these runs was
`2.62e-14`.

This removes local scratch allocation as the dominant depth-6 bottleneck and
meets the 10x target for large workloads through depth 10. The next likely
limits are the growing leaf-segment buffer and global atomic attribution
updates. Neither segment sorting nor endpoint probes were changed in this
optimization.
