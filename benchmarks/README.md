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
