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

The output reports maximum absolute feature-allocation error, maximum relative
L1 allocation error, maximum completeness error, the fraction of structural
output events merged by the numerical grid, the fraction of paths encountering
an exact zero class probability, and runtime per baseline-to-input path.
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

The quick and targeted matrices were run on August 3, 2026 with Python 3.13.11,
NumPy 2.4.6, and scikit-learn 1.9.0 on Apple silicon. Three random seeds and
eight paths per ordinary scenario were used; the largest full-profile cases
used four paths.

- Completeness errors stayed near machine precision in every run, including
  runs with materially incorrect feature allocations. Completeness is therefore
  necessary but cannot diagnose merged crossings by itself.
- At 1,024 intervals, maximum relative L1 allocation error was generally below
  1% for moderate binary forests, but reached 3.5% across seeds for the
  reference forest, 7.2% for a four-class forest, and 10.9% for a deeper binary
  forest. The 500-feature and 500-tree cases produced 2.2% and 2.3%,
  respectively; a full-depth five-class case produced 7.7%.
- Increasing the grid to 4,096 or 8,192 usually reduced error sharply, but did
  not eliminate it for every seed. No fixed grid can guarantee separation of
  arbitrarily close black-box prediction events.
- Exact zero probabilities occurred on 88--100% of paths for single trees and
  sparse unpruned ensembles in the tested seeds. They were uncommon in larger
  probability-averaging forests, though some deep-forest runs still encountered
  them. Requiring an explicit `probability_floor` is justified.
- Higher grid resolution was not uniformly slower: separating events reduced
  the number of costly coincident-event sweeps. On this machine, representative
  1,024-grid costs ranged from roughly 9--15 ms per path for 50-tree binary
  forests to 35--47 ms for four-class forests and 130--180 ms for 200-tree
  forests. The 500-tree stress case took about 1.65 seconds per path.

These fixed-grid results motivated the adaptive subdivision introduced in
TreeIG 0.1.11. With four refinement levels and the same 1,024-point global
grid, the reference, wide, ExtraTrees, and four-class scenarios fell to zero or
near-zero oracle error in the representative seed. The large forest improved
slightly; errors caused by exactly offsetting events inside an unchanged coarse
interval were, as expected, unaffected.

Across three CatBoost seeds, the 1,024-point adaptive method matched an
8,192-point raw-score reference to displayed precision for both regression and
four-class classification. A 128-point adaptive scan was already exact in two
seeds and had at most 1.4% relative L1 error in the third. Adaptive evaluation
was generally faster than fixed-grid evaluation because it avoided expensive
unresolved-event sweeps.

These results support `grid_size=1024` and `max_refine=4` as practical defaults
with an explicit convergence check for important probability-forest work.
