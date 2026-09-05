# TreeIG and TreeSHAP

TreeIG and TreeSHAP answer different attribution questions and generally produce
different decompositions. Neither dominates the other.

**TreeIG** answers: "How much does feature $j$ contribute to the change in
prediction as we move continuously from baseline $x_0$ to observation $x$?" The
attribution is the integral of partial derivatives along the path from $x_0$ to
$x$, which for piecewise-constant trees reduces exactly to a sum of prediction
jumps at the split boundaries crossed along the path.

**TreeSHAP** answers: "How much does feature $j$ shift the expected prediction,
averaged over all possible subsets of the other features?" The attribution is an
average of discrete inclusion effects, where absent features are marginalized
out over a background dataset. There is no path; the reference is the expected
prediction over the background distribution.

TreeIG can use either one baseline input or a weighted baseline distribution.
With a distribution, it averages the contributions along the individual paths.
The central distinction is how contributions are allocated: TreeIG
measures contributions through calculus — integrating how the prediction changes
as features move continuously from their baseline values — while TreeSHAP
measures them through discrete feature inclusion, asking how much each feature
changes the expected prediction when it enters a coalition.

The practical consequence is one of scope. SHAP's coalition construction is
indifferent to the prediction surface between the background and the
observation: a feature is either in the coalition or out, so the attribution is
built from discrete switches and explores a wide neighborhood of hybrid inputs,
many far from any natural path between real observations. IG instead follows a
single path and accumulates exactly the prediction changes along it, evaluating
the model only at convex combinations of two real inputs. SHAP explores a
neighborhood; IG traces a path. SHAP's breadth gives sensitivity to model
behavior across many feature combinations; IG's specificity gives a precise
account of one trajectory through input space.

For a linear model with independent features and $x_0$ equal to the background
mean, TreeIG and interventional SHAP coincide. (A linear model is not a tree, so
the comparison is to SHAP generally rather than to TreeSHAP.) As the model
becomes more nonlinear or the baseline $x_0$ diverges from the background
distribution, the two increasingly disagree — reflecting genuine differences in
the questions they answer rather than errors in either method.

## Small runtime comparison

TreeSHAP is widely recognized to be fast, even for complex trees and large data
sets. TreeIG can match or exceed that speed because it solves a smaller
computational problem: it sums the prediction changes at the boundaries crossed
by one path.

The following deliberately narrow benchmark uses two standard scikit-learn
regressors: a 200-tree depth-3 gradient-boosting ensemble and a 200-tree
depth-12 extremely randomized trees (ExtraTrees) ensemble. Both explainers use
the same single median training row as their reference and attribute raw model
output. TreeSHAP is
`shap.TreeExplainer(..., feature_perturbation="interventional")`, which is exact
for this configuration. The common reference predictions agreed within
$7.2\times10^{-15}$, and both methods reconstructed predictions within
$1.3\times10^{-5}$.

| Model | Rows explained | TreeIG | Exact TreeSHAP | TreeIG speedup |
|---|---:|---:|---:|---:|
| scikit-learn gradient boosting | 100 | 0.69 ms | 0.86 ms | 1.2x |
| scikit-learn gradient boosting | 1,000 | 4.53 ms | 8.84 ms | 1.9x |
| Extremely randomized trees (ExtraTrees) | 100 | 3.11 ms | 29.30 ms | 9.4x |
| Extremely randomized trees (ExtraTrees) | 1,000 | 25.36 ms | 273.41 ms | 10.8x |

These are medians of seven warmed attribution calls; model fitting, explainer
construction, TreeIG's Numba compilation, and validation are outside the timed
region. Results were measured on an Apple M5 (10 cores) with Python 3.13.11,
TreeIG 0.1.11, scikit-learn 1.9.0, and SHAP 0.52.0, using each library's default
runtime threading. The seeded 20-feature data, both model definitions, checks,
and timing code are in
[`benchmarks/treeig_vs_treeshap.py`](https://github.com/LudgerHentschel/treeig/blob/main/benchmarks/treeig_vs_treeshap.py); rerun
with `python -m benchmarks.treeig_vs_treeshap`.

[XGBoost](https://xgboost.readthedocs.io/en/stable/prediction.html) and
[LightGBM](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html)
also provide tightly integrated native TreeSHAP routines. These can bring
TreeSHAP much closer to TreeIG's runtime; in a companion check, native XGBoost
TreeSHAP and TreeIG were essentially tied for 100 rows when the XGBoost
`DMatrix` was prepared before timing. That is a useful operational comparison,
but not quite like-for-like. Native `tree_path_dependent` TreeSHAP normally uses
training-sample counts stored along the tree paths as an implicit background
distribution, while the interventional TreeSHAP timings above use the same
explicit reference row as TreeIG. More generally, interventional TreeSHAP's
runtime scales with its explicit background sample, while TreeIG traverses and
averages the paths from its supplied baseline distribution.

This is a runtime comparison, not an equivalence claim: sharing a reference
aligns the output scale and reference prediction, but TreeIG and TreeSHAP still
compute the different quantities described above. Nor does the table claim that
TreeIG is always faster: TreeSHAP can have lower overhead for very small models
or batches. The point is that TreeIG remains competitive—and can be materially
faster—on the larger, slower attribution workloads where runtime matters most.
Exact timings depend on model shape, batch size, hardware, versions, and thread
settings.
