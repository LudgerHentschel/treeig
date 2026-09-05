# Supported models

TreeIG currently supports tree models with finite numeric feature inputs.

## Regression

- `sklearn.tree.DecisionTreeRegressor`
- `sklearn.ensemble.RandomForestRegressor`
- `sklearn.ensemble.ExtraTreesRegressor`
- `sklearn.ensemble.GradientBoostingRegressor`
- `xgboost.XGBRegressor`
- `xgboost.Booster`
- `lightgbm.LGBMRegressor`
- `lightgbm.Booster`

## Classification (raw margins/logits only)

- `sklearn.ensemble.GradientBoostingClassifier`
- `xgboost.XGBClassifier`
- `lightgbm.LGBMClassifier`

For classification models, exact structural TreeIG attributes raw margins or
logits. Its current exact engine does not transform ensemble probabilities.

TreeIG computes exact path decompositions directly from the fitted tree
structure. Since tree representations differ substantially across
implementations, each model family requires customized parsing and routing
logic.

## Exact support not currently available

The exact TreeIG parser does not currently support:

- CatBoost;
- categorical splits;
- missing-value routing (use feature augmentation for missingness);
- structurally exact transformed-probability attribution;
- probability-averaging or vote-share classifiers such as
  `DecisionTreeClassifier`, `RandomForestClassifier`, and
  `ExtraTreesClassifier` (because they produce probabilities, not scores).

Many of these can still be attributed with the model-agnostic
[TreeIGNumeric](numeric.md), described below.
