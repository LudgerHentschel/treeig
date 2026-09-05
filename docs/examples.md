# Worked examples

These examples use synthetic data so they can run without downloading datasets.
Install the relevant extras with `pip install "treeig[xgboost,lightgbm]"`.
The single training row is a simple illustrative baseline; use the
[baseline guide](baselines.md) to choose a substantive reference distribution.

## XGBoost regression

```python
import numpy as np
import xgboost as xgb
from treeig import TreeIG

rng = np.random.default_rng(7)
X = rng.normal(size=(300, 4))
y = X[:, 0] ** 2 - X[:, 1] + 0.5 * X[:, 2]
model = xgb.XGBRegressor(n_estimators=30, max_depth=3, random_state=7)
model.fit(X[:200], y[:200])
ig = TreeIG(model, baseline=X[0]).warmup(X[200:203])
result = ig.explain(X[200:])
np.testing.assert_allclose(result.completeness_error, 0, atol=1e-5)
```

XGBoost's own prediction accumulation can differ slightly from TreeIG's packed
float64 tree sum, so use a tolerance appropriate to the output scale. The
feature attributions explain prediction differences, not absolute predictions.

## LightGBM multiclass margins

```python
import numpy as np
import lightgbm as lgb
from treeig import TreeIG

rng = np.random.default_rng(8)
X = rng.normal(size=(300, 4))
y = np.argmax(X[:, :3], axis=1)
model = lgb.LGBMClassifier(n_estimators=20, num_leaves=7,
                           random_state=8, verbosity=-1)
model.fit(X[:200], y[:200])
ig = TreeIG(model, baseline=X[0], target=2)
result = ig.explain(X[200:])
np.testing.assert_allclose(result.completeness_error, 0, atol=1e-8)
```

`target=2` selects the third class margin. It does not select a class label or
request a probability explanation. Use `model.classes_` to map positions to
labels. The attributions need not sum to a probability change.
