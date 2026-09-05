# Getting started

## Installation

```bash
pip install "treeig[sklearn]"
```

TreeIG requires Python 3.9 or later, NumPy, and Numba. Install the model library
you use: extras `sklearn`, `xgboost`, `lightgbm`, and `catboost` are available.
The `shap` extra adds plotting integration. `all` installs the model and plotting
extras; it does not install CUDA. CatBoost uses the numerical fallback.

## A complete regression example

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from treeig import TreeIG

rng = np.random.default_rng(42)
X = rng.normal(size=(400, 4))
y = 2 * X[:, 0] + X[:, 1] ** 2 - X[:, 2]
X_train, X_eval, y_train, y_eval = train_test_split(X, y, random_state=42)
model = GradientBoostingRegressor(n_estimators=40, max_depth=3,
                                  random_state=42).fit(X_train, y_train)

# A representative row is a simple reference for this example.
ig = TreeIG(model, baseline=X_train[0])
result = ig.explain(X_eval)
print(result.values.shape)  # (100, 4)
np.testing.assert_allclose(
    result.values.sum(axis=1),
    model.predict(X_eval) - model.predict(X_train[:1])[0],
    atol=1e-8,
)
```

Each row in `values` corresponds to an observation; each column corresponds to
an input feature. Positive values increase the prediction relative to the
baseline, and negative values decrease it. The sum reconstructs the prediction
change up to floating-point error. The first call includes Numba compilation.

For array-only output, use `ig.attribute(X_eval)`. For substantive work, choose a
baseline that expresses the intended comparison; see [baselines](baselines.md).
For classification, first check [model support](models.md) and the
[classification target conventions](explanations.md#classification-targets).
