import builtins

import numpy as np
import pytest

from treeig import Explanation, TreeIG, compute
from treeig.numeric import TreeIGNumeric


def test_exact_explain_returns_unified_explanation_contract():
    pytest.importorskip("sklearn")
    from sklearn.tree import DecisionTreeRegressor

    X = np.array([[0.0, 0.0], [0.2, 1.0], [0.8, 0.0], [1.0, 1.0]])
    y = np.array([-1.0, 0.0, 2.0, 3.0])
    model = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, y)
    baseline = X[0]
    data = X[1:]

    result = TreeIG(model, baseline=baseline).explain(data)

    assert isinstance(result, Explanation)
    assert result.values.shape == result.data.shape == data.shape
    assert result.base_values.shape == (len(data),)
    np.testing.assert_allclose(
        result.base_values,
        np.repeat(model.predict(baseline[None]), len(data)),
    )
    np.testing.assert_allclose(
        result.base_values + result.values.sum(axis=1), model.predict(data)
    )
    assert result.max_abs_completeness_error < 1e-12

    functional_result = compute(model, baseline, data)
    assert isinstance(functional_result, Explanation)
    np.testing.assert_allclose(functional_result.values, result.values)


def test_numeric_explain_returns_same_contract():
    class StepRegressor:
        def predict(self, X):
            return np.where(X[:, 0] >= 0.5, 3.0, -1.0)

    data = np.array([[0.25, 1.0], [0.75, -1.0]])
    result = TreeIGNumeric(
        StepRegressor(), baseline=np.zeros(2), grid_size=32
    ).explain(data)

    assert isinstance(result, Explanation)
    assert result.values.shape == result.data.shape == data.shape
    np.testing.assert_allclose(
        result.base_values + result.values.sum(axis=1),
        StepRegressor().predict(data),
    )


def test_explain_preserves_dataframe_feature_names():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    from sklearn.tree import DecisionTreeRegressor

    data = pd.DataFrame(
        [[0.0, 0.0], [0.2, 1.0], [0.8, 0.0], [1.0, 1.0]],
        columns=["height", "weight"],
    )
    model = DecisionTreeRegressor(max_depth=2, random_state=0).fit(
        data, [-1.0, 0.0, 2.0, 3.0]
    )

    result = TreeIG(model, baseline=data.iloc[0]).explain(data.iloc[1:])

    assert result.feature_names == ["height", "weight"]
    np.testing.assert_allclose(result.data, data.iloc[1:].to_numpy())


def test_to_shap_returns_compatible_explanation():
    shap = pytest.importorskip("shap")
    result = Explanation(
        values=np.array([[0.2, -0.1], [0.3, 0.4]]),
        base_values=np.array([1.0, 1.0]),
        data=np.array([[2.0, 3.0], [4.0, 5.0]]),
        feature_names=["a", "b"],
        completeness_error=np.zeros(2),
    ).to_shap()

    assert isinstance(result, shap.Explanation)
    np.testing.assert_allclose(result.values, [[0.2, -0.1], [0.3, 0.4]])
    np.testing.assert_allclose(result.base_values, [1.0, 1.0])
    assert list(result.feature_names) == ["a", "b"]


def test_shap_plots_accept_treeig_explanation():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    shap = pytest.importorskip("shap")
    result = Explanation(
        values=np.array([[0.2, -0.1], [0.3, 0.4]]),
        base_values=np.array([1.0, 1.0]),
        data=np.array([[2.0, 3.0], [4.0, 5.0]]),
        feature_names=["a", "b"],
        completeness_error=np.zeros(2),
    ).to_shap()

    shap.plots.waterfall(result[0], show=False)
    shap.plots.beeswarm(result, show=False)

    import matplotlib.pyplot as plt
    plt.close("all")


def test_to_shap_has_actionable_optional_dependency_error(monkeypatch):
    explanation = Explanation(
        values=np.array([[0.2, -0.1]]),
        base_values=np.array([1.0]),
        data=np.array([[2.0, 3.0]]),
    )
    original_import = builtins.__import__

    def without_shap(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_shap)
    with pytest.raises(ImportError, match=r"treeig\[shap\]"):
        explanation.to_shap()
