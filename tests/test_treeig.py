"""
pytest tests for TreeIG.

These tests focus on the central invariant:

    sum_j phi_j(x; x0) = F(x) - F(x0)

where F is the scalar output being attributed. For regressors this is the
prediction. For additive-score classifiers this is the raw class margin/logit.

Run from the repository root with:

    pytest tests/test_ig_tree.py

Assumptions
-----------
The package exposes TreeIG from either:

    from treeig import TreeIG

or, during local development:

    from ig_tree import TreeIG

Optional backends are skipped automatically if unavailable.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------
# Import TreeIG
# ---------------------------------------------------------------------

import numpy as np
import pytest

from treeig import TreeIG, supports

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def make_regression_data(n=180, p=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (
        1.2 * X[:, 0]
        - 0.8 * X[:, 1] ** 2
        + 0.5 * X[:, 2] * X[:, 3]
        + np.sin(X[:, 4])
    )
    return X, y


def make_binary_classification_data(n=220, p=5, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    score = (
        1.3 * X[:, 0]
        - 0.9 * X[:, 1]
        + 0.7 * X[:, 2] * X[:, 3]
        - 0.4 * X[:, 4] ** 2
    )
    y = (score > np.median(score)).astype(int)
    return X, y


def make_multiclass_classification_data(n=260, p=5, k=3, seed=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))

    scores = np.column_stack(
        [
            1.0 * X[:, 0] - 0.4 * X[:, 1] + 0.2 * X[:, 2] ** 2,
            -0.3 * X[:, 0] + 0.9 * X[:, 1] + 0.5 * X[:, 3],
            0.4 * X[:, 2] - 0.8 * X[:, 3] + 0.6 * X[:, 4],
        ]
    )
    y = np.argmax(scores, axis=1)
    return X, y


def finite_baseline(X):
    return X.mean(axis=0)


def test_public_support_detection_and_model_output():
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor

    X, y = make_regression_data(n=50, p=5, seed=120)
    tree = DecisionTreeRegressor(max_depth=3, random_state=120).fit(X, y)
    linear = LinearRegression().fit(X, y)

    assert supports(tree)
    assert not supports(linear)
    np.testing.assert_allclose(TreeIG(tree).model_output(X[:6]), tree.predict(X[:6]))


def assert_completeness(model, X, x0, target=None, atol=1e-8, rtol=1e-8):
    ig = TreeIG(model, baseline=x0, target=target).warmup(X[:3])
    phi, infos, summary = ig.explain(X, target=target)

    assert phi.shape == X.shape
    assert np.isfinite(phi).all()

    residuals = np.array([d["residual"] for d in infos], dtype=float)
    endpoint_delta = np.array([d["endpoint_delta"] for d in infos], dtype=float)

    np.testing.assert_allclose(
        phi.sum(axis=1),
        endpoint_delta,
        atol=atol,
        rtol=rtol,
    )
    scale = max(1.0, float(np.max(np.abs(endpoint_delta))))
    assert np.max(np.abs(residuals)) <= atol + rtol * scale
    assert summary["max_abs_residual"] <= atol + rtol * scale


def assert_attribute_matches_explain(model, X, x0, target=None, atol=1e-12):
    ig = TreeIG(model, baseline=x0, target=target).warmup(X[:3])
    phi_attr = ig.attribute(X, target=target)
    phi_explain, _, _ = ig.explain(X, target=target)

    np.testing.assert_allclose(phi_attr, phi_explain, atol=atol, rtol=0.0)


def test_weighted_baselines_match_explicit_average():
    from sklearn.tree import DecisionTreeRegressor

    X, y = make_regression_data(n=80, p=5, seed=123)
    model = DecisionTreeRegressor(max_depth=4, random_state=123).fit(X, y)
    baselines = X[:3]
    weights = np.array([1.0, 2.0, 7.0])
    X_eval = X[10:16]

    explainer = TreeIG(model)
    actual, by_baseline = explainer.attribute(
        X_eval,
        baseline=baselines,
        baseline_weights=weights,
        return_by_baseline=True,
    )
    expected_by_baseline = np.stack(
        [explainer.attribute(X_eval, baseline=b) for b in baselines]
    )
    expected = np.tensordot(weights / weights.sum(), expected_by_baseline, axes=1)

    np.testing.assert_allclose(by_baseline, expected_by_baseline)
    np.testing.assert_allclose(actual, expected)
    endpoint = model.predict(X_eval) - np.dot(
        weights / weights.sum(), model.predict(baselines)
    )
    np.testing.assert_allclose(actual.sum(axis=1), endpoint, atol=1e-10)


def test_weighted_baseline_loss_matches_explicit_average():
    from sklearn.ensemble import RandomForestRegressor

    X, y = make_regression_data(n=90, p=5, seed=124)
    model = RandomForestRegressor(
        n_estimators=8, max_depth=4, random_state=124
    ).fit(X, y)
    baselines = X[:4]
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    weights /= weights.sum()
    X_eval, y_eval = X[20:31], y[20:31]
    explainer = TreeIG(model)

    actual = explainer.loss_attribution(
        X_eval, y_eval, baseline=baselines, baseline_weights=weights
    )
    parts = [
        explainer.loss_attribution(X_eval, y_eval, baseline=b)
        for b in baselines
    ]
    expected_obs = sum(
        weights[i] * parts[i]["observation_values"]
        for i in range(len(parts))
    )

    np.testing.assert_allclose(actual["observation_values"], expected_obs)
    np.testing.assert_allclose(
        actual["baseline_loss"],
        sum(weights[i] * parts[i]["baseline_loss"] for i in range(len(parts))),
    )
    np.testing.assert_allclose(actual["values"].sum(), actual["total"])


def import_or_skip(module_name):
    return pytest.importorskip(module_name)


# ---------------------------------------------------------------------
# sklearn regression backends
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "model_factory",
    [
        pytest.param(
            lambda: __import__("sklearn.tree").tree.DecisionTreeRegressor(
                max_depth=4,
                random_state=0,
            ),
            id="DecisionTreeRegressor",
        ),
        pytest.param(
            lambda: __import__("sklearn.ensemble").ensemble.RandomForestRegressor(
                n_estimators=12,
                max_depth=4,
                random_state=0,
                n_jobs=1,
            ),
            id="RandomForestRegressor",
        ),
        pytest.param(
            lambda: __import__("sklearn.ensemble").ensemble.ExtraTreesRegressor(
                n_estimators=12,
                max_depth=4,
                random_state=0,
                n_jobs=1,
            ),
            id="ExtraTreesRegressor",
        ),
        pytest.param(
            lambda: __import__("sklearn.ensemble").ensemble.GradientBoostingRegressor(
                n_estimators=20,
                max_depth=3,
                learning_rate=0.07,
                random_state=0,
            ),
            id="GradientBoostingRegressor",
        ),
    ],
)
def test_sklearn_regression_completeness(model_factory):
    import_or_skip("sklearn")

    X, y = make_regression_data()
    x0 = finite_baseline(X)
    X_eval = X[:40]

    model = model_factory()
    model.fit(X, y)

    assert_completeness(model, X_eval, x0, target=None, atol=2e-8, rtol=2e-8)
    assert_attribute_matches_explain(model, X_eval, x0, target=None)


# ---------------------------------------------------------------------
# XGBoost regression and classification
# ---------------------------------------------------------------------

def test_xgboost_regressor_completeness():
    xgb = import_or_skip("xgboost")

    X, y = make_regression_data(seed=3)
    x0 = finite_baseline(X)
    X_eval = X[:40]

    model = xgb.XGBRegressor(
        n_estimators=18,
        max_depth=3,
        learning_rate=0.08,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="reg:squarederror",
        random_state=0,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(X, y)

    # XGBoost is float32 internally. Use looser tolerances. 
    assert_completeness(model, X_eval, x0, target=None, atol=1e-6, rtol=1e-6)
    assert_attribute_matches_explain(model, X_eval, x0, target=None, atol=1e-10)


def test_xgboost_binary_classifier_margin_completeness():
    xgb = import_or_skip("xgboost")

    X, y = make_binary_classification_data(seed=4)
    x0 = finite_baseline(X)
    X_eval = X[:40]

    model = xgb.XGBClassifier(
        n_estimators=18,
        max_depth=3,
        learning_rate=0.08,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=0,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(X, y)

    # XGBoost is float32 internally. Use looser tolerances. 
    assert_completeness(model, X_eval, x0, target=1, atol=1e-6, rtol=1e-6)
    assert_completeness(model, X_eval, x0, target=0, atol=1e-6, rtol=1e-6)
    assert_attribute_matches_explain(model, X_eval, x0, target=1, atol=1e-10)


def test_xgboost_multiclass_classifier_margin_completeness():
    xgb = import_or_skip("xgboost")

    X, y = make_multiclass_classification_data(seed=5)
    x0 = finite_baseline(X)
    X_eval = X[:40]

    model = xgb.XGBClassifier(
        n_estimators=9,
        max_depth=3,
        learning_rate=0.08,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=0,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(X, y)

    for target in range(3):
        # XGBoost is float32 internally. Use looser tolerances. 
        assert_completeness(model, X_eval, x0, target=target, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------
# LightGBM regression and classification
# ---------------------------------------------------------------------

def test_lightgbm_regressor_completeness():
    lgb = import_or_skip("lightgbm")

    X, y = make_regression_data(seed=6)
    x0 = finite_baseline(X)
    X_eval = X[:40]

    model = lgb.LGBMRegressor(
        n_estimators=18,
        max_depth=3,
        num_leaves=7,
        learning_rate=0.08,
        min_data_in_leaf=5,
        random_state=0,
        n_jobs=1,
        verbose=-1,
    )
    model.fit(X, y)

    assert_completeness(model, X_eval, x0, target=None, atol=2e-7, rtol=2e-7)
    assert_attribute_matches_explain(model, X_eval, x0, target=None, atol=1e-10)


def test_lightgbm_binary_classifier_margin_completeness():
    lgb = import_or_skip("lightgbm")

    X, y = make_binary_classification_data(seed=7)
    x0 = finite_baseline(X)
    X_eval = X[:40]

    model = lgb.LGBMClassifier(
        n_estimators=18,
        max_depth=3,
        num_leaves=7,
        learning_rate=0.08,
        min_data_in_leaf=5,
        random_state=0,
        n_jobs=1,
        verbose=-1,
    )
    model.fit(X, y)

    assert_completeness(model, X_eval, x0, target=1, atol=2e-7, rtol=2e-7)
    assert_completeness(model, X_eval, x0, target=0, atol=2e-7, rtol=2e-7)


def test_lightgbm_multiclass_classifier_margin_completeness():
    lgb = import_or_skip("lightgbm")

    X, y = make_multiclass_classification_data(seed=8)
    x0 = finite_baseline(X)
    X_eval = X[:40]

    model = lgb.LGBMClassifier(
        n_estimators=9,
        max_depth=3,
        num_leaves=7,
        learning_rate=0.08,
        min_data_in_leaf=5,
        random_state=0,
        n_jobs=1,
        verbose=-1,
    )
    model.fit(X, y)

    for target in range(3):
        assert_completeness(model, X_eval, x0, target=target, atol=2e-7, rtol=2e-7)


# ---------------------------------------------------------------------
# sklearn GradientBoostingClassifier
# ---------------------------------------------------------------------

def test_sklearn_gradient_boosting_binary_classifier_completeness():
    import_or_skip("sklearn")
    from sklearn.ensemble import GradientBoostingClassifier

    X, y = make_binary_classification_data(seed=9)
    x0 = finite_baseline(X)
    X_eval = X[:40]

    model = GradientBoostingClassifier(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.07,
        random_state=0,
    )
    model.fit(X, y)

    assert_completeness(model, X_eval, x0, target=1, atol=2e-8, rtol=2e-8)
    assert_completeness(model, X_eval, x0, target=0, atol=2e-8, rtol=2e-8)


def test_sklearn_gradient_boosting_multiclass_classifier_completeness():
    import_or_skip("sklearn")
    from sklearn.ensemble import GradientBoostingClassifier

    X, y = make_multiclass_classification_data(seed=10)
    x0 = finite_baseline(X)
    X_eval = X[:40]

    model = GradientBoostingClassifier(
        n_estimators=12,
        max_depth=3,
        learning_rate=0.07,
        random_state=0,
    )
    model.fit(X, y)

    for target in range(3):
        assert_completeness(model, X_eval, x0, target=target, atol=2e-8, rtol=2e-8)


# ---------------------------------------------------------------------
# API behavior
# ---------------------------------------------------------------------

def test_regression_target_none_and_zero_equivalent():
    import_or_skip("sklearn")
    from sklearn.ensemble import GradientBoostingRegressor

    X, y = make_regression_data(seed=11)
    x0 = finite_baseline(X)
    X_eval = X[:20]

    model = GradientBoostingRegressor(
        n_estimators=10,
        max_depth=2,
        random_state=0,
    )
    model.fit(X, y)

    phi_none = TreeIG(model, baseline=x0, target=None).attribute(X_eval)
    phi_zero = TreeIG(model, baseline=x0, target=0).attribute(X_eval)

    np.testing.assert_allclose(phi_none, phi_zero, atol=0.0, rtol=0.0)


def test_invalid_shapes_raise_errors():
    import_or_skip("sklearn")
    from sklearn.tree import DecisionTreeRegressor

    X, y = make_regression_data(seed=12)
    model = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y)

    ig = TreeIG(model, baseline=finite_baseline(X))

    with pytest.raises(ValueError):
        ig.attribute(X[0])  # must be 2-D

    with pytest.raises(ValueError):
        ig.attribute(np.column_stack([X, X[:, 0]]))  # wrong p

    bad_x0 = finite_baseline(X).copy()
    bad_x0[0] = np.nan

    with pytest.raises(ValueError):
        TreeIG(model, baseline=bad_x0)


def test_probability_averaging_classifiers_rejected():
    import_or_skip("sklearn")
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

    X, y = make_binary_classification_data(seed=13)

    classifiers = [
        DecisionTreeClassifier(max_depth=3, random_state=0),
        RandomForestClassifier(n_estimators=8, max_depth=3, random_state=0, n_jobs=1),
        ExtraTreesClassifier(n_estimators=8, max_depth=3, random_state=0, n_jobs=1),
    ]

    for model in classifiers:
        model.fit(X, y)
        with pytest.raises((TypeError, NotImplementedError, ValueError)):
            TreeIG(model, baseline=finite_baseline(X), target=1)


# ---------------------------------------------------------------------
# Boundary and edge-case tests for sklearn trees
# ---------------------------------------------------------------------

def test_one_node_tree_has_zero_attributions():
    import_or_skip("sklearn")
    from sklearn.tree import DecisionTreeRegressor

    X, y = make_regression_data(n=80, p=5, seed=14)
    model = DecisionTreeRegressor(
        max_depth=1,
        min_samples_split=X.shape[0] + 1,
        random_state=0,
    )
    model.fit(X, y)

    x0 = finite_baseline(X)
    X_eval = X[:20]

    ig = TreeIG(model, baseline=x0)
    phi, infos, summary = ig.explain(X_eval)

    np.testing.assert_allclose(phi, 0.0, atol=0.0, rtol=0.0)
    assert summary["max_abs_residual"] == 0.0


def test_zero_movement_observation_has_zero_attributions():
    import_or_skip("sklearn")
    from sklearn.tree import DecisionTreeRegressor

    X, y = make_regression_data(n=100, p=5, seed=15)
    model = DecisionTreeRegressor(max_depth=4, random_state=0)
    model.fit(X, y)

    x0 = X[0].copy()
    X_eval = x0.reshape(1, -1)

    ig = TreeIG(model, baseline=x0)
    phi, infos, summary = ig.explain(X_eval)

    np.testing.assert_allclose(phi, 0.0, atol=1e-12, rtol=0.0)
    assert summary["max_abs_residual"] <= 1e-12


def test_threshold_at_baseline_or_endpoint_completeness():
    """
    Force endpoint-threshold cases by using a tiny 1-D tree.
    """
    import_or_skip("sklearn")
    from sklearn.tree import DecisionTreeRegressor

    X = np.array([[0.0], [0.25], [0.75], [1.0]], dtype=float)
    y = np.array([0.0, 0.0, 2.0, 2.0], dtype=float)

    model = DecisionTreeRegressor(max_depth=1, random_state=0)
    model.fit(X, y)

    threshold = float(model.tree_.threshold[0])

    x0 = np.array([threshold], dtype=float)
    X_eval = np.array([[1.0], [0.0]], dtype=float)
    assert_completeness(model, X_eval, x0, target=None, atol=1e-10, rtol=1e-10)

    x0 = np.array([0.0], dtype=float)
    X_eval = np.array([[threshold]], dtype=float)
    assert_completeness(model, X_eval, x0, target=None, atol=1e-10, rtol=1e-10)


def test_simultaneous_crossing_completeness_first_policy():
    """
    Simultaneous crossings are allocated by the current first-policy, but
    completeness should still hold.
    """
    import_or_skip("sklearn")
    from sklearn.tree import DecisionTreeRegressor

    X = np.array(
        [
            [-1.0, -1.0],
            [-1.0,  1.0],
            [ 1.0, -1.0],
            [ 1.0,  1.0],
        ],
        dtype=float,
    )
    y = np.array([0.0, 1.0, 2.0, 4.0], dtype=float)

    model = DecisionTreeRegressor(max_depth=2, random_state=0)
    model.fit(X, y)

    x0 = np.array([-1.0, -1.0], dtype=float)
    X_eval = np.array([[1.0, 1.0]], dtype=float)

    assert_completeness(model, X_eval, x0, target=None, atol=1e-10, rtol=1e-10)


def test_sklearn_gradient_boosting_classifier_subclass_uses_decision_function():
    import_or_skip("sklearn")
    from sklearn.ensemble import GradientBoostingClassifier

    class MyGradientBoostingClassifier(GradientBoostingClassifier):
        pass

    X, y = make_binary_classification_data(seed=21)
    x0 = finite_baseline(X)
    X_eval = X[:20]

    model = MyGradientBoostingClassifier(
        n_estimators=10,
        max_depth=2,
        learning_rate=0.07,
        random_state=0,
    )
    model.fit(X, y)

    ig = TreeIG(model, baseline=x0, target=1).warmup(X_eval[:3])
    phi, infos, _ = ig.explain(X_eval, target=1)
    endpoint_delta = np.array([d["endpoint_delta"] for d in infos], dtype=float)

    expected = model.decision_function(X_eval) - model.decision_function(x0.reshape(1, -1))[0]
    np.testing.assert_allclose(endpoint_delta, expected, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(phi.sum(axis=1), expected, atol=2e-8, rtol=2e-8)


def test_backend_round_input_flags_are_backend_specific():
    import_or_skip("sklearn")
    from sklearn.ensemble import GradientBoostingRegressor

    X, y = make_regression_data(seed=22)
    model = GradientBoostingRegressor(n_estimators=4, max_depth=2, random_state=0).fit(X, y)
    ig = TreeIG(model, baseline=finite_baseline(X))
    assert not np.asarray(ig._arrays["round_input"], dtype=bool).any()

    xgb = pytest.importorskip("xgboost")
    model_xgb = xgb.XGBRegressor(
        n_estimators=4,
        max_depth=2,
        objective="reg:squarederror",
        random_state=0,
        n_jobs=1,
        verbosity=0,
    ).fit(X, y)
    ig_xgb = TreeIG(model_xgb, baseline=finite_baseline(X))
    assert np.asarray(ig_xgb._arrays["round_input"], dtype=bool).any()


def test_batch_size_matches_full_batch():
    import_or_skip("sklearn")
    from sklearn.ensemble import RandomForestRegressor

    X, y = make_regression_data(seed=23)
    x0 = finite_baseline(X)
    X_eval = X[:35]

    model = RandomForestRegressor(
        n_estimators=6,
        max_depth=3,
        random_state=0,
        n_jobs=1,
    ).fit(X, y)

    ig = TreeIG(model, baseline=x0).warmup(X_eval[:3])
    phi_full, infos_full, summary_full = ig.explain(X_eval)
    phi_chunk, infos_chunk, summary_chunk = ig.explain(X_eval, batch_size=7)

    np.testing.assert_allclose(phi_chunk, phi_full, atol=0.0, rtol=0.0)
    assert [d["n_events"] for d in infos_chunk] == [d["n_events"] for d in infos_full]
    assert summary_chunk == summary_full

    phi_attr_full = ig.attribute(X_eval)
    phi_attr_chunk = ig.attribute(X_eval, batch_size=7)
    np.testing.assert_allclose(phi_attr_chunk, phi_attr_full, atol=0.0, rtol=0.0)

    with pytest.raises(ValueError):
        ig.attribute(X_eval, batch_size=0)


def test_loss_attribution_single_observation_standard_errors_are_nan_without_warning(recwarn):
    import_or_skip("sklearn")
    from sklearn.tree import DecisionTreeRegressor

    X, y = make_regression_data(seed=25)
    baseline = finite_baseline(X)

    model = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y)
    explainer = TreeIG(model, baseline=baseline)

    result = explainer.loss_attribution(X[:1], y[:1])

    assert len(recwarn) == 0
    assert np.isnan(result["standard_errors"]).all()


def test_trace_reconstructs_attribute():
    import_or_skip("sklearn")
    from sklearn.ensemble import RandomForestRegressor

    X, y = make_regression_data(seed=24)
    baseline = finite_baseline(X)
    X_eval = X[:25]

    model = RandomForestRegressor(
        n_estimators=6,
        max_depth=3,
        random_state=0,
        n_jobs=1,
    )
    model.fit(X, y)

    explainer = TreeIG(model, baseline=baseline).warmup(X_eval[:3])

    phis = explainer.attribute(X_eval)
    trace = explainer.trace(X_eval)

    reconstructed = np.zeros_like(phis)

    for i in range(X_eval.shape[0]):
        n = trace["counts"][i]
        for k in range(n):
            j = trace["features"][i, k]
            reconstructed[i, j] += trace["jumps"][i, k]

    np.testing.assert_allclose(reconstructed, phis, atol=1e-10)    


if __name__ == "__main__":

    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
