"""Engine-level tests that need no model library: they pass synthetic
piecewise-constant functions directly to NumericEngine."""

import numpy as np
import pytest

from treeig.numeric import NumericEngine, TreeIGNumeric


def step(v, thr, lo, hi):
    return np.where(v >= thr, hi, lo)


def test_single_split():
    # f depends on feature 0 only: jumps 0 -> 10 at 0.5
    f = lambda P: step(P[:, 0], 0.5, 0.0, 10.0)
    eng = NumericEngine(f, n_features=2)
    phi, infos = eng.attribute(np.array([0.0, 0.0]), np.array([[1.0, 1.0]]))
    assert np.allclose(phi[0], [10.0, 0.0]), phi
    assert abs(infos[0]["abs_residual"]) < 1e-12
    assert infos[0]["n_coincident_events"] == 0
    print("single_split:", phi[0], "events", infos[0]["n_events"])


def test_additive_two_features():
    # independent splits at DISTINCT path times (0.3, 0.6) -> clean separation
    f = lambda P: step(P[:, 0], 0.3, 0.0, 3.0) + step(P[:, 1], 0.6, 0.0, 5.0)
    eng = NumericEngine(f, n_features=2)
    phi, infos = eng.attribute(np.array([0.0, 0.0]), np.array([[1.0, 1.0]]))
    assert np.allclose(phi[0], [3.0, 5.0]), phi
    assert abs(infos[0]["abs_residual"]) < 1e-12
    assert infos[0]["n_coincident_events"] == 0  # distinct t -> no coincidence
    print("additive (separated):", phi[0])


def test_additive_coincident_independent():
    # independent splits at the SAME path time (both 0.5): coincident but
    # non-interacting -> sweep still recovers the marginal jumps [3, 5]
    f = lambda P: step(P[:, 0], 0.5, 0.0, 3.0) + step(P[:, 1], 0.5, 0.0, 5.0)
    eng = NumericEngine(f, n_features=2)
    phi, infos = eng.attribute(np.array([0.0, 0.0]), np.array([[1.0, 1.0]]))
    assert np.allclose(phi[0], [3.0, 5.0]), phi
    assert infos[0]["n_coincident_events"] >= 1
    print("additive (coincident, independent):", phi[0])


def test_conjunction_separated():
    # f = 7 iff both >= threshold; thresholds at distinct path times
    def f(P):
        return np.where((P[:, 0] >= 0.3) & (P[:, 1] >= 0.6), 7.0, 0.0)

    eng = NumericEngine(f, n_features=2)
    phi, infos = eng.attribute(np.array([0.0, 0.0]), np.array([[1.0, 1.0]]))
    # the prediction flips only when the *second* (x1) threshold is crossed
    assert np.allclose(phi[0], [0.0, 7.0]), phi
    assert abs(infos[0]["abs_residual"]) < 1e-12
    print("conjunction (separated):", phi[0])


def test_conjunction_coincident():
    # both thresholds at 0.5 -> exactly coincident crossing -> sweep fallback
    def f(P):
        return np.where((P[:, 0] >= 0.5) & (P[:, 1] >= 0.5), 7.0, 0.0)

    eng = NumericEngine(f, n_features=2)
    phi, infos = eng.attribute(np.array([0.0, 0.0]), np.array([[1.0, 1.0]]))
    assert abs(phi[0].sum() - 7.0) < 1e-12, phi  # completeness preserved
    assert infos[0]["n_coincident_events"] >= 1
    print("conjunction (coincident):", phi[0],
          "coincident", infos[0]["n_coincident_events"])


def test_non_moving_feature_zero():
    # feature 1 does not move (x == x0); must receive zero
    f = lambda P: step(P[:, 0], 0.5, 0.0, 4.0) + step(P[:, 1], 0.5, 0.0, 9.0)
    eng = NumericEngine(f, n_features=2)
    phi, _ = eng.attribute(np.array([0.0, 1.0]), np.array([[1.0, 1.0]]))
    assert phi[0, 1] == 0.0, phi
    assert np.allclose(phi[0, 0], 4.0), phi
    print("non-moving:", phi[0])


def test_completeness_random_forest_like():
    rng = np.random.default_rng(0)
    p = 6
    thr = rng.uniform(0.2, 0.8, size=(40, p))
    feat = rng.integers(0, p, size=40)
    w = rng.normal(size=40)

    def f(P):  # sum of 40 random single-feature stumps
        out = np.zeros(P.shape[0])
        for k in range(40):
            out += w[k] * (P[:, feat[k]] >= thr[k, feat[k]])
        return out

    eng = NumericEngine(f, n_features=p)
    x0 = np.zeros(p)
    X = rng.uniform(0, 1, size=(10, p))
    phi, infos = eng.attribute(x0, X)
    delta = f(X) - f(x0[None, :])
    assert np.allclose(phi.sum(axis=1), delta, atol=1e-10)
    print("forest-like max abs residual:",
          max(d["abs_residual"] for d in infos))


def test_public_numeric_model_output_matches_attributed_quantity():
    class StepRegressor:
        def predict(self, X):
            return np.where(X[:, 0] >= 0.4, 3.0, -1.0)

    model = StepRegressor()
    explainer = TreeIGNumeric(model, baseline=np.zeros(2), grid_size=32)
    X = np.array([[0.2, 1.0], [0.8, -0.5]])

    output = explainer.model_output(X)
    phi = explainer.attribute(X)

    np.testing.assert_allclose(output, model.predict(X))
    np.testing.assert_allclose(
        phi.sum(axis=1) + explainer.model_output(np.zeros((1, 2)))[0],
        output,
    )


def test_public_numeric_model_output_validates_inputs():
    class StepRegressor:
        def predict(self, X):
            return np.zeros(X.shape[0])

    explainer = TreeIGNumeric(StepRegressor(), baseline=np.zeros(2))

    with pytest.raises(ValueError, match="shape"):
        explainer.model_output([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="at least one"):
        explainer.model_output(np.empty((0, 2)))
    with pytest.raises(ValueError, match="finite"):
        explainer.model_output([[np.nan, 0.0]])


def _floored_scores(model, X, floor):
    proba = np.maximum(model.predict_proba(X), floor)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return np.log(proba)


def test_binary_probability_classifier_can_explain_log_odds():
    from sklearn.ensemble import RandomForestClassifier

    X = np.array(
        [[0.0, 0.0], [0.2, 1.0], [0.8, 0.0], [1.0, 1.0], [0.5, 0.4]]
    )
    y = np.array([0, 0, 1, 1, 1])
    model = RandomForestClassifier(
        n_estimators=7, max_depth=2, random_state=0
    ).fit(X, y)
    floor = 1e-6

    with pytest.warns(RuntimeWarning, match="deriving a score"):
        explainer = TreeIGNumeric(
            model,
            baseline=X[0],
            probability_to_score=True,
            probability_floor=floor,
            grid_size=128,
        )
    explained = X[2:]
    log_proba = _floored_scores(model, explained, floor)
    expected = log_proba[:, 1] - log_proba[:, 0]
    base_log_proba = _floored_scores(model, X[[0]], floor)
    expected_base = base_log_proba[0, 1] - base_log_proba[0, 0]

    np.testing.assert_allclose(explainer.model_output(explained), expected)
    np.testing.assert_allclose(
        explainer.attribute(explained).sum(axis=1) + expected_base,
        expected,
        atol=1e-12,
    )


def test_multiclass_probability_classifier_uses_centered_log_scores():
    from sklearn.datasets import make_classification
    from sklearn.ensemble import ExtraTreesClassifier

    X, y = make_classification(
        n_samples=70,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        random_state=2,
    )
    model = ExtraTreesClassifier(
        n_estimators=9, max_depth=3, random_state=2
    ).fit(X, y)
    floor = 1e-5

    with pytest.warns(RuntimeWarning, match="deriving a score"):
        explainer = TreeIGNumeric(
            model,
            baseline=X[0],
            target=2,
            probability_to_score=True,
            probability_floor=floor,
            grid_size=256,
        )
    log_proba = _floored_scores(model, X[10:14], floor)
    expected = log_proba[:, 2] - log_proba.mean(axis=1)

    np.testing.assert_allclose(explainer.model_output(X[10:14]), expected)
    np.testing.assert_allclose(
        explainer.attribute(X[10:14]).sum(axis=1)
        + explainer.model_output(X[[0]])[0],
        expected,
        atol=1e-12,
    )


def test_probability_score_requires_explicit_floor_at_zero():
    from sklearn.tree import DecisionTreeClassifier

    X = np.array([[0.0], [1.0]])
    model = DecisionTreeClassifier(random_state=0).fit(X, [0, 1])
    with pytest.warns(RuntimeWarning, match="deriving a score"):
        explainer = TreeIGNumeric(
            model,
            baseline=X[0],
            probability_to_score=True,
        )

    with pytest.raises(ValueError, match="probability_floor"):
        explainer.model_output(X)


@pytest.mark.parametrize("floor", [0.0, 1.0, -0.1, np.inf])
def test_probability_floor_is_validated(floor):
    class StepRegressor:
        def predict(self, X):
            return np.zeros(X.shape[0])

    with pytest.raises(ValueError, match="probability_floor"):
        TreeIGNumeric(
            StepRegressor(), baseline=np.zeros(1), probability_floor=floor
        )


def test_probability_floor_requires_score_transformation():
    class StepRegressor:
        def predict(self, X):
            return np.zeros(X.shape[0])

    with pytest.raises(ValueError, match="probability_to_score=True"):
        TreeIGNumeric(
            StepRegressor(), baseline=np.zeros(1), probability_floor=1e-6
        )


if __name__ == "__main__":
    test_single_split()
    test_additive_two_features()
    test_additive_coincident_independent()
    test_conjunction_separated()
    test_conjunction_coincident()
    test_non_moving_feature_zero()
    test_completeness_random_forest_like()
    print("\nall engine tests passed")
