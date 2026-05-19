"""
Debug XGBoost routing and reconstruction for TreeIG.

Run from repository root:

    python scripts/debug_xgboost.py
"""

import json

import numpy as np
import xgboost as xgb

import treeig as tig
from treeig import TreeIG


def make_regression_data(n=180, p=5, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (
        1.2 * X[:, 0]
        - 0.8 * X[:, 1] ** 2
        + 0.5 * X[:, 2] * X[:, 3]
        + np.sin(X[:, 4])
    )
    return X, y


def finite_baseline(X):
    return X.mean(axis=0)


def debug_treeig_endpoint_prediction(model, X, target=None):
    ig = TreeIG(model, baseline=X.mean(axis=0), target=target)
    arrays = ig._resolve_arrays_for_target(target)

    cl = arrays["children_left"]
    cr = arrays["children_right"]
    ft = arrays["feature"]
    th = arrays["threshold"]
    va = arrays["value"]
    li = arrays["left_inclusive"]
    tw = arrays["tree_weight"]

    Xp = ig._prepare_X(X)
    y_treeig = np.zeros(Xp.shape[0], dtype=float)

    for i in range(Xp.shape[0]):
        for m in range(cl.shape[0]):
            y_treeig[i] += tw[m] * tig._predict_leaf(
                cl, cr, ft, th, va, li, m, Xp[i]
            )

    y_model = tig._model_predict(model, Xp, target)
    diff = y_treeig - y_model

    print("max abs endpoint diff:", np.max(np.abs(diff)))
    print("bad idx:", np.where(np.abs(diff) > 1e-8)[0][:20])
    print("diffs:", diff[np.abs(diff) > 1e-8][:20])

    return y_treeig, y_model, diff


def debug_xgboost_leaf_routing(model, X, target=None):
    ig = TreeIG(model, baseline=X.mean(axis=0), target=target)
    arrays = ig._resolve_arrays_for_target(target)

    cl = arrays["children_left"]
    cr = arrays["children_right"]
    ft = arrays["feature"]
    th = arrays["threshold"]
    li = arrays["left_inclusive"]

    Xp = ig._prepare_X(X)

    booster = model.get_booster() if hasattr(model, "get_booster") else model
    xgb_leaf = booster.predict(xgb.DMatrix(Xp), pred_leaf=True)

    if xgb_leaf.ndim == 1:
        xgb_leaf = xgb_leaf.reshape(-1, 1)

    def py_leaf(m, x):
        node = 0
        while cl[m, node] != cr[m, node]:
            j = ft[m, node]
            c = th[m, node]
            go_left = x[j] <= c if li[m, node] else x[j] < c
            node = cl[m, node] if go_left else cr[m, node]
        return node

    our_leaf = np.zeros_like(xgb_leaf, dtype=int)

    for i in range(Xp.shape[0]):
        for m in range(cl.shape[0]):
            our_leaf[i, m] = py_leaf(m, Xp[i])

    bad = np.where(our_leaf != xgb_leaf)

    print("number bad leaf routes:", len(bad[0]))
    for n in range(min(20, len(bad[0]))):
        i = bad[0][n]
        m = bad[1][n]
        print("obs", i, "tree", m, "ours", our_leaf[i, m], "xgb", int(xgb_leaf[i, m]))

    return our_leaf, xgb_leaf


def debug_xgb_reconstruction(model, X, x0, target=None):
    ig = TreeIG(model, baseline=x0, target=target)
    arrays = ig._resolve_arrays_for_target(target)

    cl = arrays["children_left"]
    cr = arrays["children_right"]
    ft = arrays["feature"]
    th = arrays["threshold"]
    va = arrays["value"]
    li = arrays["left_inclusive"]
    tw = arrays["tree_weight"]

    Xp = ig._prepare_X(X)
    b = ig._prepare_baseline(x0)

    def tree_sum(Z):
        out = np.zeros(Z.shape[0])
        for i in range(Z.shape[0]):
            for m in range(cl.shape[0]):
                out[i] += tw[m] * tig._predict_leaf(
                    cl, cr, ft, th, va, li, m, Z[i]
                )
        return out

    delta_tree = tree_sum(Xp) - tree_sum(b.reshape(1, -1))[0]

    y_model_X = tig._model_predict(model, Xp, target)
    y_model_0 = tig._model_predict(model, b.reshape(1, -1), target)[0]
    delta_model = y_model_X - y_model_0

    diff = delta_tree - delta_model

    print("max abs delta diff:", np.max(np.abs(diff)))
    print("bad idx:", np.where(np.abs(diff) > 1e-8)[0])
    print("diffs:", diff[np.abs(diff) > 1e-8])

    return delta_tree, delta_model, diff


def trace_xgb_tree(model, X, obs=0, tree=5):
    ig = TreeIG(model, baseline=X.mean(axis=0))
    arrays = ig._resolve_arrays_for_target(None)

    x = ig._prepare_X(X)[obs]

    cl = arrays["children_left"]
    cr = arrays["children_right"]
    ft = arrays["feature"]
    th = arrays["threshold"]
    li = arrays["left_inclusive"]

    print("TreeIG path")
    node = 0
    while cl[tree, node] != cr[tree, node]:
        j = ft[tree, node]
        c = th[tree, node]
        go_left = x[j] <= c if li[tree, node] else x[j] < c
        nxt = cl[tree, node] if go_left else cr[tree, node]
        print(
            "node", node,
            "feature", j,
            "x", x[j],
            "threshold", c,
            "left_inc", li[tree, node],
            "go_left", go_left,
            "next", nxt,
        )
        node = nxt

    print("TreeIG leaf:", node)

    booster = model.get_booster()
    pred_leaf = booster.predict(xgb.DMatrix(x.reshape(1, -1)), pred_leaf=True)
    print("XGBoost pred_leaf:", pred_leaf[0, tree])

    dump = json.loads(booster.get_dump(dump_format="json")[tree])
    print("XGBoost dump tree:")
    print(json.dumps(dump, indent=2)[:4000])


def main():
    print("USING:", tig.__file__)

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

    debug_treeig_endpoint_prediction(model, X_eval)
    debug_xgboost_leaf_routing(model, X_eval)

    ig = TreeIG(model, baseline=x0)
    arrays = ig._resolve_arrays_for_target(None)
    print("left_inclusive unique:", np.unique(arrays["left_inclusive"]))

    debug_xgb_reconstruction(model, X_eval, x0)


if __name__ == "__main__":
    main()