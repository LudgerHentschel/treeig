from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from numba import njit, prange
except ImportError as exc:
    raise ImportError("TreeIG requires numba.") from exc


@njit
def _go_left(xj, c, left_inclusive, round_input):
    """Return True if traversal should take the left child."""
    if round_input:
        xj = np.float64(np.float32(xj))
    if left_inclusive:
        return xj <= c
    return xj < c


@njit
def _predict_leaf(cl, cr, feat, thresh, val, left_inc, round_input, tree_idx, x):
    """Traverse tree_idx for input x and return the leaf value."""
    node = 0

    while cl[tree_idx, node] != cr[tree_idx, node]:
        j = feat[tree_idx, node]
        c = thresh[tree_idx, node]
        if _go_left(x[j], c, left_inc[tree_idx, node], round_input[tree_idx, node]):
            node = cl[tree_idx, node]
        else:
            node = cr[tree_idx, node]

    return val[tree_idx, node]


@njit
def _find_divergent_feature(
    cl, cr, feat, thresh, left_inc, round_input, tree_idx, x_before, x_after
):
    """
    Walk x_before and x_after down tree_idx simultaneously.

    Returns the feature index at the first split where the two paths diverge,
    or -1 if both reach the same leaf.
    """
    node_b = 0
    node_a = 0
    parent = -1

    while True:
        if node_b != node_a:
            return feat[tree_idx, parent] if parent >= 0 else -1

        lc = cl[tree_idx, node_b]
        rc = cr[tree_idx, node_b]

        if lc == rc:
            return -1

        parent = node_b
        j = feat[tree_idx, node_b]
        c = thresh[tree_idx, node_b]
        inc = left_inc[tree_idx, node_b]
        rnd = round_input[tree_idx, node_b]

        node_b = lc if _go_left(x_before[j], c, inc, rnd) else rc
        node_a = lc if _go_left(x_after[j], c, inc, rnd) else rc


@njit
def _baseline_leaf_values(cl, cr, feat, thresh, val, left_inc, round_input, x0):
    """Traverse every tree once for the shared baseline x0."""
    n_trees = cl.shape[0]
    y0 = np.empty(n_trees, dtype=np.float64)

    for m in range(n_trees):
        y0[m] = _predict_leaf(cl, cr, feat, thresh, val, left_inc, round_input, m, x0)

    return y0


@njit
def _dfs_intervals(
    cl,
    cr,
    feat,
    thresh,
    val,
    left_inc,
    round_input,
    tree_idx,
    x0,
    dx,
    time_tol,
    stk_node,
    stk_tl,
    stk_tr,
    stk_ef,
    seg_tl,
    seg_val,
    seg_ef,
):
    """Partition [0, 1] into contiguous leaf segments."""
    stk_node[0] = 0
    stk_tl[0] = 0.0
    stk_tr[0] = 1.0
    stk_ef[0] = -1

    n_stk = 1
    n_seg = 0

    while n_stk > 0:
        n_stk -= 1

        node = stk_node[n_stk]
        tl = stk_tl[n_stk]
        tr = stk_tr[n_stk]
        ef = stk_ef[n_stk]

        lc = cl[tree_idx, node]
        rc = cr[tree_idx, node]

        if lc == rc:
            seg_tl[n_seg] = tl
            seg_val[n_seg] = val[tree_idx, node]
            seg_ef[n_seg] = ef
            n_seg += 1
            continue

        j = feat[tree_idx, node]
        c = thresh[tree_idx, node]
        d = dx[j]
        inc = left_inc[tree_idx, node]
        rnd = round_input[tree_idx, node]

        if abs(d) <= time_tol * (abs(x0[j]) + abs(x0[j] + d) + 1.0):
            child = lc if _go_left(x0[j], c, inc, rnd) else rc
            stk_node[n_stk] = child
            stk_tl[n_stk] = tl
            stk_tr[n_stk] = tr
            stk_ef[n_stk] = ef
            n_stk += 1
            continue

        t_cross = (c - x0[j]) / d

        if tl + time_tol < t_cross < tr - time_tol and time_tol < t_cross < 1.0 - time_tol:
            cb = lc if d > 0.0 else rc
            ca = rc if d > 0.0 else lc

            stk_node[n_stk] = ca
            stk_tl[n_stk] = t_cross
            stk_tr[n_stk] = tr
            stk_ef[n_stk] = j
            n_stk += 1

            stk_node[n_stk] = cb
            stk_tl[n_stk] = tl
            stk_tr[n_stk] = t_cross
            stk_ef[n_stk] = ef
            n_stk += 1

        else:
            t_mid = 0.5 * (tl + tr)
            x_mid = x0[j] + t_mid * d
            child = lc if _go_left(x_mid, c, inc, rnd) else rc

            stk_node[n_stk] = child
            stk_tl[n_stk] = tl
            stk_tr[n_stk] = tr
            stk_ef[n_stk] = ef
            n_stk += 1

    for a in range(1, n_seg):
        kt = seg_tl[a]
        kv = seg_val[a]
        kf = seg_ef[a]

        b = a - 1
        while b >= 0 and seg_tl[b] > kt:
            seg_tl[b + 1] = seg_tl[b]
            seg_val[b + 1] = seg_val[b]
            seg_ef[b + 1] = seg_ef[b]
            b -= 1

        seg_tl[b + 1] = kt
        seg_val[b + 1] = kv
        seg_ef[b + 1] = kf

    return n_seg


@njit
def _attribute_one_tree(
    cl,
    cr,
    feat,
    thresh,
    val,
    left_inc,
    round_input,
    tree_idx,
    x0,
    x1,
    dx,
    y0_tree,
    tree_weight,
    time_tol,
    phi,
    stk_node,
    stk_tl,
    stk_tr,
    stk_ef,
    seg_tl,
    seg_val,
    seg_ef,
    probe,
):
    p = x0.shape[0]
    n_events = 0

    n_seg = _dfs_intervals(
        cl,
        cr,
        feat,
        thresh,
        val,
        left_inc,
        round_input,
        tree_idx,
        x0,
        dx,
        time_tol,
        stk_node,
        stk_tl,
        stk_tr,
        stk_ef,
        seg_tl,
        seg_val,
        seg_ef,
    )

    for k in range(1, n_seg):
        j = seg_ef[k]
        jump = seg_val[k] - seg_val[k - 1]

        if j >= 0 and jump != 0.0:
            phi[j] += tree_weight * jump
            n_events += 1

    jump0 = (seg_val[0] - y0_tree) if n_seg > 0 else 0.0

    if jump0 != 0.0:
        if n_seg > 1:
            t_probe = time_tol + 0.5 * (seg_tl[1] - time_tol)
        else:
            t_probe = 0.5

        for jj in range(p):
            probe[jj] = x0[jj] + t_probe * dx[jj]

        j0 = _find_divergent_feature(
            cl, cr, feat, thresh, left_inc, round_input, tree_idx, x0, probe
        )

        if j0 >= 0:
            phi[j0] += tree_weight * jump0
            n_events += 1

    y1 = _predict_leaf(cl, cr, feat, thresh, val, left_inc, round_input, tree_idx, x1)
    jump1 = y1 - (seg_val[n_seg - 1] if n_seg > 0 else y1)

    if jump1 != 0.0:
        if n_seg > 0:
            t_probe = 0.5 * (seg_tl[n_seg - 1] + (1.0 - time_tol))
        else:
            t_probe = 0.5

        for jj in range(p):
            probe[jj] = x0[jj] + t_probe * dx[jj]

        j1 = _find_divergent_feature(
            cl, cr, feat, thresh, left_inc, round_input, tree_idx, probe, x1
        )

        if j1 >= 0:
            phi[j1] += tree_weight * jump1
            n_events += 1

    return n_events


@njit(parallel=True)
def _compute_batch(
    cl,
    cr,
    feat,
    thresh,
    val,
    left_inc,
    round_input,
    x0,
    X,
    y0_per_tree,
    tree_weight,
    time_tol,
):
    """Compute TreeIG attributions for every observation in parallel."""
    n_obs = X.shape[0]
    p = X.shape[1]
    n_trees = cl.shape[0]
    buf = cl.shape[1] * 2

    phis = np.zeros((n_obs, p), dtype=np.float64)
    event_counts = np.zeros(n_obs, dtype=np.int64)

    dx_buf = np.empty((n_obs, p), dtype=np.float64)
    stk_node = np.empty((n_obs, buf), dtype=np.int64)
    stk_tl = np.empty((n_obs, buf), dtype=np.float64)
    stk_tr = np.empty((n_obs, buf), dtype=np.float64)
    stk_ef = np.empty((n_obs, buf), dtype=np.int64)
    seg_tl = np.empty((n_obs, buf), dtype=np.float64)
    seg_val = np.empty((n_obs, buf), dtype=np.float64)
    seg_ef = np.empty((n_obs, buf), dtype=np.int64)
    probe = np.empty((n_obs, p), dtype=np.float64)

    for i in prange(n_obs):
        for j in range(p):
            dx_buf[i, j] = X[i, j] - x0[j]

        for m in range(n_trees):
            event_counts[i] += _attribute_one_tree(
                cl,
                cr,
                feat,
                thresh,
                val,
                left_inc,
                round_input,
                m,
                x0,
                X[i],
                dx_buf[i],
                y0_per_tree[m],
                tree_weight[m],
                time_tol,
                phis[i],
                stk_node[i],
                stk_tl[i],
                stk_tr[i],
                stk_ef[i],
                seg_tl[i],
                seg_val[i],
                seg_ef[i],
                probe[i],
            )

    return phis, event_counts


def _arrays_signature(arrays: Dict[str, Any]) -> Tuple[str, Optional[int]]:
    return (str(arrays.get("backend", "unknown")), arrays.get("target", None))


def _baseline_cache_key(arrays: Dict[str, Any], baseline: np.ndarray) -> Tuple[str, Optional[int], bytes]:
    backend, target = _arrays_signature(arrays)
    return (backend, target, baseline.tobytes())


def _round_input_array(arrays: Dict[str, Any]) -> np.ndarray:
    cl = arrays["children_left"]
    round_input = arrays.get("round_input", None)
    if round_input is None:
        return np.ones_like(cl, dtype=np.bool_)
    round_input = np.asarray(round_input, dtype=np.bool_)
    if round_input.shape != cl.shape:
        raise ValueError(
            f"round_input must have shape {cl.shape}, got {round_input.shape}."
        )
    return round_input


def _compute_attributions_with_y0(
    arrays: Dict[str, Any],
    baseline: np.ndarray,
    X: np.ndarray,
    time_tol: float,
    y0_per_tree: np.ndarray,
):
    cl = arrays["children_left"]
    cr = arrays["children_right"]
    ft = arrays["feature"]
    th = arrays["threshold"]
    va = arrays["value"]
    li = arrays["left_inclusive"]
    ri = _round_input_array(arrays)
    tw = arrays["tree_weight"]

    return _compute_batch(cl, cr, ft, th, va, li, ri, baseline, X, y0_per_tree, tw, time_tol)


def _compute_y0_per_tree(arrays: Dict[str, Any], baseline: np.ndarray) -> np.ndarray:
    cl = arrays["children_left"]
    cr = arrays["children_right"]
    ft = arrays["feature"]
    th = arrays["threshold"]
    va = arrays["value"]
    li = arrays["left_inclusive"]
    ri = _round_input_array(arrays)
    return _baseline_leaf_values(cl, cr, ft, th, va, li, ri, baseline)


def _compute_core(
    arrays: Dict[str, Any],
    baseline: np.ndarray,
    X: np.ndarray,
    time_tol: float,
    y0_per_tree: np.ndarray,
    endpoint_delta: np.ndarray,
):
    """Run TreeIG and compute diagnostics."""
    phis, event_counts = _compute_attributions_with_y0(
        arrays,
        baseline,
        X,
        time_tol,
        y0_per_tree,
    )

    residuals = phis.sum(axis=1) - endpoint_delta

    infos = [
        {
            "n_events": int(event_counts[i]),
            "endpoint_delta": float(endpoint_delta[i]),
            "attribution_sum": float(phis[i].sum()),
            "residual": float(residuals[i]),
            "abs_residual": float(abs(residuals[i])),
        }
        for i in range(X.shape[0])
    ]

    summary = {
        "mean_abs_residual": float(np.mean(np.abs(residuals))),
        "median_abs_residual": float(np.median(np.abs(residuals))),
        "max_abs_residual": float(np.max(np.abs(residuals))),
        "mean_events": float(np.mean(event_counts)),
        "median_events": float(np.median(event_counts)),
        "max_events": int(np.max(event_counts)),
    }

    return phis, infos, summary
