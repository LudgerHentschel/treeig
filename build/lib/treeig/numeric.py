"""
Numeric path-event attribution for piecewise-constant models.

This module implements :class:`TreeIGNumeric`, a model-agnostic counterpart to
the exact, structure-based :class:`TreeIG` explainer. Unlike ``TreeIG``,
``TreeIGNumeric`` does not parse tree internals. It treats the fitted model as
an opaque scalar function evaluated along the straight-line path

    x(t) = x0 + t (x - x0),    0 <= t <= 1,

detects prediction jumps along that path, and allocates each detected jump to
features using local axis-aligned probes.

The method is intended for piecewise-constant models whose internal tree
structure is unavailable, inconvenient, or not yet supported by the exact
parser, such as CatBoost models. It is not a silent fallback for ``TreeIG``:
users choose it explicitly because its guarantees differ from exact structural
TreeIG.

Guarantees
----------
* Detected events telescope exactly. The sum of reported feature attributions
  equals the sum of detected prediction jumps.
* The reported residual compares this detected jump sum with ``f(x) - f(x0)``.
  A nonzero residual indicates that the event search did not fully recover the
  endpoint prediction difference.
* Featurewise attributions match exact split-crossing attributions when each
  detected event contains a single responsible crossing and the local
  axis-aligned probe identifies that feature.
* Coincident or interacting crossings are allocated by a deterministic ordered
  cumulative sweep. This convention preserves completeness for the detected
  event but is order-dependent.

TreeIGNumeric is model-agnostic in its attribution engine. The adapter is
model-aware only to extract the scalar output being explained, preferably a
raw margin. It does not parse tree structure.

In contrast, :class:`TreeIG` uses model structure to enumerate split crossings
directly and is exact featurewise for supported tree implementations.
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Tuple

import numpy as np

ArrayF = np.ndarray
ScalarFn = Callable[[ArrayF], ArrayF]  # (m, p) -> (m,)


# ---------------------------------------------------------------------------
# Engine: pure NumPy, no model knowledge, operates on a scalar function f.
# ---------------------------------------------------------------------------
class NumericEngine:
    """
    Numerical event-detection engine for scalar path attribution.

    Model-free core of ``TreeIGNumeric``. Detects prediction jumps along the
    straight-line path by a batched grid scan, then identifies the responsible
    feature for each jump by a *batched binary search* over feature indices:
    ~log2(p) group probes per event instead of p single-feature probes. Changed
    intervals are subdivided adaptively before feature identification. Truly
    coincident, interacting, or still-unresolved crossings fall back to an
    ordered cumulative sweep, which preserves completeness.

    The initial grid scan, adaptive subdivision, and feature search are
    batched. Only intervals whose endpoints differ are subdivided.
    ``atom_chunk`` bounds batches of interval probes.

    Parameters
    ----------
    f : callable
        Vectorized scalar function ``(m, p) -> (m,)``.
    n_features : int
        Number of input features ``p``.
    grid_size : int, default=1024
        Number of path intervals scanned for prediction changes.
    max_refine : int, default=4
        Maximum bisection depth for changed intervals. Set to zero to disable
        adaptive refinement.
    t_min : float, default=1e-9
        Minimum path-time width for adaptive refinement.
    tol : float, default=0.0
        Tolerance for comparing scalar predictions.
    warn_residual : bool, default=True
        Warn when the recovered attribution sum misses the endpoint difference.
    obs_chunk, atom_chunk : int
        Batch sizes bounding peak memory.
    """

    def __init__(
        self,
        f,
        n_features,
        *,
        grid_size: int = 1024,
        max_refine: int = 4,
        t_min: float = 1e-9,
        tol: float = 0.0,
        warn_residual: bool = True,
        obs_chunk: int = 64,
        atom_chunk: int = 8192,
    ) -> None:
        self.f = f
        self.p = int(n_features)
        self.grid_size = int(grid_size)
        self.max_refine = int(max_refine)
        self.t_min = float(t_min)
        self.tol = float(tol)
        self.warn_residual = bool(warn_residual)
        self.obs_chunk = int(obs_chunk)
        self.atom_chunk = int(atom_chunk)

        if self.p <= 0:
            raise ValueError("n_features must be positive")
        if self.grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if self.max_refine < 0:
            raise ValueError("max_refine must be nonnegative")
        if self.t_min <= 0.0:
            raise ValueError("t_min must be positive")
        if self.tol < 0.0:
            raise ValueError("tol must be nonnegative")
        if self.obs_chunk <= 0 or self.atom_chunk <= 0:
            raise ValueError("obs_chunk and atom_chunk must be positive")

    # -- public ------------------------------------------------------------
    def attribute(self, x0, X):
        x0 = np.asarray(x0, dtype=float).ravel()
        X = np.atleast_2d(np.asarray(X, dtype=float))

        if x0.size != self.p or X.shape[1] != self.p:
            raise ValueError(
                f"feature dimension mismatch: engine p={self.p}, "
                f"baseline={x0.size}, X has {X.shape[1]}"
            )
        if not np.isfinite(x0).all() or not np.isfinite(X).all():
            raise ValueError("inputs must be finite; NaN/Inf not supported")

        N = X.shape[0]
        phi = np.zeros((N, self.p), dtype=float)
        infos = [None] * N
        T = np.linspace(0.0, 1.0, self.grid_size + 1)
        for s in range(0, N, self.obs_chunk):
            e = min(s + self.obs_chunk, N)
            self._chunk(x0, X[s:e], T, phi[s:e], infos, s)
        return phi, infos

    # -- per chunk ---------------------------------------------------------
    def _chunk(self, x0, Xc, T, phi_c, infos, base):
        nc, p = Xc.shape
        G = self.grid_size
        tol = self.tol
        ac = self.atom_chunk
        diff = Xc - x0[None, :]

        # pass 1: one batched grid scan for the whole chunk
        P = x0[None, None, :] + T[None, :, None] * diff[:, None, :]
        V = np.asarray(self.f(P.reshape(nc * (G + 1), p)), dtype=float)
        V = V.reshape(nc, G + 1)
        endpoint_delta = V[:, -1] - V[:, 0]
        changed = np.abs(V[:, 1:] - V[:, :-1]) > tol

        refined_counts = np.zeros(nc, dtype=int)
        max_refinement_depth = np.zeros(nc, dtype=int)
        unresolved_counts = np.zeros(nc, dtype=int)
        ii, kk = np.nonzero(changed)

        if ii.size:
            fa = V[ii, kk]
            fb = V[ii, kk + 1]
            x_minus = x0[None, :] + T[kk][:, None] * diff[ii]
            x_plus = x0[None, :] + T[kk + 1][:, None] * diff[ii]
            (
                ii,
                x_minus,
                x_plus,
                fa,
                fb,
                refined_counts,
                max_refinement_depth,
            ) = self._refine_changed_intervals(
                ii, x_minus, x_plus, fa, fb, nc
            )
            delta = fb - fa
            A = ii.size

            # pass 2: batched binary search for the responsible feature.
            # Each event keeps a candidate index range [lo, hi); each round
            # probes its lower half [lo, mid). Reaching fb -> responsible in the
            # lower half; staying at fa -> upper half; anything else -> the jump
            # is not due to a single feature (coincident/interacting) -> sweep.
            lo = np.zeros(A, dtype=int)
            hi = np.full(A, p, dtype=int)
            bad = np.zeros(A, dtype=bool)
            feat = np.arange(p)

            while True:
                active = np.nonzero((~bad) & ((hi - lo) > 1))[0]
                if active.size == 0:
                    break
                for s in range(0, active.size, ac):
                    sel = active[s:s + ac]
                    lo_s, hi_s = lo[sel], hi[sel]
                    mid_s = (lo_s + hi_s) // 2
                    mask = (feat[None, :] >= lo_s[:, None]) & \
                           (feat[None, :] < mid_s[:, None])
                    probe = np.where(mask, x_plus[sel], x_minus[sel])
                    r = np.asarray(self.f(probe), dtype=float).ravel()
                    eq_fb = np.abs(r - fb[sel]) <= tol
                    eq_fa = np.abs(r - fa[sel]) <= tol
                    hi[sel[eq_fb]] = mid_s[eq_fb]      # lower half carries the jump
                    lo[sel[eq_fa]] = mid_s[eq_fa]      # lower half inert
                    bad[sel[~(eq_fa | eq_fb)]] = True  # not a single feature

            # verification: the converged feature alone must reproduce the jump
            conv = np.nonzero((~bad) & ((hi - lo) == 1))[0]
            for s in range(0, conv.size, ac):
                sel = conv[s:s + ac]
                j = lo[sel]
                probe = x_minus[sel].copy()
                probe[np.arange(sel.size), j] = x_plus[sel, j]
                r = np.asarray(self.f(probe), dtype=float).ravel()
                ok = np.abs(r - fb[sel]) <= tol
                good = sel[ok]
                np.add.at(phi_c, (ii[good], lo[good]), delta[good])
                bad[sel[~ok]] = True

            # Truly coincident or still-unresolved events retain the
            # deterministic cumulative sweep after refinement.
            for a in np.nonzero(bad)[0]:
                obs = ii[a]
                unresolved_counts[obs] += 1
                self._sweep(
                    x_minus[a], x_plus[a], fa[a], diff[obs], obs, phi_c
                )

        event_counts = (
            np.bincount(ii, minlength=nc)
            if ii.size
            else np.zeros(nc, int)
        )
        attr_sum = phi_c.sum(axis=1)
        for i in range(nc):
            res = float(attr_sum[i] - endpoint_delta[i])
            abs_res = abs(res)
            infos[base + i] = {
                "n_events": int(event_counts[i]),
                "endpoint_delta": float(endpoint_delta[i]),
                "attribution_sum": float(attr_sum[i]),
                "residual": res,
                "abs_residual": abs_res,
                "engine": "numeric",
                "n_coincident_events": int(unresolved_counts[i]),
                "n_refined_intervals": int(refined_counts[i]),
                "max_refinement_depth": int(max_refinement_depth[i]),
                "n_unresolved_intervals": int(unresolved_counts[i]),
            }
            if self.warn_residual and abs_res > tol:
                warnings.warn(
                    "TreeIGNumeric did not recover f(x) - f(x0) within "
                    "tolerance. Increase grid_size/max_refine or use "
                    "structure-based TreeIG when available.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _refine_changed_intervals(self, ii, xm, xp, fa, fb, n_obs):
        """Bisect changed intervals in batched adaptive levels."""
        refined = np.zeros(n_obs, dtype=int)
        max_depth = np.zeros(n_obs, dtype=int)
        settled = []
        width = 1.0 / self.grid_size

        for depth in range(self.max_refine):
            if ii.size == 0 or width <= self.t_min:
                break
            midpoint = 0.5 * (xm + xp)
            fm = np.empty(ii.size, dtype=float)
            for start in range(0, ii.size, self.atom_chunk):
                stop = min(start + self.atom_chunk, ii.size)
                fm[start:stop] = np.asarray(
                    self.f(midpoint[start:stop]), dtype=float
                ).ravel()

            np.add.at(refined, ii, 1)
            max_depth[np.unique(ii)] = depth + 1
            left = np.abs(fm - fa) > self.tol
            right = np.abs(fb - fm) > self.tol
            neither = ~(left | right)
            if np.any(neither):
                settled.append(
                    (
                        ii[neither],
                        xm[neither],
                        xp[neither],
                        fa[neither],
                        fb[neither],
                    )
                )

            ii = np.concatenate((ii[left], ii[right]))
            xm = np.concatenate((xm[left], midpoint[right]), axis=0)
            xp = np.concatenate((midpoint[left], xp[right]), axis=0)
            fa = np.concatenate((fa[left], fm[right]))
            fb = np.concatenate((fm[left], fb[right]))
            width *= 0.5

        settled.append((ii, xm, xp, fa, fb))
        return (
            np.concatenate([part[0] for part in settled]),
            np.concatenate([part[1] for part in settled], axis=0),
            np.concatenate([part[2] for part in settled], axis=0),
            np.concatenate([part[3] for part in settled]),
            np.concatenate([part[4] for part in settled]),
            refined,
            max_depth,
        )

    def _sweep(self, xm, xp, fa, diff_row, row, phi_c):
        moving = np.flatnonzero(diff_row != 0.0)
        m = moving.size
        if m == 0:
            return
        pts = np.tile(xm, (m, 1))
        for k in range(m):
            pts[k:, moving[k]] = xp[moving[k]]
        vals = np.asarray(self.f(pts), dtype=float).ravel()
        prev = fa
        for k in range(m):
            phi_c[row, moving[k]] += vals[k] - prev
            prev = vals[k]


class RefiningNumericEngine:
    """
    Numerical event-detection engine for scalar path attribution.

    ``NumericEngine`` is the model-free core of :class:`TreeIGNumeric`. It
    receives a vectorized scalar function ``f`` and a feature dimension, then
    computes attributions from repeated evaluations of ``f`` along
    interpolation paths.

    The engine has no knowledge of tree libraries, model classes, split
    thresholds, or leaf values. This makes it useful both for unsupported
    model backends and for unit tests based on synthetic piecewise-constant
    functions.

    Parameters
    ----------
    f : callable
        Vectorized scalar function. The callable must accept an ``(m, p)``
        floating point array and return an ``(m,)`` array containing the
        scalar quantity to explain.
    n_features : int
        Number of input features ``p``.
    grid_size : int, default=64
        Number of coarse intervals used to scan the path for prediction changes.
    max_refine : int, default=20
        Maximum number of bisection refinements used to separate multiple
        detected changes inside a coarse interval.
    t_min : float, default=1e-9
        Minimum interval width in path-time coordinates. Refinement stops once
        an interval is no wider than this value.
    tol : float, default=0.0
        Numerical tolerance used when deciding whether two scalar predictions
        differ.
    warn_residual : bool, default=True
        Whether to warn when the recovered attribution sum differs from the
        endpoint prediction difference by more than ``tol``.
    """

    def __init__(
        self,
        f: ScalarFn,
        n_features: int,
        *,
        grid_size: int = 64,
        max_refine: int = 20,
        t_min: float = 1e-9,
        tol: float = 0.0,
        warn_residual: bool = True,
    ) -> None:
        self.f = f
        self.p = int(n_features)
        self.grid_size = int(grid_size)
        self.max_refine = int(max_refine)
        self.t_min = float(t_min)
        self.tol = float(tol)
        self.warn_residual = bool(warn_residual)

        if self.p <= 0:
            raise ValueError("n_features must be positive")
        if self.grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if self.max_refine < 0:
            raise ValueError("max_refine must be nonnegative")
        if self.t_min <= 0.0:
            raise ValueError("t_min must be positive")
        if self.tol < 0.0:
            raise ValueError("tol must be nonnegative")

    # -- public ------------------------------------------------------------
    def attribute(self, x0: ArrayF, X: ArrayF) -> Tuple[ArrayF, List[Dict]]:
        x0 = np.asarray(x0, dtype=float).ravel()
        X = np.atleast_2d(np.asarray(X, dtype=float))

        if x0.size != self.p or X.shape[1] != self.p:
            raise ValueError(
                f"feature dimension mismatch: engine p={self.p}, "
                f"baseline={x0.size}, X has {X.shape[1]}"
            )
        if not np.isfinite(x0).all() or not np.isfinite(X).all():
            raise ValueError("inputs must be finite; NaN/Inf not supported")

        phi = np.zeros_like(X, dtype=float)
        infos: List[Dict] = []
        for i in range(X.shape[0]):
            phi[i], info = self._attribute_one(x0, X[i])
            infos.append(info)
        return phi, infos

    # -- per observation ---------------------------------------------------
    def _attribute_one(self, x0: ArrayF, x: ArrayF) -> Tuple[ArrayF, Dict]:
        diff = x - x0
        moving = np.flatnonzero(diff != 0.0)  # only these can be responsible
        phi = np.zeros(self.p, dtype=float)

        f_x0 = float(self.f(x0[None, :])[0])
        f_x = float(self.f(x[None, :])[0])
        endpoint_delta = f_x - f_x0

        info = {
            "n_events": 0,
            "endpoint_delta": endpoint_delta,
            "attribution_sum": 0.0,
            "residual": 0.0,
            "abs_residual": 0.0,
            "engine": "numeric",
            "n_coincident_events": 0,
        }

        if moving.size == 0:
            return phi, info  # path does not move; all attributions zero

        def g(ts: ArrayF) -> ArrayF:
            ts = np.asarray(ts, dtype=float).reshape(-1, 1)
            pts = x0[None, :] + ts * diff[None, :]
            return np.asarray(self.f(pts), dtype=float).ravel()

        atoms = self._atomic_intervals(g)
        info["n_events"] = len(atoms)

        for ta, tb, fa, fb in atoms:
            self._allocate_interval(x0, diff, ta, tb, fa, fb, moving, phi, info)

        attribution_sum = float(phi.sum())
        info["attribution_sum"] = attribution_sum
        info["residual"] = attribution_sum - endpoint_delta
        info["abs_residual"] = abs(info["residual"])

        if self.warn_residual and info["abs_residual"] > self.tol:
            warnings.warn(
                "TreeIGNumeric did not recover f(x) - f(x0) within tolerance. "
                "Increase grid_size/max_refine or use structure-based TreeIG "
                "when available.",
                RuntimeWarning,
                stacklevel=2,
            )

        return phi, info

    # -- helpers -----------------------------------------------------------
    def _changed(self, a: float, b: float) -> bool:
        return abs(float(a) - float(b)) > self.tol

    # -- pass 1: locate and separate crossings -----------------------------
    def _atomic_intervals(
        self, g: Callable[[ArrayF], ArrayF]
    ) -> List[Tuple[float, float, float, float]]:
        ts = np.linspace(0.0, 1.0, self.grid_size + 1)
        vs = g(ts)

        # Changed coarse intervals.
        stack: List[Tuple[float, float, float, float, int]] = []
        for i in range(1, ts.size):
            if self._changed(vs[i], vs[i - 1]):
                stack.append((ts[i - 1], ts[i], float(vs[i - 1]), float(vs[i]), 0))

        atoms: List[Tuple[float, float, float, float]] = []
        while stack:
            ta, tb, fa, fb, depth = stack.pop()

            if depth >= self.max_refine or (tb - ta) <= self.t_min:
                atoms.append((ta, tb, fa, fb))
                continue

            tm = 0.5 * (ta + tb)
            fm = float(g(np.array([tm]))[0])

            left_changed = self._changed(fa, fm)
            right_changed = self._changed(fm, fb)

            if left_changed and right_changed:  # at least two distinct changes
                stack.append((ta, tm, fa, fm, depth + 1))
                stack.append((tm, tb, fm, fb, depth + 1))
            elif left_changed:
                stack.append((ta, tm, fa, fm, depth + 1))
            elif right_changed:
                stack.append((tm, tb, fm, fb, depth + 1))
            else:
                # Defensive fallback: this can occur if small changes are
                # suppressed by tol or if the black-box prediction is unstable.
                atoms.append((ta, tb, fa, fb))

        atoms.sort(key=lambda z: z[0])
        return atoms

    # -- pass 2: identify responsible feature and allocate -----------------
    def _allocate_interval(
        self,
        x0: ArrayF,
        diff: ArrayF,
        ta: float,
        tb: float,
        fa: float,
        fb: float,
        moving: ArrayF,
        phi: ArrayF,
        info: Dict,
    ) -> None:
        x_minus = x0 + ta * diff
        x_plus = x0 + tb * diff
        delta = fb - fa

        # Axis-aligned probes: move each moving feature alone, interval-local.
        m = moving.size
        probes = np.tile(x_minus, (m, 1))
        for k in range(m):
            j = moving[k]
            probes[k, j] = x_plus[j]

        dvals = np.asarray(self.f(probes), dtype=float).ravel() - fa
        responsible = moving[np.abs(dvals) > self.tol]

        if responsible.size == 1:
            phi[responsible[0]] += delta
            return

        # Coincident or interacting crossing: ordered cumulative sweep.
        # This convention is order-dependent, but it telescopes to the detected
        # interval change.
        info["n_coincident_events"] += 1
        self._sweep_allocate(x_minus, x_plus, fa, moving, phi)

    def _sweep_allocate(
        self,
        x_minus: ArrayF,
        x_plus: ArrayF,
        fa: float,
        moving: ArrayF,
        phi: ArrayF,
    ) -> None:
        m = moving.size
        pts = np.tile(x_minus, (m, 1))

        for k in range(m):
            j = moving[k]
            pts[k:, j] = x_plus[j]  # rows k..m-1 have moving[0..k] advanced

        vals = np.asarray(self.f(pts), dtype=float).ravel()
        prev = fa

        for k in range(m):
            phi[moving[k]] += vals[k] - prev
            prev = vals[k]


# ---------------------------------------------------------------------------
# Model adapter: the only backend-aware code on the numeric path.
# Maps (model, target) -> scalar function returning the explained quantity.
# This is about output extraction, not structure parsing, so it is short
# and stable across library versions, and inherits each backend's routing.
# ---------------------------------------------------------------------------
def _warn_exact_available(model_name: str) -> None:
    warnings.warn(
        f"{model_name} is supported by structure-based TreeIG. "
        "TreeIGNumeric will still run, but it is a numerical event-search "
        "method and may miss or merge crossings. Prefer TreeIG for exact "
        "featurewise attributions when model structure is supported.",
        RuntimeWarning,
        stacklevel=3,
    )


def _warn_if_sklearn_treeig_supported(model) -> None:
    """Warn when a scikit-learn tree model has exact TreeIG support."""
    try:
        from sklearn.ensemble import (
            ExtraTreesRegressor,
            GradientBoostingClassifier,
            GradientBoostingRegressor,
            RandomForestRegressor,
        )
        from sklearn.tree import DecisionTreeRegressor

        supported_types = (
            DecisionTreeRegressor,
            RandomForestRegressor,
            ExtraTreesRegressor,
            GradientBoostingClassifier,
            GradientBoostingRegressor,
        )

        if isinstance(model, supported_types):
            _warn_exact_available(type(model).__name__)
    except Exception:
        pass


def _select_margin(out: ArrayF, target) -> ArrayF:
    out = np.asarray(out, dtype=float)

    if out.ndim == 1:  # regression, or binary positive-class margin
        if target in (None, 1):
            return out
        if target == 0:
            return -out
        raise ValueError(f"target={target!r} invalid for 1D/binary output")

    if target is None:
        raise ValueError("multiclass output requires an explicit target index")

    return out[:, int(target)]


def _select_proba(proba: ArrayF, target) -> ArrayF:
    proba = np.asarray(proba, dtype=float)

    if proba.ndim == 1:
        return proba

    if target is None:
        if proba.shape[1] == 2:
            target = 1
        else:
            raise ValueError("multiclass probability requires an explicit target")

    return proba[:, int(target)]


def _select_probability_score(
    proba: ArrayF,
    target,
    probability_floor=None,
) -> ArrayF:
    """Convert probabilities to a binary margin or centered log score."""

    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        proba = np.column_stack((1.0 - proba, proba))
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError(
            "class probabilities must have shape (n_observations, n_classes)"
        )
    if (
        not np.isfinite(proba).all()
        or np.any(proba < 0.0)
        or np.any(proba > 1.0)
    ):
        raise ValueError("class probabilities must be finite and in [0, 1]")
    row_sums = proba.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("class probabilities must have a positive row sum")
    proba = proba / row_sums

    if probability_floor is None:
        if np.any(proba <= 0.0):
            raise ValueError(
                "probability-derived scores are infinite when a class "
                "probability is zero; set probability_floor explicitly"
            )
    else:
        floor = float(probability_floor)
        if not np.isfinite(floor) or not 0.0 < floor < 1.0:
            raise ValueError("probability_floor must be strictly between 0 and 1")
        proba = np.maximum(proba, floor)
        proba = proba / proba.sum(axis=1, keepdims=True)

    log_proba = np.log(proba)
    n_classes = proba.shape[1]
    if n_classes == 2:
        if target in (None, 1):
            return log_proba[:, 1] - log_proba[:, 0]
        if target == 0:
            return log_proba[:, 0] - log_proba[:, 1]
        raise ValueError(f"target={target!r} invalid for binary probabilities")

    if target is None:
        raise ValueError("multiclass probability scores require a target index")
    target = int(target)
    if target < 0 or target >= n_classes:
        raise ValueError(f"target={target!r} invalid for {n_classes} classes")
    return log_proba[:, target] - log_proba.mean(axis=1)


def make_scalar_fn(
    model,
    target=None,
    *,
    probability_to_score: bool = False,
    probability_floor=None,
) -> ScalarFn:
    """Construct a vectorized scalar-output function for attribution.

    ``TreeIGNumeric`` operates on scalar functions of the form

        f : R^p -> R.

    This helper converts a fitted model into a vectorized callable returning
    the scalar quantity to be explained. The resulting function accepts an
    ``(m, p)`` array of feature values and returns an ``(m,)`` array of
    scalar outputs suitable for the numeric attribution engine.

    The helper is model-aware only in the limited sense needed to extract the
    scalar output. It does not inspect split thresholds, traverse trees, or
    parse tree structure.

    For classifiers, the preferred quantity is an additive raw margin or
    logit because additive outputs preserve the natural completeness property
    of attribution methods. When no margin interface is available, the
    function falls back to class probabilities, in which case completeness
    holds in probability space rather than margin space.

    If the fitted model is already supported by exact structure-based
    ``TreeIG``, this helper emits a warning. The numeric path remains
    available, but exact ``TreeIG`` should generally be preferred for supported
    models.

    Supported model interfaces
    --------------------------
    * XGBoost native boosters via ``predict(..., output_margin=True)``
    * LightGBM native boosters via ``predict(..., raw_score=True)``
    * CatBoost models via
      ``predict(..., prediction_type="RawFormulaVal")``
    * Scikit-learn classifiers exposing ``decision_function``
    * XGBoost and LightGBM scikit-learn wrappers
    * Classifiers exposing only ``predict_proba`` (probability or derived score)
    * Regressors exposing ``predict``

    Parameters
    ----------
    model : object
        Fitted model to explain.
    target : int or None, default=None
        Output index for classification models.

        For binary classifiers, ``None`` selects the positive-class margin
        when available. For multiclass classifiers, an explicit target index
        is required whenever the model returns one score per class.
    probability_to_score : bool, default=False
        When no native margin exists, explain binary log odds or centered
        multiclass log probabilities rather than a class probability.
    probability_floor : float or None, default=None
        Explicit lower bound applied before taking logarithms. If omitted,
        encountering a zero probability raises an actionable error.

    Returns
    -------
    callable
        Vectorized scalar function ``f(P)`` where ``P`` has shape
        ``(m, p)`` and the returned array has shape ``(m,)``.

    Notes
    -----
    This function performs output extraction only. It does not inspect,
    traverse, or parse model structure. Consequently, it remains stable
    across model implementations and library versions while inheriting the
    prediction semantics of the underlying model.

    The returned scalar function is the only model-specific component used
    by ``NumericEngine``. All event detection, crossing localization, and
    attribution logic operate purely on repeated evaluations of this
    function.
    """
    # Native XGBoost booster. Exact TreeIG supports XGBoost tree structures.
    try:
        import xgboost as xgb

        if isinstance(model, xgb.Booster):
            _warn_exact_available("xgboost.Booster")

            def f(P):
                d = xgb.DMatrix(np.asarray(P, dtype=float))
                return _select_margin(model.predict(d, output_margin=True), target)

            return f
    except Exception:
        pass

    # Native LightGBM booster. Exact TreeIG supports LightGBM tree structures.
    try:
        import lightgbm as lgb

        if isinstance(model, lgb.Booster):
            _warn_exact_available("lightgbm.Booster")

            def f(P):
                out = model.predict(np.asarray(P, dtype=float), raw_score=True)
                return _select_margin(out, target)

            return f
    except Exception:
        pass

    # CatBoost estimators. This path does not parse CatBoost internals; it only
    # requests raw formula values from CatBoost's public prediction API.
    # No exact-TreeIG warning is issued here because CatBoost is a main use case
    # for the numeric structure-free path.
    try:
        import catboost as cb

        catboost_types = (
            cb.CatBoost,
            cb.CatBoostClassifier,
            cb.CatBoostRegressor,
        )

        if isinstance(model, catboost_types):

            def f(P):
                out = model.predict(
                    np.asarray(P, dtype=float),
                    prediction_type="RawFormulaVal",
                )
                return _select_margin(out, target)

            return f
    except Exception:
        pass

    # XGBoost scikit-learn wrappers. Place before generic sklearn classifier
    # handling so raw margins are used for both classifiers and regressors.
    try:
        import xgboost as xgb

        if isinstance(model, xgb.XGBModel):
            _warn_exact_available(type(model).__name__)

            def f(P):
                return _select_margin(
                    model.predict(
                        np.asarray(P, dtype=float),
                        output_margin=True,
                    ),
                    target,
                )

            return f
    except Exception:
        pass

    # LightGBM scikit-learn wrappers. Place before generic sklearn classifier
    # handling so raw scores are used for both classifiers and regressors.
    try:
        import lightgbm as lgb

        if isinstance(model, lgb.LGBMModel):
            _warn_exact_available(type(model).__name__)

            def f(P):
                return _select_margin(
                    model.predict(
                        np.asarray(P, dtype=float),
                        raw_score=True,
                    ),
                    target,
                )

            return f
    except Exception:
        pass

    # sklearn-style estimators.
    _warn_if_sklearn_treeig_supported(model)

    try:
        from sklearn.base import is_classifier

        is_clf = is_classifier(model)
    except Exception:
        is_clf = hasattr(model, "predict_proba") or hasattr(model, "classes_")

    if is_clf:
        if hasattr(model, "decision_function"):

            def f(P):
                return _select_margin(
                    model.decision_function(np.asarray(P, dtype=float)), target
                )

            return f

        if hasattr(model, "predict_proba"):
            if probability_to_score:
                warnings.warn(
                    "No native decision score is available; deriving a score "
                    "from class probabilities.",
                    RuntimeWarning,
                    stacklevel=2,
                )

                def f(P):
                    return _select_probability_score(
                        model.predict_proba(np.asarray(P, dtype=float)),
                        target,
                        probability_floor,
                    )

                return f

            warnings.warn(
                "No additive margin available; explaining a class probability. "
                "Completeness holds in probability space "
                "(sum phi = p(x) - p(x0)).",
                RuntimeWarning,
                stacklevel=2,
            )

            def f(P):
                return _select_proba(
                    model.predict_proba(np.asarray(P, dtype=float)), target
                )

            return f

        raise TypeError(f"unsupported classifier type: {type(model).__name__}")

    # Regressor / generic predict.
    if not hasattr(model, "predict"):
        raise TypeError(
            f"unsupported model type: {type(model).__name__}; expected a model "
            "with predict, predict_proba, decision_function, or a supported "
            "booster interface"
        )

    def f(P):
        return np.asarray(
            model.predict(np.asarray(P, dtype=float)), dtype=float
        ).ravel()

    return f


# ---------------------------------------------------------------------------
# Public explainer, mirroring the TreeIG API surface.
# ---------------------------------------------------------------------------
def _summarize(infos: List[Dict]) -> Dict:
    abs_res = np.array([d["abs_residual"] for d in infos], dtype=float)
    n_ev = np.array([d["n_events"] for d in infos], dtype=float)
    n_co = np.array([d["n_coincident_events"] for d in infos], dtype=float)
    n_ref = np.array(
        [d.get("n_refined_intervals", 0) for d in infos], dtype=float
    )
    n_un = np.array(
        [d.get("n_unresolved_intervals", 0) for d in infos], dtype=float
    )
    depths = np.array(
        [d.get("max_refinement_depth", 0) for d in infos], dtype=float
    )

    return {
        "engine": "numeric",
        "n_observations": len(infos),
        "max_abs_residual": float(abs_res.max()) if abs_res.size else 0.0,
        "mean_abs_residual": float(abs_res.mean()) if abs_res.size else 0.0,
        "mean_n_events": float(n_ev.mean()) if n_ev.size else 0.0,
        "total_coincident_events": int(n_co.sum()),
        "total_refined_intervals": int(n_ref.sum()),
        "total_unresolved_intervals": int(n_un.sum()),
        "max_refinement_depth": int(depths.max()) if depths.size else 0,
    }


class TreeIGNumeric:
    """
    Model-agnostic numeric TreeIG-style explainer.

    ``TreeIGNumeric`` applies numeric path-event detection to a fitted model.
    It is designed for piecewise-constant models whose tree structure is not
    parsed by the exact :class:`TreeIG` backend. The class mirrors the
    high-level TreeIG API but intentionally provides different guarantees.

    Parameters
    ----------
    model : object
        Fitted model exposing a supported prediction interface. Raw margins are
        used when available. Probability outputs are used only as a fallback.
    baseline : array-like of shape (p,)
        Baseline input ``x0`` for the interpolation path.
    target : int or None, default=None
        Target output for classification models. Binary classifiers default to
        the positive-class margin or probability where possible. Multiclass
        outputs require an explicit target.
    probability_to_score : bool, default=False
        For classifiers without native margins, transform probabilities to
        binary log odds or centered multiclass log scores.
    probability_floor : float or None, default=None
        Explicit lower bound used before the logarithm. Without a floor, zero
        probabilities raise rather than being clipped silently.
    **engine_kwargs
        Optional controls passed to :class:`NumericEngine`, such as
        ``grid_size``, ``max_refine``, ``t_min``, ``tol``, and
        ``warn_residual``. See :class:`NumericEngine` for details.

    Notes
    -----
    This class does not parse split thresholds and should not be described as
    exact structural TreeIG. It is a structure-free numerical event detector
    whose accuracy depends on recovering the relevant prediction jumps along
    the path.
    """

    def __init__(
        self,
        model,
        baseline,
        target=None,
        *,
        probability_to_score: bool = False,
        probability_floor=None,
        **engine_kwargs,
    ) -> None:
        if not isinstance(probability_to_score, bool):
            raise TypeError("probability_to_score must be a boolean")
        if probability_floor is not None:
            floor = float(probability_floor)
            if not np.isfinite(floor) or not 0.0 < floor < 1.0:
                raise ValueError(
                    "probability_floor must be strictly between 0 and 1"
                )
            if not probability_to_score:
                raise ValueError(
                    "probability_floor requires probability_to_score=True"
                )
        self.model = model
        self.baseline = np.asarray(baseline, dtype=float).ravel()
        self.target = target
        self.probability_to_score = probability_to_score
        self.probability_floor = probability_floor
        self._f = make_scalar_fn(
            model,
            target,
            probability_to_score=probability_to_score,
            probability_floor=probability_floor,
        )
        self._engine = NumericEngine(
            self._f,
            n_features=self.baseline.size,
            **engine_kwargs,
        )

    def attribute(self, X) -> ArrayF:
        phi, _ = self._engine.attribute(self.baseline, X)
        return phi

    def model_output(self, X) -> ArrayF:
        """Return the scalar model output attributed by this explainer.

        The output scale follows the model adapter: raw margins are preferred
        for classifiers, probabilities are used only when no margin interface
        exists, and regressors use their predictions. ``target`` selection is
        the same as for :meth:`attribute`.
        """

        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.ndim != 2 or X.shape[1] != self.baseline.size:
            raise ValueError(
                f"X must have shape (n_observations, {self.baseline.size})"
            )
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one observation")
        if not np.isfinite(X).all():
            raise ValueError("inputs must be finite; NaN/Inf not supported")
        output = np.asarray(self._f(X), dtype=float).ravel()
        if output.shape != (X.shape[0],):
            raise ValueError("model output must contain one scalar per observation")
        if not np.isfinite(output).all():
            raise ValueError("model output must contain only finite values")
        return output

    def explain(self, X):
        phi, infos = self._engine.attribute(self.baseline, X)
        return phi, infos, _summarize(infos)

    def warmup(self, X=None):
        # No Numba kernels on the numeric path; present for API parity.
        return self


def compute_numeric(model, baseline, X, target=None, **engine_kwargs):
    """Functional mirror of :class:`TreeIGNumeric`."""
    return TreeIGNumeric(model, baseline, target=target, **engine_kwargs).explain(X)
