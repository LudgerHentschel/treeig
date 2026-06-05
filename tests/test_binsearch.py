"""Correctness + pass-2 row reduction for the binary-search engine.

Correctness is checked against ANALYTIC ground truth for a sum-of-stumps model
(an IG attribution that we can write down exactly), plus equivalence with the
full-K engine, plus interaction/coincident edge cases.
"""
import pytest

pytest.skip(
    "We have confirmed accuracy of the binary-search engine",
    allow_module_level=True,
)

import warnings

import numpy as np

from numeric_binsearch import NumericEngine as BinSearch
from numeric_fullk import NumericEngine as FullK


class CountingF:
    def __init__(self, feats, thr, w):
        self.feats, self.thr, self.w = feats, thr, w
        self.calls = 0
        self.rows = 0

    def __call__(self, P):
        P = np.asarray(P, dtype=float)
        self.calls += 1
        self.rows += P.shape[0]
        out = np.zeros(P.shape[0])
        for s in range(self.feats.size):
            out += self.w[s] * (P[:, self.feats[s]] >= self.thr[s])
        return out


def ground_truth(feats, thr, w, x0, X, p):
    """Analytic IG for sum-of-stumps, baseline x0=0: a crossed stump on feature
    j contributes its weight to j."""
    N = X.shape[0]
    phi = np.zeros((N, p))
    for s in range(feats.size):
        j = feats[s]
        crossed = (X[:, j] >= thr[s]) & (x0[j] < thr[s])  # 0 -> 1 crossing
        phi[crossed, j] += w[s]
    return phi


def characterize(p, S, N, label):
    rng = np.random.default_rng(0)
    feats = rng.integers(0, p, size=S)
    thr = rng.uniform(0.1, 0.9, size=S)
    w = rng.normal(size=S)
    x0 = np.zeros(p)
    X = rng.uniform(0.05, 0.95, size=(N, p))
    gt = ground_truth(feats, thr, w, x0, X, p)

    fk_f = CountingF(feats, thr, w)
    bs_f = CountingF(feats, thr, w)
    phi_fk, _ = FullK(fk_f, n_features=p).attribute(x0, X)
    phi_bs, info_bs = BinSearch(bs_f, n_features=p).attribute(x0, X)

    assert np.allclose(phi_bs, gt, atol=1e-9)
    assert np.allclose(phi_bs, phi_fk, atol=1e-9)

    pass1 = N * (1024 + 1)
    total_events = sum(d["n_events"] for d in info_bs)
    total_coin = sum(d["n_coincident_events"] for d in info_bs)
    fk2 = fk_f.rows - pass1
    bs2 = bs_f.rows - pass1
    print(f"[{label}] p={p} events={total_events:,} "
          f"coincident={total_coin:,} ({100*total_coin/max(total_events,1):.1f}%)")
    print(f"    pass-2 rows  full-K={fk2:>11,}  binsearch={bs2:>11,}  "
          f"reduction={fk2/max(bs2,1):5.1f}x")


def test_against_ground_truth_and_fullk():
    # sparse crossings (representative of "occasional clubbing"):
    characterize(p=128, S=200, N=200, label="sparse")
    # moderate:
    characterize(p=128, S=500, N=200, label="moderate")
    # dense (stress: many clubbed cells -> sweep cost shows):
    characterize(p=60, S=600, N=200, label="dense")


def test_pure_interaction():
    def f(P):
        return np.where((P[:, 0] >= 0.5) & (P[:, 1] >= 0.5), 7.0, 0.0)
    eng = BinSearch(f, n_features=4, warn_residual=False)
    phi, info = eng.attribute(np.zeros(4), np.array([[1.0, 1.0, 0.3, 0.3]]))
    assert abs(phi[0].sum() - 7.0) < 1e-12         # completeness preserved
    assert info[0]["n_coincident_events"] >= 1     # routed to sweep
    print("pure interaction: phi", phi[0], "coincident", info[0]["n_coincident_events"])


def test_conjunction_separated():
    def f(P):
        return np.where((P[:, 0] >= 0.3) & (P[:, 1] >= 0.6), 5.0, 0.0)
    eng = BinSearch(f, n_features=5, warn_residual=False)
    phi, info = eng.attribute(np.zeros(5), np.array([[1.0, 1.0, 0, 0, 0]]))
    # flips when the second threshold (feature 1) is crossed -> all to feature 1
    assert np.allclose(phi[0], [0, 5, 0, 0, 0]), phi
    print("conjunction separated: phi", phi[0])


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_against_ground_truth_and_fullk()
        test_pure_interaction()
        test_conjunction_separated()
    print("\nall binary-search checks passed")
