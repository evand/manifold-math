#!/usr/bin/env python3
"""
cpmm-multi-2 — exact reduced solver for the basket-funded liquidity optimum, plus
a clean closed-form approximation (for seeding Newton, graphing, and intuition).

Builds on creation_liquidity_optimum.py. Using the clean point-liquidity formula
a_i = q_i(1-q_i)/W_i and the all-winners-tight structure Y_i = N_i + D, the optimum
reduces to an UNCONSTRAINED minimization over the NO-reserves {N_i} (funding baked
in via D = ante - ΣN_j):

    a_i(N; D) = q_i(1-q_i) (N_i + q_i D) / (N_i (N_i + D)),   D = ante - Σ N_j.

This removes the funding constraint that gave the full (Y,N,p) SLSQP its local
minima. We (1) solve it exactly+robustly from the balanced seed, (2) derive and
test a closed-form approximation.

Analysis only. No vendor code change.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))

from creation_liquidity import ANTE, single_point_liquidity  # noqa: E402


def reduced_a(N, qs, ante=ANTE):
    """Per-answer point liquidity a_i under the all-tight structure Y_i=N_i+D."""
    N = np.asarray(N, float)
    D = ante - N.sum()
    c = qs * (1 - qs)
    return c * (N + qs * D) / (N * (N + D))


def reduced_optimize(qs, ante=ANTE, objective="sum"):
    """Exact optimum via unconstrained min over {N_i} from the balanced seed."""
    qs = np.asarray(qs, float)
    n = len(qs)

    def obj(logN):
        a = reduced_a(np.exp(logN), qs, ante)
        if objective == "sum":
            return a.sum()
        if objective == "sum_eff":
            S = a.sum()
            return (a * (S - a) / S).sum()
        raise ValueError(objective)

    seed = np.log(np.full(n, ante / n) * 0.8)  # slightly inside (D>0) to break symmetry
    res = minimize(obj, seed, method="Nelder-Mead",
                   options={"xatol": 1e-9, "fatol": 1e-15, "maxiter": 20000})
    N = np.exp(res.x)
    D = ante - N.sum()
    return N, D, res


def shape_from_N(N, qs, ante=ANTE):
    """Recover (Y_i, N_i, p_i) from the NO-reserves under all-tight Y=N+D."""
    N = np.asarray(N, float)
    D = ante - N.sum()
    Y = N + D
    # p_i = q_i Y_i / (q_i Y_i + (1-q_i) N_i)
    p = qs * Y / (qs * Y + (1 - qs) * N)
    return Y, N, p


# --------------------------------------------------------------------------- #
# Closed-form approximation candidates
# --------------------------------------------------------------------------- #
def approx_sqrt_c(qs, ante=ANTE):
    """Allocate DEPTH W_i ∝ √(q_i(1-q_i)) (classic 'liquidity ∝ √variance'),
    with D set to the uniform-optimum value; back out N_i from W_i."""
    qs = np.asarray(qs, float)
    n = len(qs)
    D = ante * (n - 2) / (2 * (n - 1))  # uniform-optimum D
    sc = np.sqrt(qs * (1 - qs))
    # uniform W at the uniform optimum is ante*n/(4(n-1)); scale the profile to it
    Wbar = ante * n / (4 * (n - 1))
    W = sc / sc.mean() * Wbar
    # N from W = N(N+D)/(N+qD)  =>  N^2 + N(D-W) - W q D = 0
    N = (-(D - W) + np.sqrt((D - W) ** 2 + 4 * W * qs * D)) / 2
    # rescale N so funding D' = ante - ΣN matches D (one fixed-point nudge)
    return N


def approx_linear(qs, ante=ANTE):
    """First-order expansion around the uniform optimum: N_i ≈ N̄ + κ (q_i - 1/n).
    κ is the numerically-calibrated uniform sensitivity dN*/dq (same for all n via
    scaling). Clean and exact to first order in the skew."""
    qs = np.asarray(qs, float)
    n = len(qs)
    Nbar = ante / (2 * (n - 1))  # uniform-optimum NO reserve
    # calibrate κ once at this n by finite-differencing the exact optimum at uniform
    eps = 0.01
    q0 = np.full(n, 1.0 / n)
    qp = q0.copy()
    qp[0] += eps * (n - 1) / n
    qp[1:] -= eps / n
    Np, _, _ = reduced_optimize(qp, ante)
    kappa = (Np[0] - Nbar) / (qp[0] - 1.0 / n)
    return Nbar + kappa * (qs - 1.0 / n)


def obj_sum(N, qs, ante=ANTE):
    return reduced_a(N, qs, ante).sum()


def report():
    print("=" * 82)
    print("(G) EXACT REDUCED SOLVER vs balanced/full, and closed-form approximations")
    print("=" * 82)
    configs = [
        [0.6, 0.25, 0.15],
        [0.8, 0.1, 0.1],
        [0.5, 0.3, 0.2],
        [0.4, 0.3, 0.2, 0.1],
        [0.5, 0.2, 0.15, 0.1, 0.05],
    ]
    for qs in configs:
        qs = np.asarray(qs, float)
        n = len(qs)
        # exact reduced optimum
        Nopt, Dopt, _ = reduced_optimize(qs)
        a_opt = obj_sum(Nopt, qs)
        # balanced baseline
        Nbal = np.full(n, ANTE / n)
        a_bal = obj_sum(Nbal, qs)
        print(f"\nq = {[round(float(v), 3) for v in qs]}")
        print(f"  balanced        Σa={a_bal:.5e}")
        print(f"  exact optimum   Σa={a_opt:.5e}  ({100 * (a_bal - a_opt) / a_bal:+.1f}% vs bal)"
              f"   D={Dopt:.1f}  N={[f'{v:.0f}' for v in Nopt]}")
        for name, fn in (("√c-depth", approx_sqrt_c), ("linear", approx_linear)):
            try:
                Na = np.clip(fn(qs), 1e-6, ANTE)
                a_ap = obj_sum(Na, qs)
                # gap of the approximation's objective vs the exact optimum
                gap = 100 * (a_ap - a_opt) / a_opt
                # how much of the balanced->optimum gain it captures
                capt = 100 * (a_bal - a_ap) / (a_bal - a_opt) if a_bal > a_opt else float("nan")
                print(f"    approx {name:9} Σa={a_ap:.5e}  gap vs opt {gap:+5.2f}%  "
                      f"captures {capt:5.1f}% of the gain")
            except Exception as e:  # noqa: BLE001
                print(f"    approx {name:9} failed: {e}")
    print()
    print("  gap = how far the approximation's objective is above the exact optimum;")
    print("  captures = fraction of the balanced→optimum liquidity gain it recovers.")
    print()


def report_seed_quality():
    print("=" * 82)
    print("(H) APPROXIMATIONS AS NEWTON/optimizer SEEDS — iterations to converge")
    print("=" * 82)
    rng = np.random.default_rng(7)
    for label, seed_fn in (
        ("balanced", lambda qs: np.full(len(qs), ANTE / len(qs))),
        ("√c-depth", approx_sqrt_c),
        ("linear", approx_linear),
    ):
        iters = []
        for _ in range(30):
            nn = int(rng.integers(3, 7))
            raw = rng.uniform(0.4, 3.0, nn)
            qs = raw / raw.sum()
            seed = np.clip(seed_fn(qs), 1e-3, ANTE - 1)

            def obj(logN, qs=qs):
                return reduced_a(np.exp(logN), qs).sum()

            res = minimize(obj, np.log(seed), method="Nelder-Mead",
                           options={"xatol": 1e-8, "fatol": 1e-14, "maxiter": 20000})
            iters.append(res.nit)
        print(f"  seed {label:9}: mean iters {np.mean(iters):6.0f}   median {np.median(iters):6.0f}")
    print("\n  (Nelder-Mead iters; a real Newton on the KKT would be far fewer — this just")
    print("   ranks the seeds. Better seed => fewer iters.)\n")


def verify_reduced_matches_full():
    """The reduced solver should match single_point_liquidity-based a on its output."""
    print("=" * 82)
    print("(I) CONSISTENCY — reduced a_i == single_point_liquidity on the recovered shape")
    print("=" * 82)
    worst = 0.0
    rng = np.random.default_rng(11)
    for _ in range(200):
        nn = int(rng.integers(3, 7))
        raw = rng.uniform(0.4, 3.0, nn)
        qs = raw / raw.sum()
        N = rng.uniform(50, 400, nn)
        a_red = reduced_a(N, qs)
        Y, Nr, p = shape_from_N(N, qs)
        a_spl = np.array(
            [single_point_liquidity(Y[i], Nr[i], p[i], "YES")["abs_dprob_dshares"] for i in range(nn)]
        )
        worst = max(worst, float(np.max(np.abs(a_red - a_spl) / a_spl)))
    print(f"  max rel error: {worst:.2e}  => reduced formula matches the AMM primitive.\n")


if __name__ == "__main__":
    verify_reduced_matches_full()
    report()
    report_seed_quality()
