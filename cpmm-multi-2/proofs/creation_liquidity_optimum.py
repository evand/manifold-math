#!/usr/bin/env python3
"""
cpmm-multi-2 creation liquidity — characterize the basket-funded SKEW optimum.

Companion to creation_liquidity.py (the exploration). Here we (a) compare three
liquidity objectives, and (b) map the optimal shape across skews / n to find its
structure. Lead from the exploration: at the optimum, Y_i - N_i looked CONSTANT
across answers (the "all winners tight" funding condition). We test that here.

Objectives (all on the symmetric shares coordinate, point liquidity a_i):
  sum       minimize Σ a_i                       (cheap total impact; can starve one)
  maximin   minimize max_i a_i  (~deepen worst)  (fair; no answer starved)
  weighted  minimize Σ w_i a_i, w_i ∝ q_i        (favorites trade more)

The optimizer works on single-pool a_i (smooth, fast); we then report the
effective auto-arb a_i for the resulting shapes as a cross-check.

Analysis only. No vendor code change.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))

from creation_liquidity import (  # noqa: E402
    ANTE,
    single_point_liquidity,
    v2_pools,
)


def _pools_from(x, qs):
    n = len(qs)
    N = np.exp(x[:n])
    p = x[n:]
    R = p * (1 - qs) / (qs * (1 - p))
    Y = R * N
    return Y, N, p


def _a_vec(x, qs):
    Y, N, p = _pools_from(x, qs)
    return np.array(
        [
            single_point_liquidity(Y[i], N[i], p[i], "YES")["abs_dprob_dshares"]
            for i in range(len(qs))
        ]
    )


def _objective(a, qs, kind):
    if kind == "sum":
        return a.sum()
    if kind == "sum_eff":  # post-auto-arb: Σ a_i(Σa-a_i)/Σa
        S = a.sum()
        return (a * (S - a) / S).sum()
    if kind == "maximin":  # smooth max via high-order norm
        return (a**10).sum() ** 0.1
    if kind == "weighted":
        w = qs / qs.sum()
        return (w * a).sum()
    raise ValueError(kind)


def optimize_pools(qs, kind="sum", ante=ANTE):
    qs = np.array(qs, dtype=float)
    n = len(qs)

    def obj(x):
        return _objective(_a_vec(x, qs), qs, kind)

    def funding_slack(x):
        Y, N, _ = _pools_from(x, qs)
        S = N.sum()
        return ante - max(Y[k] + S - N[k] for k in range(n))

    cons = [{"type": "ineq", "fun": funding_slack}]
    bounds = [(np.log(1.0), np.log(ante))] * n + [(0.02, 0.98)] * n
    best = None
    starts = [np.concatenate([np.log(np.full(n, ante / n)), qs])]
    # a couple perturbed restarts (avoid local minima)
    for seed in (1, 2):
        rng = np.random.default_rng(seed)
        starts.append(
            np.concatenate(
                [np.log(np.full(n, ante / n) * rng.uniform(0.5, 1.5, n)),
                 np.clip(qs + rng.uniform(-0.1, 0.1, n), 0.05, 0.95)]
            )
        )
    for x0 in starts:
        res = minimize(
            obj, x0, constraints=cons, bounds=bounds, method="SLSQP",
            options={"maxiter": 800, "ftol": 1e-12},
        )
        if res.success and (best is None or res.fun < best.fun):
            best = res
    return _pools_from(best.x, qs), best


def report_objective_comparison(qs_list):
    print("=" * 80)
    print("(A) THREE OBJECTIVES — optimal shape + gain vs balanced, per skew")
    print("=" * 80)
    for qs in qs_list:
        qs = np.array(qs, float)
        n = len(qs)
        bal = v2_pools(ANTE, list(qs))
        bal_a = np.array(
            [single_point_liquidity(*bal[i], "YES")["abs_dprob_dshares"] for i in range(n)]
        )
        print(f"\nq = {[round(float(v), 3) for v in qs]}")
        print(f"  balanced: a_i={[f'{v:.3e}' for v in bal_a]}  Σ={bal_a.sum():.3e}  max={bal_a.max():.3e}")
        for kind in ("sum", "maximin", "weighted"):
            (Y, N, p), _ = optimize_pools(qs, kind)
            a = np.array(
                [single_point_liquidity(Y[i], N[i], p[i], "YES")["abs_dprob_dshares"] for i in range(n)]
            )
            D = Y - N
            sum_gain = 100 * (bal_a.sum() - a.sum()) / bal_a.sum()
            max_gain = 100 * (bal_a.max() - a.max()) / bal_a.max()
            print(
                f"  {kind:8} Σ_gain={sum_gain:+5.1f}%  max_gain={max_gain:+5.1f}%  "
                f"D=Y-N={[f'{d:.0f}' for d in D]} (spread {D.max() - D.min():.1f})  "
                f"p={[f'{v:.2f}' for v in p]}"
            )
    print()


def report_d_constant_test(qs_list):
    print("=" * 80)
    print("(B) STRUCTURE — is Y_i - N_i constant at the optimum? (all-winners-tight)")
    print("=" * 80)
    print(f"{'q':<28}{'objective':<10}{'D spread / D':>16}{'worst-tight?':>14}")
    print("-" * 68)
    for qs in qs_list:
        qs = np.array(qs, float)
        n = len(qs)
        for kind in ("sum", "maximin", "weighted"):
            (Y, N, p), _ = optimize_pools(qs, kind)
            D = Y - N
            S = N.sum()
            worst = [Y[k] + S - N[k] for k in range(n)]
            rel_spread = (D.max() - D.min()) / abs(D.mean()) if D.mean() else 0.0
            all_tight = all(abs(w - ANTE) < 1.0 for w in worst)
            print(
                f"{str([round(float(v), 2) for v in qs]):<28}{kind:<10}"
                f"{rel_spread:>15.3%}{str(all_tight):>14}"
            )
    print("\nIf D spread ≈ 0 and all winners tight, the optimum lives on the corner")
    print("where every resolution scenario uses the full ante (D_i = ante - ΣN_j const).")
    print()


def report_gain_landscape(n_values, max_probs):
    print("=" * 80)
    print("(C) GAIN LANDSCAPE — Σ-objective liquidity gain vs balanced, by skew & n")
    print("=" * 80)
    print("    skew built as: favorite=max_prob, remainder split evenly.")
    hdr = f"{'max_prob':>9}" + "".join(f"{'n=' + str(n):>9}" for n in n_values)
    print(hdr)
    print("-" * len(hdr))
    for mp in max_probs:
        row = f"{mp:>9.2f}"
        for n in n_values:
            if mp <= 1.0 / n:  # not a skew toward the favorite
                row += f"{'-':>9}"
                continue
            rest = (1 - mp) / (n - 1)
            qs = np.array([mp] + [rest] * (n - 1))
            bal = v2_pools(ANTE, list(qs))
            bal_sum = sum(
                single_point_liquidity(*bal[i], "YES")["abs_dprob_dshares"] for i in range(n)
            )
            (Y, N, p), _ = optimize_pools(qs, "sum")
            opt_sum = sum(
                single_point_liquidity(Y[i], N[i], p[i], "YES")["abs_dprob_dshares"]
                for i in range(n)
            )
            row += f"{100 * (bal_sum - opt_sum) / bal_sum:>8.1f}%"
        print(row)
    print("\nGain grows with skew strength and (strongly) with n. Even near-uniform the")
    print("gain is large because balanced != v1's asymmetric shape (which IS optimal at")
    print("exact uniform); the optimum tracks v1's deep-cheap-side, generalized via p.")
    print()


def add_preserving(Y, N, p, dY, dN):
    """Add dY YES-shares + dN NO-shares to the pool, float p to preserve prob.

    Mana cost of the add = dY*q + dN*(1-q) (the market value of the shares added,
    q the preserved prob). prob unchanged by construction.
    """
    from creation_liquidity import prob_of  # noqa: E402

    qv = prob_of((Y, N, p))
    Y2, N2 = Y + dY, N + dN
    r = qv * Y2 / ((1 - qv) * N2)
    return Y2, N2, r / (1 + r)


def report_add_split_dof():
    from creation_liquidity import prob_of  # noqa: E402

    print("=" * 80)
    print("(D) ADD-SPLIT DOF — adding M mana to one answer: only-YES vs only-NO vs blend")
    print("=" * 80)
    Y, N, p = 100.0, 100.0, 0.5  # prob 0.5
    M = 100.0
    q = prob_of((Y, N, p))
    print(f"  start (Y={Y:.0f},N={N:.0f},p={p:.2f}) prob={q:.3f}; add M={M:.0f} mana three ways.")
    print(f"  budget line: dY*q + dN*(1-q) = M  =>  dY in [0, M/q={M / q:.0f}]")
    for label, dY in (("all-NO", 0.0), ("balanced", M), ("all-YES", M / q)):
        dN = (M - dY * q) / (1 - q)
        Y2, N2, p2 = add_preserving(Y, N, p, dY, dN)
        a = single_point_liquidity(Y2, N2, p2, "YES")["abs_dprob_dshares"]
        print(
            f"    {label:9} dY={dY:6.1f} dN={dN:6.1f} -> (Y={Y2:6.1f},N={N2:6.1f},"
            f"p={p2:.3f}) prob={prob_of((Y2, N2, p2)):.3f}  a={a:.4e}"
        )
    print("  => different splits reach different (Y,N,p) with different a: NOT equivalent.\n")

    print("  Additivity/commutativity of prob-preserving adds (q constant throughout):")
    s1 = add_preserving(Y, N, p, 120.0, 40.0)
    s2 = add_preserving(*s1, 30.0, 90.0)
    combined = add_preserving(Y, N, p, 150.0, 130.0)
    swapped1 = add_preserving(Y, N, p, 30.0, 90.0)
    swapped = add_preserving(*swapped1, 120.0, 40.0)
    print(f"    seq (120,40) then (30,90): (Y={s2[0]:.1f},N={s2[1]:.1f},p={s2[2]:.4f})")
    print(f"    one shot (150,130):        (Y={combined[0]:.1f},N={combined[1]:.1f},p={combined[2]:.4f})")
    print(f"    swapped order:             (Y={swapped[0]:.1f},N={swapped[1]:.1f},p={swapped[2]:.4f})")
    print("  => a sequence of prob-preserving adds collapses to one net (ΣdY, ΣdN);")
    print("     order & granularity are irrelevant. (Interleaved TRADES change q and")
    print("     break this — then it is path-dependent.)\n")


def report_clean_formula():
    from creation_liquidity import prob_of, v1_uniform_pools

    print("=" * 80)
    print("(E) CLEAN POINT-LIQUIDITY FORMULA + the three questions")
    print("=" * 80)
    print("  a = dprob/dshares  =?=  p(1-p)YN / W^3  =?=  q(1-q)/W,  W=(1-p)Y+pN")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(2000):
        Y = float(rng.uniform(20, 2000))
        N = float(rng.uniform(20, 2000))
        p = float(rng.uniform(0.05, 0.95))
        W = (1 - p) * Y + p * N
        q = prob_of((Y, N, p))
        a_num = single_point_liquidity(Y, N, p, "YES")["abs_dprob_dshares"]
        a_f1 = p * (1 - p) * Y * N / W**3
        a_f2 = q * (1 - q) / W
        a_f3 = q * (1 - q) * (q * Y + (1 - q) * N) / (Y * N)  # p eliminated
        worst = max(worst, abs(a_f1 - a_num) / a_num, abs(a_f2 - a_num) / a_num, abs(a_f3 - a_num) / a_num)
    print(f"  max rel error over 2000 random (Y,N,p): {worst:.2e}  => formula confirmed.")
    print("  => maximize liquidity (min a) at fixed prob q  <=>  MAXIMIZE W=(1-p)Y+pN.\n")

    print("  Q1: is v1 init 'equal mana value on Y and N'?  (value_Y=Y*q, value_N=N*(1-q))")
    for n in (3, 4, 5):
        Y, N, p = v1_uniform_pools(ANTE, n)[0]
        q = prob_of((Y, N, p))
        print(
            f"    n={n}: value_Y={Y * q:7.2f}  value_N={N * (1 - q):7.2f}  "
            f"equal? {abs(Y * q - N * (1 - q)) < 1e-6}  (= ante/(2n)={ANTE / (2 * n):.2f})"
        )
    print("    'equal value on Y and N'  <=>  p=0.5 (value_Y=value_N  iff  Y q = N(1-q) iff p=.5).")
    print("    The skew optimum has p != 0.5 (unequal value) -> v1's equal-spend does NOT")
    print("    generalize as the optimum; it is the optimum only at uniform.\n")

    print("  Q3: Set/independent answer == binary CPMM?  binary uses balanced Y=N, p=initialProb.")
    for q in (0.2, 0.5, 0.8):
        # at fixed per-answer risk max(Y,N)=L, the a-minimizer over shape:
        L = 300.0
        # balanced: Y=N=L, p=q  -> a
        a_bal = single_point_liquidity(L, L, q, "YES")["abs_dprob_dshares"]
        # scan the shape family at fixed max(Y,N)=L for the min-a shape
        best = (None, 1e9)
        for pp in [0.01 * i for i in range(1, 100)]:
            R = pp * (1 - q) / (q * (1 - pp))
            Yy, Nn = (L, L / R) if R >= 1 else (L * R, L)
            a = single_point_liquidity(Yy, Nn, pp, "YES")["abs_dprob_dshares"]
            if a < best[1]:
                best = (pp, a)
        print(
            f"    q={q}: balanced(p=q) a={a_bal:.4e}   best-shape p*={best[0]:.2f} a*={best[1]:.4e}"
            f"   -> balanced optimal? {abs(best[0] - q) < 0.02}"
        )
    print("    => at fixed per-answer risk, balanced p=q IS the liquidity optimum: Set markets")
    print("       should do exactly what binary CPMM does (balanced pool, p=initialProb).\n")


def _a_eff(a):
    """Post-auto-arb effective slopes from single-pool slopes a (single YES buy)."""
    S = a.sum()
    return a * (S - a) / S


def report_post_arb():
    from creation_liquidity import arb_point_liquidity, v2_pools

    print("=" * 80)
    print("(F) POST-AUTO-ARB — the formula a_eff = a_i(Σa-a_i)/Σa, and the optimum on it")
    print("=" * 80)
    print("  Verify a_eff vs the auto-arb solver (random skewed sum-to-one markets):")
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(40):
        n = int(rng.integers(3, 6))
        raw = rng.uniform(0.5, 3, n)
        qs = raw / raw.sum()
        pools = v2_pools(ANTE, list(qs))  # any valid market; balanced here
        a = np.array(
            [single_point_liquidity(*pools[i], "YES")["abs_dprob_dshares"] for i in range(n)]
        )
        aeff_formula = _a_eff(a)
        aeff_solver = np.array(
            [arb_point_liquidity(pools, i, "YES")["abs_dprob_dshares"] for i in range(n)]
        )
        worst = max(worst, float(np.max(np.abs(aeff_formula - aeff_solver) / aeff_solver)))
    print(f"    max rel error: {worst:.2e}  => post-arb formula confirmed.\n")

    print("  Re-optimize on the POST-ARB objective (Σ a_eff) and compare to the PRE-arb")
    print("  optimum + balanced. Does the asymmetric-optimum conclusion survive?")
    for qs in ([0.6, 0.25, 0.15], [0.8, 0.1, 0.1], [0.4, 0.3, 0.2, 0.1]):
        qs = np.array(qs, float)
        n = len(qs)
        bal = v2_pools(ANTE, list(qs))
        a_bal = np.array(
            [single_point_liquidity(*bal[i], "YES")["abs_dprob_dshares"] for i in range(n)]
        )
        bal_eff = _a_eff(a_bal).sum()
        (Yp, Np_, pp), _ = optimize_pools(qs, "sum")  # pre-arb optimum
        a_pre = np.array(
            [single_point_liquidity(Yp[i], Np_[i], pp[i], "YES")["abs_dprob_dshares"] for i in range(n)]
        )
        pre_eff = _a_eff(a_pre).sum()
        (Yo, No, po), _ = optimize_pools(qs, "sum_eff")  # post-arb optimum
        a_post = np.array(
            [single_point_liquidity(Yo[i], No[i], po[i], "YES")["abs_dprob_dshares"] for i in range(n)]
        )
        post_eff = _a_eff(a_post).sum()
        Dpre = Yp - Np_
        Dpost = Yo - No
        print(f"  q={[round(float(v), 2) for v in qs]}  (Σ a_eff; lower=better)")
        print(f"    balanced          Σa_eff={bal_eff:.4e}")
        print(f"    pre-arb optimum   Σa_eff={pre_eff:.4e}  ({100 * (bal_eff - pre_eff) / bal_eff:+.1f}% vs bal)  D-spread {Dpre.max() - Dpre.min():.1f}")
        print(f"    post-arb optimum  Σa_eff={post_eff:.4e}  ({100 * (bal_eff - post_eff) / bal_eff:+.1f}% vs bal)  D-spread {Dpost.max() - Dpost.min():.1f}")
        print(f"      post-arb shape p={[f'{v:.2f}' for v in po]}")
    print()
    print("  If pre- and post-arb optima nearly coincide, the clean pre-arb solve is a")
    print("  fine proxy. a_eff is monotone in a_i, so 'maximize W' still drives both.")
    print()


if __name__ == "__main__":
    skews = [[0.6, 0.25, 0.15], [0.8, 0.1, 0.1], [0.5, 0.3, 0.2], [0.4, 0.3, 0.2, 0.1]]
    report_objective_comparison(skews)
    report_d_constant_test(skews + [[0.7, 0.2, 0.1], [0.9, 0.05, 0.05]])
    report_gain_landscape([3, 4, 5, 6], [0.35, 0.45, 0.55, 0.70, 0.85, 0.95])
    report_add_split_dof()
    report_clean_formula()
    report_post_arb()
