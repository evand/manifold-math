#!/usr/bin/env python3
"""
cpmm-multi-2 creation liquidity — how should a v2 market choose its creation pools?

OPEN design question (onboarding.md "DEFERRED MATH"). v1 and v2 fund the SAME ante
worst-case payout but with different pool *shapes*:

  v1 (uniform only):  poolYes = ante/2,  poolNo = ante/(2n-2),  p = 0.5   -> prob = 1/n
  v2 (any target):    poolYes = poolNo = ante/n,  p = target            -> prob = p

The static sqrt(Y*N) "liquidity" metric says v1 is deeper for n>=3, but that is a
*directional* artifact. The honest yardsticks (Evan, 2026-06-28):

  (A) point liquidity = differential prob movement per unit traded, AT the created
      prob. SHARES is the YES/NO-symmetric coordinate (this script confirms it);
      mana is not.
  (B) the full curve of spend (shares and mana) to move a single answer to a target
      prob.

There is no p!=0.5 oracle, so the validation is internal consistency:
  - the p=0.5 point-liquidity closed form a = 2YN/(Y+N)^3 must match finite diff;
  - YES/NO symmetry in the shares coordinate under (Y,N,p)->(N,Y,1-p);
  - funding worst-case payout == ante for both shapes.

Analysis only. No vendor code changes.
"""

from __future__ import annotations

import math
import os
import sys

# This sim uses our amm_core / market_simulator as the auto-arb oracle (no
# standalone p!=0.5 oracle exists). Add the package root so it runs from anywhere.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "manifold"))
)

from manifold.amm_core import (
    pool_after_trade,
    probability_from_pool,
    shares_for_cost,
)
from manifold.market_simulator import MarketSimulator

ANTE = 1000.0


# --------------------------------------------------------------------------- #
# Creation shapes (mirror vendor common/src/new-contract.ts createAnswers)
# --------------------------------------------------------------------------- #
def v1_uniform_pools(ante: float, n: int) -> list[tuple[float, float, float]]:
    """v1 sum-to-one uniform: asymmetric pool, p=0.5, prob=1/n."""
    poolYes = ante / 2
    poolNo = ante / (2 * n - 2) if n > 1 else ante
    return [(poolYes, poolNo, 0.5) for _ in range(n)]


def v2_pools(ante: float, probs: list[float]) -> list[tuple[float, float, float]]:
    """v2 balanced: Y=N=ante/n, p=target prob (prob=p exactly when balanced)."""
    n = len(probs)
    L = ante / n
    return [(L, L, q) for q in probs]


def prob_of(pool: tuple[float, float, float]) -> float:
    Y, N, p = pool
    return probability_from_pool(Y, N, p)


def worst_case_payout(pools: list[tuple[float, float, float]]) -> float:
    """Max mana the AMM owes if any single answer resolves YES (sum-to-one)."""
    out = []
    for i, (Yi, _Ni, _pi) in enumerate(pools):
        pay = Yi + sum(Nj for j, (_Yj, Nj, _pj) in enumerate(pools) if j != i)
        out.append(pay)
    return max(out)


# --------------------------------------------------------------------------- #
# Single-answer point liquidity (no auto-arb): pure pool-shape effect
# --------------------------------------------------------------------------- #
def single_point_liquidity(
    Y: float, N: float, p: float, position: str, dm: float = 1e-4
) -> dict:
    """dprob/dshares and dprob/dmana for an infinitesimal trade on one pool.

    dprob is the signed change in P(YES). For a NO buy P(YES) falls, so dprob<0;
    we report |dprob/dshares| as the magnitude of point liquidity.
    """
    prob0 = probability_from_pool(Y, N, p)
    shares = shares_for_cost(Y, N, dm, position, p=p)
    Y2, N2 = pool_after_trade(Y, N, dm, position, p=p)
    prob1 = probability_from_pool(Y2, N2, p)
    dprob = prob1 - prob0
    return {
        "dprob_dshares": dprob / shares,
        "dprob_dmana": dprob / dm,
        "abs_dprob_dshares": abs(dprob / shares),
        "abs_dprob_dmana": abs(dprob / dm),
    }


def a_closed_form_half(Y: float, N: float) -> float:
    """p=1/2 point liquidity in shares: a = 2YN/(Y+N)^3 (proofs GP, symmetric)."""
    return 2 * Y * N / (Y + N) ** 3


# --------------------------------------------------------------------------- #
# Market construction for the with-arb (sum-to-one) oracle
# --------------------------------------------------------------------------- #
def build_market(
    pools: list[tuple[float, float, float]], sum_to_one: bool = True
) -> dict:
    answers = []
    for i, (Y, N, p) in enumerate(pools):
        pr = probability_from_pool(Y, N, p)
        answers.append(
            {
                "id": f"a{i}",
                "text": f"A{i}",
                "poolYes": Y,
                "poolNo": N,
                "poolYES": Y,
                "poolNO": N,
                "p": p,
                "prob": pr,
                "probability": pr,
            }
        )
    return {
        "id": "m",
        "question": "q",
        "outcomeType": "MULTIPLE_CHOICE",
        "mechanism": "cpmm-multi-1",  # per-answer p prototype rides on multi-1
        "shouldAnswersSumToOne": sum_to_one,
        "answers": answers,
    }


def _sim(pools, sum_to_one=True) -> MarketSimulator:
    return MarketSimulator(
        build_market(pools, sum_to_one), _internal=True, _validate=False
    )


def arb_point_liquidity(
    pools: list[tuple[float, float, float]], idx: int, position: str, dm: float = 1e-2
) -> dict:
    """Effective dprob_i/dshares INCLUDING auto-arb (what a trader actually sees)."""
    sim = _sim(pools)
    aid = f"a{idx}"
    p0 = sim.get_probability(aid)
    res = sim.simulate_buy(aid, dm, position)
    p1 = sim.get_probability(aid)
    shares = res["shares"]
    dprob = p1 - p0
    return {
        "dprob_dshares": dprob / shares,
        "dprob_dmana": dprob / dm,
        "abs_dprob_dshares": abs(dprob / shares),
        "abs_dprob_dmana": abs(dprob / dm),
    }


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def report_symmetry_check() -> None:
    print("=" * 78)
    print("(0) VALIDATION — shares is the YES/NO-symmetric coordinate, mana is not")
    print("=" * 78)
    print("Single pool (Y=300, N=100, p=0.5). |dprob/d.| for a YES buy vs a NO buy:")
    Y, N, p = 300.0, 100.0, 0.5
    yes = single_point_liquidity(Y, N, p, "YES")
    no = single_point_liquidity(Y, N, p, "NO")
    print(
        f"  shares: YES {yes['abs_dprob_dshares']:.6e}  "
        f"NO {no['abs_dprob_dshares']:.6e}  "
        f"equal? {math.isclose(yes['abs_dprob_dshares'], no['abs_dprob_dshares'], rel_tol=1e-3)}"
    )
    print(
        f"  mana:   YES {yes['abs_dprob_dmana']:.6e}  "
        f"NO {no['abs_dprob_dmana']:.6e}  "
        f"equal? {math.isclose(yes['abs_dprob_dmana'], no['abs_dprob_dmana'], rel_tol=1e-3)}"
    )
    cf = a_closed_form_half(Y, N)
    print(
        f"  closed form a=2YN/(Y+N)^3 = {cf:.6e}  vs finite-diff shares "
        f"{yes['abs_dprob_dshares']:.6e}  match? "
        f"{math.isclose(cf, yes['abs_dprob_dshares'], rel_tol=1e-3)}"
    )
    print()


def report_uniform(n_values: list[int]) -> None:
    print("=" * 78)
    print(f"(1) UNIFORM INIT — v1 vs v2 at prob=1/n, ante={ANTE:.0f}")
    print("=" * 78)
    hdr = (
        f"{'n':>2} {'shape':<4} {'maxYN':>8} {'sqrtYN':>9} {'fund':>7} "
        f"{'a_single':>11} {'a_arb_YES':>11} {'a_arb_NO':>11} {'mana_arbY':>11}"
    )
    print(hdr)
    print("-" * len(hdr))
    for n in n_values:
        for label, pools in (
            ("v1", v1_uniform_pools(ANTE, n)),
            ("v2", v2_pools(ANTE, [1.0 / n] * n)),
        ):
            Y, N, p = pools[0]
            sp = single_point_liquidity(Y, N, p, "YES")
            ay = arb_point_liquidity(pools, 0, "YES")
            an = arb_point_liquidity(pools, 0, "NO")
            print(
                f"{n:>2} {label:<4} {max(Y, N):>8.1f} {math.sqrt(Y * N):>9.2f} "
                f"{worst_case_payout(pools):>7.0f} "
                f"{sp['abs_dprob_dshares']:>11.4e} "
                f"{ay['abs_dprob_dshares']:>11.4e} {an['abs_dprob_dshares']:>11.4e} "
                f"{ay['abs_dprob_dmana']:>11.4e}"
            )
        print()
    print("Lower a = more liquid (prob moves less per share/mana). a_single = no-arb")
    print("single-pool slope; a_arb = effective slope incl. auto-arb.")
    print()


def materialize_v1_traded(n: int, q0: float) -> list[tuple[float, float, float]]:
    """v1 uniform market traded so answer 0 sits at P(YES)=q0; return its pools."""
    sim = _sim(v1_uniform_pools(ANTE, n))
    start = 1.0 / n
    if q0 >= start:
        r = sim.buy_to_probability("a0", q0, "YES")
        sim.simulate_buy("a0", r["cost"], "YES")
    else:
        r = sim.buy_to_probability("a0", 1 - q0, "NO")
        sim.simulate_buy("a0", r["cost"], "NO")
    return [(a["poolYes"], a["poolNo"], a["p"]) for a in sim.data["answers"]]


def v2_skew(n: int, q0: float) -> list[tuple[float, float, float]]:
    """v2 minted directly at answer-0 prob q0, remainder split evenly."""
    rest = (1 - q0) / (n - 1)
    return v2_pools(ANTE, [q0] + [rest] * (n - 1))


def report_matched_skew(n: int, q0_grid: list[float]) -> None:
    print("=" * 78)
    print(
        f"(2) MATCHED-STATE SKEW (n={n}) — liquidity AT answer-0 prob q0:\n"
        f"    v2 minted at q0  vs  v1 minted uniform then TRADED to q0"
    )
    print("=" * 78)
    hdr = (
        f"{'q0':>5} {'shape':<9} {'a0_prob':>8} {'aYES_sh':>10} {'aNO_sh':>10} "
        f"{'manaY':>10} {'manaN':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for q0 in q0_grid:
        for label, pools in (
            ("v2@q0", v2_skew(n, q0)),
            ("v1->q0", materialize_v1_traded(n, q0)),
        ):
            ay = arb_point_liquidity(pools, 0, "YES")
            an = arb_point_liquidity(pools, 0, "NO")
            actual = prob_of(pools[0])
            print(
                f"{q0:>5.2f} {label:<9} {actual:>8.4f} "
                f"{ay['abs_dprob_dshares']:>10.4e} {an['abs_dprob_dshares']:>10.4e} "
                f"{ay['abs_dprob_dmana']:>10.4e} {an['abs_dprob_dmana']:>10.4e}"
            )
        print()
    print("Lower = more liquid. aYES_sh/aNO_sh: shares coord (symmetric pool metric);")
    print("manaY/manaN: per-mana cost to push up / down (the trader's actual cost).")
    print()


def report_spend_curve(n: int, targets: list[float]) -> None:
    print("=" * 78)
    print(
        f"(3) FULL SPEND CURVE (n={n}) — cumulative cost to move answer 0 to a target\n"
        f"    P(YES), from three creation shapes. mana = M$ spent; sh = shares bought."
    )
    print("=" * 78)
    shapes = {
        "v1-unif": v1_uniform_pools(ANTE, n),
        "v2-unif": v2_pools(ANTE, [1.0 / n] * n),
        "v2@0.60": v2_skew(n, 0.60),
    }
    hdr = f"{'target':>7} " + " ".join(
        f"{name + '_mana':>13} {name + '_sh':>11}" for name in shapes
    )
    print(hdr)
    print("-" * len(hdr))
    for t in targets:
        row = f"{t:>7.2f} "
        for name, pools in shapes.items():
            start = prob_of(pools[0])
            sim = _sim(pools)
            try:
                if t >= start:
                    r = sim.buy_to_probability("a0", t, "YES")
                else:
                    r = sim.buy_to_probability("a0", 1 - t, "NO")
                row += f"{r['cost']:>13.2f} {r['shares']:>11.2f} "
            except ValueError:
                row += f"{'n/a':>13} {'n/a':>11} "
        print(row)
    print()
    print("Cost is cumulative from each shape's OWN creation prob (so v2@0.60 starts")
    print("at 0). Compare slopes where the curves overlap to read relative liquidity.")
    print()


def shape_for_prob_and_risk(q: float, p: float, L: float) -> tuple[float, float, float]:
    """The (Y,N,p) with P(YES)=q at the given p, scaled so max(Y,N)=L.

    prob = p·N/((1-p)·Y + p·N)  =>  Y/N = p(1-q) / (q(1-p)).
    One free shape param (p); p=q gives the balanced pool (Y=N=L).
    """
    R = p * (1 - q) / (q * (1 - p))  # Y/N
    if R >= 1:
        Y, N = L, L / R
    else:
        Y, N = L * R, L
    return (Y, N, p)


def report_fixed_risk_family(L: float, q_grid: list[float]) -> None:
    print("=" * 78)
    print(
        f"(4) SHAPE FAMILY AT FIXED PER-ANSWER RISK (max(Y,N)=L={L:.0f}) — single answer\n"
        f"    All shapes below price P(YES)=q; they differ only in p (the shape DOF)."
    )
    print("=" * 78)
    hdr = (
        f"{'q':>5} {'shape':<10} {'p':>5} {'Y':>7} {'N':>7} {'sqrtYN':>8} "
        f"{'a_shares':>11} {'best_p':>7} {'a_best':>11}"
    )
    print(hdr)
    print("-" * len(hdr))
    ps = [0.01 * i for i in range(1, 100)]
    for q in q_grid:
        # liquidity-optimal p over the family
        best_p, best_a = None, math.inf
        for p in ps:
            Y, N, _ = shape_for_prob_and_risk(q, p, L)
            a = single_point_liquidity(Y, N, p, "YES")["abs_dprob_dshares"]
            if a < best_a:
                best_a, best_p = a, p
        for label, p in (("balanced", q), ("asym p=.5", 0.5)):
            Y, N, pp = shape_for_prob_and_risk(q, p, L)
            a = single_point_liquidity(Y, N, pp, "YES")["abs_dprob_dshares"]
            print(
                f"{q:>5.2f} {label:<10} {p:>5.2f} {Y:>7.1f} {N:>7.1f} "
                f"{math.sqrt(Y * N):>8.1f} {a:>11.4e} {best_p:>7.2f} {best_a:>11.4e}"
            )
        print()
    print("At fixed max-loss L, lower a = more liquid. best_p = the p in (0,1) that")
    print("minimizes a over the whole shape family (is it p=q, i.e. balanced?).")
    print()


def report_funding_lossless(n: int, q0: float) -> None:
    print("=" * 78)
    print(
        f"(5) FUNDING + LOSSLESSNESS at a skew (n={n}, answer0={q0}) — why shape matters"
    )
    print("=" * 78)
    # v2 balanced at the skew:
    bal = v2_skew(n, q0)
    # asym p=0.5 minted at the same skew, each answer scaled to max(Y,N)=ante/n:
    L = ANTE / n
    probs = [q0] + [(1 - q0) / (n - 1)] * (n - 1)
    asym = [shape_for_prob_and_risk(q, 0.5, L) for q in probs]
    for label, pools in (("v2 balanced", bal), ("asym p=.5 (max=ante/n)", asym)):
        maxes = [max(Y, N) for (Y, N, _p) in pools]
        print(
            f"  {label:<24} maxYN/ans={[f'{m:.0f}' for m in maxes]}  "
            f"worst-case payout={worst_case_payout(pools):.0f} (ante={ANTE:.0f})"
        )
    print()
    print("  Balanced keeps max(Y,N)=ante/n on EVERY answer (risk flat, fully funded).")
    print("  Asym p=.5 inflates the low-prob answers' YES reserve -> basket worst case")
    print("  overshoots ante (house must over-fund) or, capped at ante, discards shares.")
    print()


def basket_shape(q: float, p: float, others_N_sum: float, ante: float):
    """For one answer at prob q with shape p, the (Y,N) scaled so this answer's
    basket worst case Y + others_N_sum == ante. ratio Y/N = p(1-q)/(q(1-p))."""
    R = p * (1 - q) / (q * (1 - p))
    # Y + others_N_sum = ante  and  Y = R*N  =>  but N also enters others for
    # other answers; for the symmetric uniform case we solve jointly below.
    return R


def report_basket_uniform_family(n_values: list[int]) -> None:
    print("=" * 78)
    print(
        "(6) BASKET-FUNDED UNIFORM FAMILY — sweep p, prob=1/n, funding Y+(n-1)N=ante.\n"
        "    Is v1's p=0.5 the liquidity-optimal shape, or can v2 beat it?"
    )
    print("=" * 78)
    hdr = (
        f"{'n':>2} {'p*':>6} {'a*_single':>11} {'a(p=.5)=v1':>12} "
        f"{'a*_arb':>10} {'arb(p=.5)':>10} {'gain%':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for n in n_values:

        def shape(p, n=n):
            R = (n - 1) * p / (1 - p)  # Y/N for prob=1/n at this p
            N = ANTE / (R + n - 1)
            Y = R * N
            return Y, N, p

        best_p, best_a = None, math.inf
        for i in range(1, 1000):
            p = i / 1000
            Y, N, _ = shape(p)
            a = single_point_liquidity(Y, N, p, "YES")["abs_dprob_dshares"]
            if a < best_a:
                best_a, best_p = a, p
        # v1 baseline (p=0.5) single + arb; optimal arb
        Yv1, Nv1, _ = shape(0.5)
        a_v1 = single_point_liquidity(Yv1, Nv1, 0.5, "YES")["abs_dprob_dshares"]
        arb_v1 = arb_point_liquidity([shape(0.5)] * n, 0, "YES")["abs_dprob_dshares"]
        arb_best = arb_point_liquidity([shape(best_p)] * n, 0, "YES")[
            "abs_dprob_dshares"
        ]
        gain = 100 * (arb_v1 - arb_best) / arb_v1
        print(
            f"{n:>2} {best_p:>6.3f} {best_a:>11.4e} {a_v1:>12.4e} "
            f"{arb_best:>10.4e} {arb_v1:>10.4e} {gain:>7.2f}"
        )
    print()
    print("p* = liquidity-optimal shape under basket funding. If p*≈0.5, v1 is already")
    print("optimal; if p* differs, v2's p DOF beats v1. gain% = arb-liquidity improvement.")
    print()


def float_p_add(Y: float, N: float, p: float, a: float) -> tuple[float, float, float]:
    """Lossless add: inject `a` into both reserves, float p to hold prob (GP6a)."""
    q = probability_from_pool(Y, N, p)
    Y2, N2 = Y + a, N + a
    r = q * Y2 / ((1 - q) * N2)  # p2/(1-p2)
    return Y2, N2, r / (1 + r)


def report_add_funding(qs: list[float]) -> None:
    print("=" * 78)
    print(
        "(8) DOES A LOSSLESS ADD KEEP AN *ASYMMETRIC* POOL FUNDED? (the load-bearing\n"
        "    claim was 'balanced required'). Take a deep/asym sum-to-one market, add."
    )
    print("=" * 78)
    # an asymmetric (deep-YES) funded market at the skew
    n = len(qs)
    L = ANTE / n
    pools = [shape_for_prob_and_risk(q, min(0.9, q + 0.25), 1.5 * L) for q in qs]

    def worst(pls):
        Sn = sum(N for _Y, N, _p in pls)
        return max(Y + Sn - N for Y, N, _p in pls)

    scale = ANTE / worst(pools)
    pools = [(Y * scale, N * scale, p) for Y, N, p in pools]
    a = 100.0  # add 100 to both reserves of each answer
    added = [float_p_add(Y, N, p, a) for Y, N, p in pools]
    print(f"  before: worst-case={worst(pools):.1f}  probs={[round(prob_of(x),3) for x in pools]}")
    print(f"  +{a:.0f}/reserve: worst-case={worst(added):.1f}  probs={[round(prob_of(x),3) for x in added]}")
    print(f"  worst-case grew by {worst(added) - worst(pools):.1f} (= n*a = {n * a:.0f}); probs unchanged.")
    print("  => lossless add preserves funding for ANY shape; balanced is NOT required.\n")


def report_skew_optimization(qs_list: list[list[float]]) -> None:
    import numpy as np
    from scipy.optimize import minimize

    print("=" * 78)
    print(
        "(7) BASKET-FUNDED SKEW OPTIMUM — maximize liquidity (min Σ a_i, shares)\n"
        "    over (N_i, p_i) s.t. prob_i=q_i and basket worst-case ≤ ante."
    )
    print("=" * 78)
    for qs in qs_list:
        qs = np.array(qs, dtype=float)
        n = len(qs)

        def unpack(x):
            N = np.exp(x[:n])
            p = x[n:]
            R = p * (1 - qs) / (qs * (1 - p))
            Y = R * N
            return Y, N, p

        def total_a(x):
            Y, N, p = unpack(x)
            return sum(
                single_point_liquidity(Y[i], N[i], p[i], "YES")["abs_dprob_dshares"]
                for i in range(n)
            )

        def funding_slack(x):
            Y, N, _ = unpack(x)
            S = N.sum()
            worst = max(Y[k] + S - N[k] for k in range(n))
            return ANTE - worst

        cons = [{"type": "ineq", "fun": funding_slack}]
        bounds = [(math.log(1.0), math.log(ANTE))] * n + [(0.02, 0.98)] * n
        x0 = np.concatenate([np.log(np.full(n, ANTE / n)), qs])
        res = minimize(
            total_a, x0, constraints=cons, bounds=bounds, method="SLSQP",
            options={"maxiter": 500, "ftol": 1e-12},
        )
        Yb, Nb, pb = unpack(x0)
        Yo, No, po = unpack(res.x)
        bal_pools = [(Yb[i], Nb[i], pb[i]) for i in range(n)]
        opt_pools = [(Yo[i], No[i], po[i]) for i in range(n)]
        print(f"q = {[round(float(v), 3) for v in qs]}   (ante={ANTE:.0f})")
        print(f"  {'':14}{'Σa_single':>12}{'worst-pay':>11}")
        print(
            f"  {'balanced':14}{total_a(x0):>12.4e}{ANTE - funding_slack(x0):>11.0f}"
        )
        print(
            f"  {'optimum':14}{total_a(res.x):>12.4e}{ANTE - funding_slack(res.x):>11.0f}"
            f"   ({100 * (total_a(x0) - total_a(res.x)) / total_a(x0):.1f}% more liquid)"
        )
        for i in range(n):
            print(
                f"    ans{i} q={qs[i]:.2f}  balanced(Y={Yb[i]:.0f},N={Nb[i]:.0f},p={pb[i]:.2f})"
                f"  -> optimum(Y={Yo[i]:.0f},N={No[i]:.0f},p={po[i]:.3f})"
            )
        # verify with effective arb liquidity
        arb_bal = [arb_point_liquidity(bal_pools, i, "YES")["abs_dprob_dshares"] for i in range(n)]
        arb_opt = [arb_point_liquidity(opt_pools, i, "YES")["abs_dprob_dshares"] for i in range(n)]
        print(f"    arb a_i balanced: {[f'{a:.3e}' for a in arb_bal]}")
        print(f"    arb a_i optimum : {[f'{a:.3e}' for a in arb_opt]}")
        print()


if __name__ == "__main__":
    report_symmetry_check()
    report_uniform([2, 3, 4, 5, 6])
    report_matched_skew(3, [0.10, 0.25, 0.50, 0.70, 0.90])
    report_spend_curve(3, [0.05, 0.15, 0.25, 0.3333, 0.45, 0.60, 0.75, 0.90])
    report_fixed_risk_family(500.0, [0.10, 0.25, 0.50, 0.75, 0.90])
    report_funding_lossless(3, 0.60)
    report_basket_uniform_family([2, 3, 4, 5, 6, 8])
    report_skew_optimization(
        [[0.6, 0.25, 0.15], [0.8, 0.1, 0.1], [0.5, 0.3, 0.2], [0.4, 0.3, 0.2, 0.1]]
    )
    report_add_funding([0.6, 0.25, 0.15])
