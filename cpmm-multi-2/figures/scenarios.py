#!/usr/bin/env python3
"""Shared scenario machinery for the cpmm-multi-2 figures — and a CLI for
digging into any specific case yourself:

    python scenarios.py --n 5 --featured 0.9 --ante 1000
    python scenarios.py --profile 0.55,0.25,0.1,0.06,0.04 --ante 2000

Prints the v1 creator cost (ante + self-trade to the profile, with the
forced-position resolution range) and the v2 cost (= ante) for the same target.

Everything computes through the reference implementation's auto-arb simulator
(the proofs' oracle) plus direct ports of the vendor creation rules.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reference")))
from manifold.amm_core import probability_from_pool  # noqa: E402
from manifold.market_simulator import MarketSimulator  # noqa: E402


# --------------------------------------------------------------------------- #
# Creation shapes (ports of common/src/new-contract.ts / calculate-cpmm.ts)
# --------------------------------------------------------------------------- #
def v1_uniform_pools(ante: float, n: int) -> list[tuple[float, float, float]]:
    """v1 sum-to-one creation: uniform 1/n only. (Y, N, p) per answer."""
    return [(ante / 2, ante / (2 * n - 2), 0.5) for _ in range(n)]


def v2_sum_to_one_pools(q: list[float], ante: float) -> list[tuple[float, float, float]]:
    """Port of cpmmMulti2SumToOnePools (sqrt-variance creation rule)."""
    n = len(q)
    sqrt_c = [math.sqrt(qi * (1 - qi)) for qi in q]
    mean_sqrt_c = sum(sqrt_c) / n
    d0 = (ante * (n - 2)) / (2 * (n - 1))
    wbar = (ante * n) / (4 * (n - 1))
    no = []
    for i, qi in enumerate(q):
        wi = wbar * sqrt_c[i] / mean_sqrt_c
        b = d0 - wi
        no.append((-b + math.sqrt(b * b + 4 * wi * qi * d0)) / 2)
    d = ante - sum(no)
    return [
        (ni + d, ni, (qi * (ni + d)) / (qi * (ni + d) + (1 - qi) * ni))
        for ni, qi in zip(no, q)
    ]


def worst_case_payout(pools: list[tuple[float, float, float]]) -> float:
    """Max mana the AMM owes across single-YES resolutions (sum-to-one)."""
    return max(
        y + sum(nj for j, (_, nj, _) in enumerate(pools) if j != i)
        for i, (y, _, _) in enumerate(pools)
    )


# --------------------------------------------------------------------------- #
# Simulator plumbing
# --------------------------------------------------------------------------- #
def build_market(pools: list[tuple[float, float, float]]) -> dict:
    answers = []
    for i, (y, n_, p) in enumerate(pools):
        pr = probability_from_pool(y, n_, p)
        answers.append(
            {"id": f"a{i}", "text": f"A{i}", "poolYes": y, "poolNo": n_,
             "poolYES": y, "poolNO": n_, "p": p, "prob": pr, "probability": pr}
        )
    return {
        "id": "m", "question": "q", "outcomeType": "MULTIPLE_CHOICE",
        "mechanism": "cpmm-multi-1", "shouldAnswersSumToOne": True,
        "answers": answers,
    }


def make_sim(pools: list[tuple[float, float, float]]) -> MarketSimulator:
    return MarketSimulator(build_market(pools), _internal=True, _validate=False)


def featured_profile(q: float, n: int) -> list[float]:
    """One featured answer at q, the rest splitting the remainder equally."""
    return [q] + [(1 - q) / (n - 1)] * (n - 1)


# --------------------------------------------------------------------------- #
# v1 self-trade: reach an arbitrary target profile from the uniform start
# --------------------------------------------------------------------------- #
def _buy_to_prob(sim: MarketSimulator, aid: str, target: float,
                 hi: float) -> tuple[float, float]:
    """Bisect the YES-buy amount on `aid` (auto-arb applied) to reach `target`.

    Returns (spend, shares). Only buys (target above current prob)."""
    lo = 0.0
    for _ in range(55):
        mid = 0.5 * (lo + hi)
        probe = sim.copy()
        probe.simulate_buy(aid, mid, "YES")
        if probe.get_probability(aid) < target:
            lo = mid
        else:
            hi = mid
    spend = 0.5 * (lo + hi)
    res = sim.simulate_buy(aid, spend, "YES")  # apply to the real sim
    return spend, res["shares"]


def v1_self_trade_to_profile(
    profile: list[float], ante: float, tol: float = 1e-4, max_passes: int = 200
) -> dict:
    """Cost for a v1 creator to push the uniform start to `profile`.

    Strategy: round-robin YES buys on every answer currently BELOW its target
    (auto-arb pushes the others down), repeated until all answers are within
    `tol`. Buying only the low answers converges because auto-arb distributes
    the complement. Returns spend, YES shares held per answer, and the final
    creator cost range over resolutions (ante sunk; shares pay out M1 iff
    their answer wins).
    """
    n = len(profile)
    assert abs(sum(profile) - 1) < 1e-9
    sim = make_sim(v1_uniform_pools(ante, n))
    spend = 0.0
    shares = {f"a{i}": 0.0 for i in range(n)}
    for _ in range(max_passes):
        worst = max(profile[i] - sim.get_probability(f"a{i}") for i in range(n))
        if worst < tol:
            break
        for i in range(n):
            aid = f"a{i}"
            if profile[i] - sim.get_probability(aid) > tol / 2:
                s, sh = _buy_to_prob(sim, aid, profile[i], hi=ante * 100)
                spend += s
                shares[aid] += sh
    else:
        raise RuntimeError("self-trade did not converge")
    for i in range(n):
        assert abs(sim.get_probability(f"a{i}") - profile[i]) < 2 * tol
    outlay = ante + spend
    finals = {aid: outlay - sh for aid, sh in shares.items()}  # if that answer wins
    return {
        "spend": spend,
        "outlay": outlay,
        "shares": shares,
        "final_cost_by_winner": finals,
        "final_min": min(finals.values()),
        "final_max": max(finals.values()),
        "sim": sim,
    }


def traded_v1_pools(profile: list[float], ante: float) -> list[dict]:
    """v1 pools (p = 0.5) for a market created uniform at `ante` and traded to
    `profile` — the realistic pre-state for add-answer / liquidity scenarios.
    Returns [{'YES': y, 'NO': n}, ...] in profile order."""
    if max(abs(q - 1 / len(profile)) for q in profile) < 1e-9:
        pools = v1_uniform_pools(ante, len(profile))
        return [{"YES": y, "NO": n} for y, n, _ in pools]
    sim = v1_self_trade_to_profile(profile, ante)["sim"]
    return [dict(sim.get_pool(f"a{i}")) for i in range(len(profile))]


def v1_fixed_outlay_pools(
    profile: list[float], budget: float
) -> tuple[list[tuple[float, float, float]], float]:
    """v1 pools at `profile` with TOTAL creator outlay (ante + self-trade) = budget.

    Everything is degree-1 homogeneous in the ante, so we run the self-trade at
    a reference ante and rescale all reserves by budget/outlay — probabilities
    are preserved exactly. Returns (pools, ante_used)."""
    r = v1_self_trade_to_profile(profile, 1000.0)
    f = budget / r["outlay"]
    sim = r["sim"]
    pools = []
    for i in range(len(profile)):
        pl = sim.get_pool(f"a{i}")
        pools.append((pl["YES"] * f, pl["NO"] * f, 0.5))
    return pools, 1000.0 * f


# --------------------------------------------------------------------------- #
# v1 whole-market liquidity add (port of addCpmmMultiLiquidityAnswersSumToOne)
# --------------------------------------------------------------------------- #
def v1_sum_to_one_add(
    probs: list[float], amount: float, ante: float = 1000.0, eps: float = 1e-9
) -> dict:
    """Port of calculate-cpmm.ts addCpmmMultiLiquidityAnswersSumToOne, with the
    discarded value tracked (the vendor drops it on the floor).

    Each round splits the remaining amount equally across answers and does the
    fixed-p add; thrown NO shares on answer i become YES shares on every other
    answer; the MIN of the per-answer thrown-YES tallies forms complete sets
    that are reinvested; the EXCESS above the min is discarded. The discard
    fraction is a pure function of the probability profile (adds preserve
    probability, and the map is homogeneous in the remaining amount).

    Returns discarded YES shares per answer, their EV at the profile, the
    per-resolution realized loss (discarded[j] materializes iff answer j wins),
    and the reserve deltas (dY_i, dN_i) the add left in each pool. Both the
    deltas and the discard are pure functions of (probs, amount) — each round's
    split depends only on the (preserved) probabilities, not on pool depth.
    """
    n = len(probs)
    pools = [[y, no] for (y, no, _p) in v1_uniform_pools(ante, n)]
    # reprice the uniform pools to the target profile shape at p = 0.5:
    # scale each answer's NO reserve so N/(Y+N) = q_i (prob preserved by adds).
    pools = [[y, y * q / (1 - q)] for (y, _), q in zip(pools, probs)]
    pools_before = [list(pl) for pl in pools]

    discarded = [0.0] * n
    remaining = amount
    rounds = 0
    while remaining > eps and rounds < 10_000:
        rounds += 1
        thrown_yes = [0.0] * n
        per = remaining / n
        for i in range(n):
            y, no = pools[i]
            q = no / (y + no)  # p = 1/2 probability
            if q < 0.5:
                pools[i] = [y + per, no + (q / (1 - q)) * per]
                thrown_no = per - (q / (1 - q)) * per
            else:
                pools[i] = [y + ((1 - q) / q) * per, no + per]
                thrown_yes[i] += per - ((1 - q) / q) * per
                thrown_no = 0.0
            for j in range(n):
                if j != i:
                    thrown_yes[j] += thrown_no
        m = min(thrown_yes)
        for i in range(n):
            discarded[i] += thrown_yes[i] - m
        remaining = m
    # probabilities are preserved by construction; assert it
    for (y, no), q in zip(pools, probs):
        assert abs(no / (y + no) - q) < 1e-7
    ev_loss = sum(d * q for d, q in zip(discarded, probs))
    deltas = [
        (after[0] - before[0], after[1] - before[1])
        for after, before in zip(pools, pools_before)
    ]
    # EV of the reserves the add left in pool i, at the profile's prices —
    # q·dY + (1−q)·dN.  Conservation: landed EV + discarded EV == amount.
    landed = [q * dy + (1 - q) * dn for (dy, dn), q in zip(deltas, probs)]
    assert abs(sum(landed) + ev_loss - amount) < 1e-6 * max(1.0, amount)
    return {
        "discarded_by_answer": discarded,
        "ev_loss": ev_loss,
        "loss_if_winner": discarded,  # realized loss if answer j resolves YES
        "min_loss": min(discarded),
        "max_loss": max(discarded),
        "rounds": rounds,
        "reserve_deltas": deltas,
        "landed_ev_by_answer": landed,
    }


# --------------------------------------------------------------------------- #
# v2 whole-market liquidity add (√variance MERGE, GP17 — vendor
# addCpmmMultiLiquidityAnswersSumToOneV2): merge a Δ-ante creation computed at
# the current probs into the existing reserves, re-price each p to hold prob.
# --------------------------------------------------------------------------- #
def v2_merge_landed(probs: list[float], amount: float) -> dict:
    """Per-answer EV of the reserves a v2 √variance merge add leaves in each pool.

    The merge's reserve deltas ARE cpmmMulti2SumToOnePools(probs, amount)
    (GP17a homogeneity), independent of the existing pool depth; re-pricing p
    moves no reserves. Valued at the profile's prices, q·dY + (1−q)·dN, the
    per-answer EVs sum to exactly `amount` (GP17c all-winners-tight identity) —
    the add is lossless, so there is no discarded term.
    """
    delta = v2_sum_to_one_pools(probs, amount)
    landed = [q * dy + (1 - q) * dn for (dy, dn, _p), q in zip(delta, probs)]
    assert abs(sum(landed) - amount) < 1e-6 * max(1.0, amount)
    return {"reserve_deltas": [(dy, dn) for dy, dn, _p in delta],
            "landed_ev_by_answer": landed}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ante", type=float, default=1000.0)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--featured", type=float,
                    help="featured answer's target prob (others equal)")
    ap.add_argument("--profile", type=str,
                    help="comma-separated full target profile (sums to 1)")
    args = ap.parse_args()

    if args.profile:
        profile = [float(x) for x in args.profile.split(",")]
    elif args.featured:
        profile = featured_profile(args.featured, args.n)
    else:
        ap.error("give --featured or --profile")

    n = len(profile)
    print(f"target profile (n={n}, ante=M{args.ante:,.0f}): "
          + ", ".join(f"{q:.3f}" for q in profile))

    r = v1_self_trade_to_profile(profile, args.ante)
    print("\ncpmm-multi-1 (uniform creation + forced self-trade):")
    print(f"  self-trade spend   M{r['spend']:10,.1f}")
    print(f"  cash outlay        M{r['outlay']:10,.1f}")
    print("  YES shares held:   "
          + ", ".join(f"a{i}={r['shares'][f'a{i}']:,.0f}" for i in range(n)))
    print("  final cost if answer i wins: "
          + ", ".join(f"a{i}=M{r['final_cost_by_winner'][f'a{i}']:,.0f}" for i in range(n)))
    print(f"  final cost range   [M{r['final_min']:,.1f}, M{r['final_max']:,.1f}]")

    pools = v2_sum_to_one_pools(profile, args.ante)
    assert abs(worst_case_payout(pools) - args.ante) < 1e-6
    print("\ncpmm-multi-2 (profile creation):")
    print(f"  outlay = final cost = M{args.ante:,.1f} (any resolution; no position)")


if __name__ == "__main__":
    _main()
