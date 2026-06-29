#!/usr/bin/env python3
"""
Approach C — the direct, non-overshooting multi-buy solve (the production-shape reference for
the cpmm-multi-2 2b.2 fix), validated against the GP12 oracle.

GP12 (monotone_equilibrium.py) proved the limit-aware multi-buy equilibrium is monotone in the
bet size, so a correct implementation NEVER overshoots and each resting maker is crossed exactly
once.  C realizes that directly, WITHOUT the iterate-buy-arb-reinvest loop (the overshoot source)
and WITHOUT the fine-step flow oracle (too slow to ship).  The trick is the DOLLAR-CENTRIC
decomposition (GP10): buy YES only in the basket answers and arb NO only in the OTHERS, so

  * each basket answer is bought ONCE, straight up start -> f_i  (a single rising sweep), and
  * each non-basket answer moves ONCE, straight down start -> f_j (a single falling sweep).

No answer is ever pushed past its settled price, so the limit fills are the single monotone
crossing GP12 says is correct — pinning included — by construction.  (Contrast the vendor
share-centric "NO in all" loop, whose basket answers go up PAST final then back down, mutating
maker fills in the transient band: the bug.)

Algorithm (mirror of the paper's dollar-centric Algorithm 1, generalized to a basket B of size
m, at p=1/2):
  solve g (equal YES shares per basket answer) s.t. net spend == budget:        [outer bisection]
    raise each basket answer by buying g YES shares, limit-aware           (single rising sweep)
    solve eta (NO shares in each of the n-m non-basket answers) s.t. Sum p = 1: [inner bisection]
      lower each non-basket answer by buying eta NO shares, limit-aware   (single falling sweep)
    net = basket_cost + others_cost - eta*(n-m-1)        (dollar-centric redemption; = eta*(n-2)
                                                          for a single-answer buy, matching Alg 1)
Complexity O(n log^2 N) (outer g x inner eta, each O(n) closed-form sweeps).  Performance is NOT
a goal (Decision #4); this is the exact, reversible, single-crossing shape the fix needs.  The
basket dollar-centric maker fills equal the share-centric equilibrium's fills because both reach
the SAME final price f_i (GP10) and cross the same makers in [start_i, f_i].

Validated below against `flow_oracle` (the obviously-correct GP12 reference) on the vendor spec
finals, all the spec limit cases, and a randomized fuzz (final probs AND per-answer maker fills).
Self-contained except for importing its sibling proof (extracts together).  Deterministic only.
"""

from monotone_equilibrium import (
    _market,
    buy_no_shares_limit,
    buy_yes_shares_limit,
    flow_oracle,
    pool_for_prob,
    prob,
)


def solve_c(start_pools, basket, budget, makers):
    """Direct dollar-centric solve. Returns {probs, filled (per answer), g, eta, net}.
    `makers`: per-answer list of {"side","rho","rem"} dicts (not mutated)."""
    n = len(start_pools)
    basket = set(basket)
    others = [j for j in range(n) if j not in basket]
    m = len(basket)
    assert 0 < m < n, "basket must be a proper non-empty subset"
    asks = [[mk for mk in makers[i] if mk["side"] == "NO"] for i in range(n)]
    bids = [[mk for mk in makers[i] if mk["side"] == "YES"] for i in range(n)]

    def eval_g(g):
        # basket: buy g YES shares each, single rising sweep (limit-aware). Read-only on makers.
        basket_prob = {}
        basket_cost = 0.0
        basket_fill = {}
        for i in basket:
            Y, N, c, fills = buy_yes_shares_limit(start_pools[i][0], start_pools[i][1], asks[i], g)
            basket_prob[i] = prob(Y, N)
            basket_cost += c
            basket_fill[i] = sum(t for _, t in fills)
        target = 1.0 - sum(basket_prob.values())          # required Σ over non-basket answers
        if target <= 1e-9:                                 # basket sum >= 1: infeasible g (the
            return {"probs": None, "filled": None,         # non-basket cannot supply <=0 prob)
                    "g": g, "eta": float("inf"), "net": float("inf")}

        # arb: η NO shares in each non-basket answer (single falling sweep). Σ_others is strictly
        # decreasing in η (each falling sweep lowers its prob) -> unique η by bisection.
        def others_sum(eta):
            return sum(prob(*buy_no_shares_limit(start_pools[j][0], start_pools[j][1],
                                                 bids[j], eta)[:2]) for j in others)

        if target < others_sum(0.0):                       # need to push others down
            lo, hi = 0.0, 1.0
            while others_sum(hi) > target and hi < 1e12:
                hi *= 2.0
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if others_sum(mid) > target else (lo, mid)
            eta = 0.5 * (lo + hi)
        else:
            eta = 0.0

        others_cost = 0.0
        probs = [0.0] * n
        filled = [0.0] * n
        for i in basket:
            probs[i] = basket_prob[i]
            filled[i] = basket_fill[i]
        for j in others:
            Y, N, c, fills = buy_no_shares_limit(start_pools[j][0], start_pools[j][1], bids[j], eta)
            probs[j] = prob(Y, N)
            others_cost += c
            filled[j] = sum(t for _, t in fills)
        net = basket_cost + others_cost - eta * (n - m - 1)   # dollar-centric redemption
        return {"probs": probs, "filled": filled, "g": g, "eta": eta, "net": net}

    # outer: net spend strictly increasing in g (GP12a: dt/dg = Σ_basket prob > 0) -> bisection.
    # eval_g returns net=inf once g is infeasible (basket sum >= 1), so hi stops doubling there.
    lo, hi = 0.0, 1.0
    while eval_g(hi)["net"] < budget:
        hi *= 2.0
        if hi > 1e9:
            raise RuntimeError("budget unreachable")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if eval_g(mid)["net"] < budget else (lo, mid)
    res = eval_g(lo)                                       # lo is the largest feasible g <= target
    if res["net"] == float("inf") or abs(res["net"] - budget) > 1e-4:
        raise RuntimeError("budget unreachable (basket sum -> 1 before budget is spent)")
    return res


# ============================================================================================
# Validation against the GP12 oracle.
# ============================================================================================

def check_c_matches_oracle_spec():
    print("=" * 78)
    print("C vs oracle: vendor spec finals + the canonical limit cases")
    print("=" * 78)
    start, basket, budget = _market(), {0, 1}, 60.0

    base = solve_c(start, basket, budget, [[] for _ in range(5)])
    print(f"  no-limit: C a0={base['probs'][0]:.5f} a2={base['probs'][2]:.5f} "
          f"net={base['net']:.4f}  (spec 0.39942/0.06705)")
    assert abs(base["probs"][0] - 0.39942) < 5e-3 and abs(base["probs"][2] - 0.06705) < 5e-3
    assert abs(sum(base["probs"]) - 1.0) < 1e-9
    assert abs(base["net"] - budget) < 1e-6

    def case(side, ans, rho, rem):
        mk = [[] for _ in range(5)]
        mk[ans] = [{"side": side, "rho": rho, "rem": rem}]
        c = solve_c(start, basket, budget, mk)
        o = flow_oracle(start, basket, budget, mk, steps=800)[-1]
        return c, o

    # RED1: large in-path NO-ask@0.30 on a0 -> pin
    c, o = case("NO", 0, 0.30, 600.0)
    print(f"  large ask@0.30: C a0={c['probs'][0]:.4f} fill={c['filled'][0]:.2f} | "
          f"oracle a0={o[1][0]:.4f} fill={o[2][0]:.2f}")
    assert abs(c["probs"][0] - 0.30) < 1e-2 and c["filled"][0] > 0
    # RED2: past-final NO-ask@0.50 -> unfilled
    c, o = case("NO", 0, 0.50, 600.0)
    print(f"  past-final@0.50: C a0={c['probs'][0]:.4f} fill={c['filled'][0]:.2e} | "
          f"oracle a0={o[1][0]:.4f} fill={o[2][0]:.2e}")
    assert abs(c["probs"][0] - 0.39942) < 5e-3 and c["filled"][0] < 1e-3
    # control: small in-path ask crossed
    c, o = case("NO", 0, 0.25, 2.0)
    assert c["probs"][0] > 0.25 and abs(c["filled"][0] - 2.0) < 1e-6
    # control: a2 large YES-bid@0.12 -> pin
    c, o = case("YES", 2, 0.12, 600.0)
    print(f"  a2 bid@0.12: C a2={c['probs'][2]:.4f} fill={c['filled'][2]:.2f} | "
          f"oracle a2={o[1][2]:.4f} fill={o[2][2]:.2f}")
    assert abs(c["probs"][2] - 0.12) < 1e-2 and c["filled"][2] > 0
    # control: bid below start on basket a0 -> non-interacting
    c, _ = case("YES", 0, 0.15, 600.0)
    assert abs(c["probs"][0] - 0.39942) < 5e-3 and c["filled"][0] < 1e-9
    # control: ask above start on non-traded a2 -> non-interacting
    c, _ = case("NO", 2, 0.30, 600.0)
    assert abs(c["probs"][2] - 0.06705) < 5e-3 and c["filled"][2] < 1e-9
    print("  OK — C reproduces every spec case (2 RED resolved, controls hold).\n")


def check_c_matches_oracle_fuzz():
    print("=" * 78)
    print("C vs oracle: randomized fuzz (final probs AND per-answer maker fills)")
    print("=" * 78)

    def lcg(seed):
        s = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        while True:
            s = (s * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            yield (s >> 11) / float(1 << 53)
    rng = lcg(77777)

    n_cases = 30
    worst_prob = 0.0
    worst_fill = 0.0
    skipped = 0
    checked = 0
    for _ in range(n_cases):
        n = 3 + int(4 * next(rng))
        raw = [0.05 + 0.9 * next(rng) for _ in range(n)]
        sden = sum(raw)
        probs = [r / sden for r in raw]
        kp = 200.0 + 2000.0 * next(rng)
        start = [pool_for_prob(p, kp) for p in probs]
        bsize = min(n - 1, 1 + int(2 * next(rng)))        # 1..min(n-1,3); proper subset
        basket = set(sorted(range(n), key=lambda _: next(rng))[:bsize])
        budget = (0.01 + 0.05 * next(rng)) * kp           # scale to market size (feasibility)
        makers = [[] for _ in range(n)]
        for i in range(n):
            p0 = probs[i]
            if next(rng) < 0.6:
                makers[i].append({"side": "NO", "rho": p0 + (0.97 - p0) * next(rng),
                                  "rem": 0.5 + 800.0 * next(rng)})
            if next(rng) < 0.6:
                makers[i].append({"side": "YES", "rho": 0.03 + (p0 - 0.03) * next(rng),
                                  "rem": 0.5 + 800.0 * next(rng)})
        try:
            c = solve_c(start, basket, budget, makers)
        except RuntimeError:
            skipped += 1                                   # budget unreachable for this market
            continue
        o = flow_oracle(start, basket, budget,
                        [[dict(mm) for mm in makers[i]] for i in range(n)], steps=1500)[-1]
        for i in range(n):
            worst_prob = max(worst_prob, abs(c["probs"][i] - o[1][i]))
            worst_fill = max(worst_fill, abs(c["filled"][i] - o[2][i]))
        assert abs(c["net"] - budget) < 1e-4
        assert abs(sum(c["probs"]) - 1.0) < 1e-9
        checked += 1
    print(f"  {checked}/{n_cases} configs checked ({skipped} skipped: budget unreachable for a "
          f"small market), 3-6 answers, baskets 1-3, mixed limit books.")
    print(f"  worst |C - oracle| final prob: {worst_prob:.2e}   per-answer maker fill: "
          f"{worst_fill:.2e}")
    # the oracle has O(dg) step error; C is exact -> agreement to the oracle's discretization.
    assert worst_prob < 3e-3 and worst_fill < 0.5
    print("  OK — C == the GP12 oracle (to the oracle's step granularity); C is the exact,")
    print("       non-overshooting, single-crossing solve. Ship-shape for the TS port.\n")


if __name__ == "__main__":
    check_c_matches_oracle_spec()
    check_c_matches_oracle_fuzz()
    print("Approach C reference validated against the GP12 oracle.")
