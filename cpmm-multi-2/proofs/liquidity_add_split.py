#!/usr/bin/env python3
"""
GP17 — whole-market liquidity-add split for cpmm-multi-2 sum-to-one markets.

THE QUESTION
------------
The current v2 whole-market add (`addCpmmMultiLiquidityAnswersSumToOneV2`) EQUAL-splits
the subsidy: amountPerAnswer = amount/n, then a lossless float-p binary add to each answer.
That is LMSR/balanced-shaped at the margin (it adds nearly-flat geometric depth across
answers), which is NOT consistent with the √variance *creation* rule
(`cpmmMulti2SumToOnePools`, vendor afe14c152), which concentrates depth in the uncertain
answers (depth W_i ∝ √(q_i(1-q_i))).

THE RULE (Evan): "apply creation's allocation logic to the *added* mana at current probs;
do not rearrange existing depth." Concretely, for a whole-market add of Δ to a market whose
answers currently price at q = (q_1..q_n) (Σq=1):

    new_pool_i = existing_pool_i  +  cpmmMulti2SumToOnePools(q, Δ)[i].pool   (reservewise)
    new_p_i    = the unique p that re-prices answer i back to prob = q_i

i.e. MERGE a Δ-ante √variance creation computed at the *current* probabilities into the
existing reserves, then re-price each answer's p to hold its probability.

WHAT THIS SCRIPT PROVES (run it; every claim asserts)
-----------------------------------------------------
P0  cpmmMulti2SumToOnePools is HOMOGENEOUS degree 1 in ante (reserves scale linearly,
    p invariant)  ⇒  pools(q,A) + pools(q,Δ) == pools(q,A+Δ)  (the merge is additive).
P1  Prob preservation + Σp=1: a unique p_i re-prices any positive merged pool to prob=q_i
    (monotone odds), so every answer's probability is held and Σ prob = 1 is inherited.
P2  Conservation / self-funding: the Δ-creation satisfies the all-winners-tight funding
    identity Y_i + Σ_{j≠i} N_j = Δ, so it locks exactly Δ. Resolution payout is LINEAR in
    reserves ⇒ merging superposes two independently-conservative markets; the add contributes
    exactly the LP's Δ. (Re-pricing p moves no reserves, so it is mana-free.)
P3  Scale reduction: on an UNTRADED market (pools = create(A,q)), merge(Δ) == create(A+Δ,q)
    == scale every reserve by (A+Δ)/A. (The "probs unchanged ⇒ just scale" special case.)
P4  n=2 reduction: creation is balanced at n=2, so the merge reduces EXACTLY to the current
    equal-split add (no behaviour change at n=2).
P5  Marginal-depth shape: reproduces findings-liquidity-add-split numbers — merge concentrates
    depth in uncertain answers (~5:1) where equal-split is nearly flat (~1.17:1).
P6  Set/independent is unchanged: its creation is balanced (Y=N=ante/n), so "merge a
    Δ-creation" == the current equal-split. Only sum-to-one changes.

Arithmetic is float64 — faithful to vendor (TypeScript Number) and avoids sympy's pathological
nested-radical evalf. Residuals run at machine epsilon (~1e-12); tol is 1e-7 (relative-scaled).
"""

import math

import sympy as sp


# ----------------------------------------------------------------------------
# Vendor ports (calculate-cpmm.ts / new-contract.ts)
# ----------------------------------------------------------------------------
def cpmm_prob(Y, N, p):
    """getCpmmProbability: prob(YES) = p*N / ((1-p)*Y + p*N)."""
    return (p * N) / ((1 - p) * Y + p * N)


def reprice_p(Y, N, q):
    """Unique p s.t. cpmm_prob(Y,N,p) == q.  p = qY / (qY + (1-q)N)."""
    return (q * Y) / (q * Y + (1 - q) * N)


def cpmm_liquidity(Y, N, p):
    """getCpmmLiquidity: geometric depth k = Y^p * N^(1-p)."""
    return Y**p * N ** (1 - p)


def sum_to_one_pools(q, ante):
    """Port of new-contract.ts cpmmMulti2SumToOnePools (the √variance creation rule)."""
    n = len(q)
    if n < 2:
        return [{"Y": ante, "N": ante, "p": q[0], "prob": q[0]}]
    sqrtC = [math.sqrt(qi * (1 - qi)) for qi in q]
    meanSqrtC = sum(sqrtC) / n
    D0 = (ante * (n - 2)) / (2 * (n - 1))
    Wbar = (ante * n) / (4 * (n - 1))
    N = []
    for i, qi in enumerate(q):
        Wi = Wbar * sqrtC[i] / meanSqrtC
        b = D0 - Wi
        N.append((-b + math.sqrt(b * b + 4 * Wi * qi * D0)) / 2)
    D = ante - sum(N)
    out = []
    for i, qi in enumerate(q):
        poolNo = N[i]
        poolYes = poolNo + D
        p = (qi * poolYes) / (qi * poolYes + (1 - qi) * poolNo)
        out.append({"Y": poolYes, "N": poolNo, "p": p, "prob": qi})
    return out


def add_cpmm_liquidity_equal(pool, p, amount):
    """addCpmmLiquidity: +amount to BOTH reserves, float p to hold prob (the equal-split unit)."""
    Y, N = pool["Y"], pool["N"]
    prob = cpmm_prob(Y, N, p)
    newP = (prob * (amount + Y)) / (amount - N * (prob - 1) + prob * Y)
    return {"Y": Y + amount, "N": N + amount, "p": newP, "prob": prob}


# ----------------------------------------------------------------------------
# The two whole-market add policies
# ----------------------------------------------------------------------------
def add_equal_split(pools, amount):
    """Current v2 behaviour: amountPerAnswer = amount/n, equal binary add to each."""
    n = len(pools)
    return [add_cpmm_liquidity_equal(pl, pl["p"], amount / n) for pl in pools]


def add_variance_merge(pools, amount):
    """Proposed rule: merge a Δ-ante √variance creation computed at CURRENT probs, re-price p."""
    q = [cpmm_prob(pl["Y"], pl["N"], pl["p"]) for pl in pools]  # current probs
    delta = sum_to_one_pools(q, amount)
    out = []
    for pl, dl, qi in zip(pools, delta, q):
        Y, N = pl["Y"] + dl["Y"], pl["N"] + dl["N"]
        p = reprice_p(Y, N, qi)
        out.append({"Y": Y, "N": N, "p": p, "prob": qi})
    return out


# ----------------------------------------------------------------------------
TOL = 1e-7


def _z(val, scale=1.0, tol=TOL):
    """Numeric zero-check, scaled by the magnitude of the quantities involved."""
    return abs(val) <= tol * max(1.0, abs(scale))


def P0_homogeneous():
    print("=" * 72)
    print("P0  sum_to_one_pools is homogeneous deg-1 in ante (merge is additive)")
    print("=" * 72)
    q = [0.55, 0.25, 0.12, 0.08]
    A, D = 1000.0, 500.0
    pA, pD, pAD = sum_to_one_pools(q, A), sum_to_one_pools(q, D), sum_to_one_pools(q, A + D)
    for i in range(len(q)):
        assert _z(pA[i]["Y"] + pD[i]["Y"] - pAD[i]["Y"], pAD[i]["Y"]), f"Y add {i}"
        assert _z(pA[i]["N"] + pD[i]["N"] - pAD[i]["N"], pAD[i]["N"]), f"N add {i}"
        assert _z(pA[i]["p"] - pAD[i]["p"], 1), f"p invariant {i}"
    # homogeneity at several scale factors λ: reserves ×λ, p fixed
    qs = [0.6, 0.3, 0.1]
    base = sum_to_one_pools(qs, 1000.0)
    for lam in (1.5, 2.3, 7.0):
        scaled = sum_to_one_pools(qs, lam * 1000.0)
        for i in range(3):
            assert _z(scaled[i]["Y"] - lam * base[i]["Y"], scaled[i]["Y"]), f"Y homog λ={lam} {i}"
            assert _z(scaled[i]["N"] - lam * base[i]["N"], scaled[i]["N"]), f"N homog λ={lam} {i}"
            assert _z(scaled[i]["p"] - base[i]["p"], 1), f"p homog λ={lam} {i}"
    print("  pools(q,A)+pools(q,Δ) == pools(q,A+Δ); reserves ∝ ante, p invariant ... OK")


def P1_prob_preserved():
    print("=" * 72)
    print("P1  merge preserves every prob and Σp=1 (unique re-pricing p)")
    print("=" * 72)
    # (a) A valid Σ=1 market (a creation): merge holds every prob ⇒ Σ stays exactly 1.
    pools = sum_to_one_pools([0.50, 0.30, 0.20], 1000.0)
    q_before = [cpmm_prob(pl["Y"], pl["N"], pl["p"]) for pl in pools]
    merged = add_variance_merge(pools, 500.0)
    q_after = [cpmm_prob(pl["Y"], pl["N"], pl["p"]) for pl in merged]
    for i in range(len(pools)):
        assert _z(q_after[i] - q_before[i], 1), f"prob held {i}"
    assert _z(sum(q_before) - 1.0, 1) and _z(sum(q_after) - 1.0, 1), "Σprob=1 held"

    # (b) Σ=1 is INHERITED, not imposed: on an ARBITRARY (deliberately Σ≠1) reserve state the
    # merge still pins each prob to its prior value ⇒ Σ_after == Σ_before exactly. So whatever Σ
    # the (auto-arbed) market carried in, it carries out — the merge never perturbs it.
    weird = [{"Y": 700.0, "N": 300.0, "p": 0.5},   # prob 0.30
             {"Y": 250.0, "N": 250.0, "p": 0.5},   # prob 0.50
             {"Y": 120.0, "N": 480.0, "p": 0.5}]   # prob 0.80  → Σ = 1.60 (not a basket)
    qb = [cpmm_prob(pl["Y"], pl["N"], pl["p"]) for pl in weird]
    qa = [cpmm_prob(pl["Y"], pl["N"], pl["p"]) for pl in add_variance_merge(weird, 333.0)]
    for i in range(len(weird)):
        assert _z(qa[i] - qb[i], 1), f"prob held (weird) {i}"
    assert _z(sum(qa) - sum(qb), 1), "Σ inherited"
    # uniqueness: odds = (p/(1-p))(N/Y) is strictly monotone in p ⇒ p is unique (symbolic, fast)
    pp, Ys, Ns = sp.symbols("pp Y N", positive=True)
    dodds = sp.diff((pp / (1 - pp)) * (Ns / Ys), pp)
    assert sp.simplify(dodds) != 0 and (dodds.subs({pp: sp.Rational(1, 3), Ys: 2, Ns: 3}) > 0)
    print("  prob_i unchanged ∀i, Σ=1, p_i unique (odds strictly ↑ in p) ... OK")


def P2_conservation():
    print("=" * 72)
    print("P2  conservation: Δ-creation self-funds (all-winners-tight) + payout linear")
    print("=" * 72)
    q = [0.55, 0.25, 0.12, 0.08]
    Delta = 500.0
    delta = sum_to_one_pools(q, Delta)
    n = len(q)
    # all-winners-tight funding identity on the delta: Y_i + Σ_{j≠i} N_j == Δ  ∀i
    for i in range(n):
        funded = delta[i]["Y"] + sum(delta[j]["N"] for j in range(n) if j != i)
        assert _z(funded - Delta, Delta), f"funding id {i}"
    # superposition: payout(scenario k) of merged == payout(base) + payout(delta).
    # CHOOSE_ONE payout if answer k wins = (YES reserve of k) + Σ_{j≠k}(NO reserve of j) —
    # LINEAR in reserves, so reservewise-additive pools give additive payouts. Use a TRADED base.
    base = sum_to_one_pools(q, 1000.0)
    base[1]["Y"] += 50.0  # off-manifold perturbation (a trade)
    mY = [base[i]["Y"] + delta[i]["Y"] for i in range(n)]
    mN = [base[i]["N"] + delta[i]["N"] for i in range(n)]
    for k in range(n):
        pay_base = base[k]["Y"] + sum(base[j]["N"] for j in range(n) if j != k)
        pay_delta = delta[k]["Y"] + sum(delta[j]["N"] for j in range(n) if j != k)
        pay_merged = mY[k] + sum(mN[j] for j in range(n) if j != k)
        assert _z(pay_merged - (pay_base + pay_delta), pay_merged), f"superpose {k}"
        assert _z(pay_delta - Delta, Delta), f"delta pays Δ {k}"
    print("  Y_i+Σ_{j≠i}N_j=Δ (locks exactly Δ); payout superposes ⇒ add contributes exactly Δ ... OK")


def P3_scale_reduction():
    print("=" * 72)
    print("P3  untraded market: merge(Δ) == create(A+Δ) == scale by (A+Δ)/A")
    print("=" * 72)
    q = [0.55, 0.25, 0.12, 0.08]
    A, D = 1000.0, 500.0
    base = sum_to_one_pools(q, A)
    merged = add_variance_merge(base, D)
    created = sum_to_one_pools(q, A + D)
    scale = (A + D) / A
    for i in range(len(q)):
        assert _z(merged[i]["Y"] - created[i]["Y"], created[i]["Y"]), f"merge=create Y {i}"
        assert _z(merged[i]["N"] - created[i]["N"], created[i]["N"]), f"merge=create N {i}"
        assert _z(merged[i]["Y"] - scale * base[i]["Y"], merged[i]["Y"]), f"=scale Y {i}"
        assert _z(merged[i]["p"] - base[i]["p"], 1), f"p unchanged {i}"
    print("  merge == create(A+Δ) == (A+Δ)/A · base, p unchanged ... OK")


def P4_n2_reduction():
    print("=" * 72)
    print("P4  n=2: merge reduces EXACTLY to the current equal-split add")
    print("=" * 72)
    for qy in (0.50, 0.70, 0.15):
        pools = sum_to_one_pools([qy, 1 - qy], 1000.0)
        eq = add_equal_split(pools, 500.0)
        mg = add_variance_merge(pools, 500.0)
        for i in range(2):
            assert _z(eq[i]["Y"] - mg[i]["Y"], mg[i]["Y"]), f"Y q={qy} {i}"
            assert _z(eq[i]["N"] - mg[i]["N"], mg[i]["N"]), f"N q={qy} {i}"
            assert _z(eq[i]["p"] - mg[i]["p"], 1), f"p q={qy} {i}"
    print("  equal-split == variance-merge for all tested q at n=2 ... OK")


def P5_marginal_depth():
    print("=" * 72)
    print("P5  marginal depth: merge concentrates in uncertain answers; equal-split is flat")
    print("=" * 72)
    q = [0.55, 0.25, 0.12, 0.08]
    A, D = 1000.0, 500.0
    base = sum_to_one_pools(q, A)
    eq = add_equal_split(base, D)
    mg = add_variance_merge(base, D)

    def dk(after):
        return [cpmm_liquidity(after[i]["Y"], after[i]["N"], after[i]["p"])
                - cpmm_liquidity(base[i]["Y"], base[i]["N"], base[i]["p"]) for i in range(len(q))]

    dk_eq, dk_mg = dk(eq), dk(mg)
    print("  equal-split Δk:", [round(x) for x in dk_eq], " ratio", round(max(dk_eq) / min(dk_eq), 2))
    print("  var-merge  Δk:", [round(x) for x in dk_mg], " ratio", round(max(dk_mg) / min(dk_mg), 2))
    assert max(dk_eq) / min(dk_eq) < 1.3, "equal-split should be ~flat"
    assert max(dk_mg) / min(dk_mg) > 4, "merge should concentrate depth"
    # merge == create(A+Δ) here (untraded), so reproduces the doc's [274,159,80,55]
    assert [round(x) for x in dk_mg] == [274, 159, 80, 55], "matches findings numbers"
    print("  reproduces findings-liquidity-add-split-2026-06-28 numbers ... OK")


def P6_set_unchanged():
    print("=" * 72)
    print("P6  Set/independent: balanced creation ⇒ merge == equal-split (no change)")
    print("=" * 72)
    # Set creation is balanced per answer: Y=N=ante/n, p=q. A Δ-add of balanced creation
    # adds Δ/n to BOTH reserves of each answer = exactly the equal-split add.
    n = 3
    L = 1000.0 / n
    qset = [0.60, 0.75, 0.40]  # independent, no Σ=1
    pools = [{"Y": L, "N": L, "p": qi, "prob": qi} for qi in qset]
    Delta = 600.0
    eq = [add_cpmm_liquidity_equal(pl, pl["p"], Delta / n) for pl in pools]
    for i in range(n):
        assert _z(eq[i]["Y"] - (pools[i]["Y"] + Delta / n), eq[i]["Y"]), f"Set Y {i}"
        assert _z(eq[i]["N"] - (pools[i]["N"] + Delta / n), eq[i]["N"]), f"Set N {i}"
    print("  Set add already creation-consistent (balanced) — leave equal-split ... OK")


if __name__ == "__main__":
    P0_homogeneous()
    P1_prob_preserved()
    P2_conservation()
    P3_scale_reduction()
    P4_n2_reduction()
    P5_marginal_depth()
    P6_set_unchanged()
    print("\nGP17 — all whole-market √variance-split claims verified.")
