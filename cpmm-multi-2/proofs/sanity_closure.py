#!/usr/bin/env python3
"""
GP19 — SANITY CLOSURE for cpmm-multi-2: if a market starts sane, it stays sane.

THE GAP BEING FIXED (found 2026-07-01, external code review of PR 3934)
-----------------------------------------------------------------------
"Sane" = every answer has poolYes > 0, poolNo > 0, p in (0,1) (hence prob in (0,1)).
GP15's sqrt-variance creation optimum, implemented as `cpmmMulti2SumToOnePools`
(vendor calculate-cpmm.ts:847-878), produces NEGATIVE poolYes and p outside (0,1)
for skewed many-answer prob vectors: ante=1000, n=30, q = [0.90, 29 x 0.1/29]
gives D = -171.7 and poolYes = -170.2 on every longshot. The GP15 sympy proof
assumed positivity via symbols(..., positive=True); the numerical companions
sampled only n in {3..6}. Neither saw it. This script closes the gap: it
characterizes exactly WHEN each closed form is sane, and proves the market
operations preserve sanity inside those domains.

WHAT THIS SCRIPT PROVES (run it; every claim asserts)
-----------------------------------------------------
GP19a  Creation feasibility lemma. Feasibility depends only on (q, n) — the
       construction is homogeneous deg-1 in ante (symbolic). poolNo_i > 0 ALWAYS;
       the sole failure mode is poolYes: sane <=> D > -min_i N_i <=>
       sum_j N_j(q,1) - min_i N_i(q,1) < 1 (exact algebraic characterization,
       fuzz-verified). Boundary numerically mapped: rest-uniform q_max(n) table
       (feasible for ALL q when n <= 20; boundary appears at n = 21), plus
       two-big-answers and geometric-decay families.
GP19b  Trading closure — UNCONDITIONAL. Buys/sells move along Y^p N^(1-p) = k
       with k > 0: both reserves stay positive for ANY spend and any p in (0,1)
       (symbolic: explicit positive powers), and trades never touch p. The
       market-order prob clamp [MIN,MAX]_CPMM_PROB = [0.01,0.99]
       (calculate-cpmm.ts:250-257, new-bet.ts:121, contract.ts:518-519) bounds
       the state away from the boundary: reserve floors Y >= k*rho^(p-1),
       N >= k*rho^p with rho = N/Y pinned by prob. Numeric fuzz incl. adversarial
       states (n up to 100, dominant probs, tiny reserves, extreme p): random
       buy/sell/auto-arb sequences keep reserves > 0, p unchanged, Sigma q = 1.
GP19c  Whole-market add — the CONDITIONAL closure (the headline). The add
       (addCpmmMultiLiquidityAnswersSumToOneV2, calculate-cpmm.ts:897-932) merges
       Delta = create(q_current, A). (i) DeltaN_i >= 0 always (nonneg quadratic
       root); only DeltaY_i can go negative, exactly when q is creation-infeasible.
       (ii) The critical amount A*(state) = min_{i: dY_i<0} Y_i / |dY_i(q, ante=1)|:
       below A* the merged pools are positive and the repricing gives p in (0,1);
       at A >= A* sanity breaks. (iii) Homogeneity => repeated small adds at fixed
       q accumulate to the SAME bound — TOTAL added liquidity vs A*, independent of
       drip size (100-tick drizzle breaches exactly when the lump does).
       (iv) Corollary: feasible q => A* = infinity, closure unconditional.
GP19d  Other-split closure. The GP18 construction (create-answer-cpmm.ts:472-508:
       targetProb = probOther/2, both pools {YES: Yo+a, NO: a}, a = answerCost/2,
       p* = GP6a repricing) is sane whenever probOther in (0,1) and a > 0 — but
       repeated splits HALVE probOther geometrically toward MIN_CPMM_PROB = 0.01:
       a market starting at Other = 50% survives exactly 5 splits; the 6th lands
       both children at 0.0078 < 0.01. Precondition: probOther >= 2*MIN_CPMM_PROB.
GP19e  Per-answer independent add + conversion. The lossless per-answer add
       (addCpmmLiquidity, calculate-cpmm.ts:744-763) floats p to exactly the GP6a
       weight of the grown pool => p in (0,1) and prob preserved for any positive
       reserves (GP17b's hypothesis made explicit). v1 -> v2 conversion is the
       identity embedding (same pools, p = 0.5): sane -> sane trivially.

PRECONDITIONS DISCIPLINE: every closed form promoted to code must have its domain
condition promoted to a theorem or a runtime check. GP19 implies three guards:
creation feasibility check (GP19a), add A* guard (GP19c), split floor (GP19d).

Vendor formulas transcribed faithfully (TypeScript Number = float64), cited
file:line. No project imports. Companion to general_p_cost.py (GP2/GP4 mechanics),
liquidity_add_split.py (GP17), other_split.py (GP6). Indexed in theorems_summary.md.
"""

import math

import numpy as np
import sympy as sp
from sympy import simplify, sqrt, symbols

# ----------------------------------------------------------------------------
# Vendor ports (float64, faithful transcriptions)
# ----------------------------------------------------------------------------
MIN_CPMM_PROB = 0.01  # contract.ts:519
MAX_CPMM_PROB = 0.99  # contract.ts:518


def cpmm_prob(Y, N, p):
    """getCpmmProbability (calculate-cpmm.ts:27-33): prob = pN / ((1-p)Y + pN)."""
    return (p * N) / ((1 - p) * Y + p * N)


def weight_for_prob(Y, N, q):
    """GP6a repricing weight (create-answer-cpmm.ts:499-500, calculate-cpmm.ts:924-925):
    the unique p with cpmm_prob(Y,N,p) == q, namely p* = qY / (qY + (1-q)N)."""
    return (q * Y) / (q * Y + (1 - q) * N)


def sum_to_one_pools(q, ante):
    """Port of cpmmMulti2SumToOnePools (calculate-cpmm.ts:847-878), the sqrt-variance
    creation rule: D0 = ante(n-2)/(2(n-1)), Wbar = ante*n/(4(n-1)),
    W_i = Wbar*sqrt(q_i(1-q_i))/mean(sqrt(q(1-q))),
    N_i = (-b + sqrt(b^2 + 4 W_i q_i D0))/2 with b = D0 - W_i,
    D = ante - Sigma N_j, poolYes_i = N_i + D, p_i = q_i Y/(q_i Y + (1-q_i) N).
    Returns raw output — including insane values when q is infeasible."""
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
        denom = qi * poolYes + (1 - qi) * poolNo
        p = (qi * poolYes / denom) if denom != 0 else float("nan")
        out.append({"Y": poolYes, "N": poolNo, "p": p, "prob": qi})
    return out


def calc_amount_to_prob(Y, N, p, prob, outcome):
    """calculateCpmmAmountToProb (calculate-cpmm.ts:152-169): the bet amount that moves
    the market to `prob` (YES-frame for YES, converted internally for NO)."""
    if outcome == "NO":
        prob = 1 - prob
    k = Y**p * N ** (1 - p)
    if outcome == "YES":
        r = (p * (prob - 1)) / ((p - 1) * prob)
        return r**-p * (k - N * r**p)
    r = ((1 - p) * (prob - 1)) / (-p * prob)
    return r ** (p - 1) * (k - Y * r ** (1 - p))


def add_cpmm_liquidity(Y, N, p, amount):
    """addCpmmLiquidity (calculate-cpmm.ts:744-763): +amount to BOTH reserves, float p
    to hold prob. newP = prob(amount+Y) / (amount - N(prob-1) + prob*Y)."""
    prob = cpmm_prob(Y, N, p)
    newP = (prob * (amount + Y)) / (amount - N * (prob - 1) + prob * Y)
    return Y + amount, N + amount, newP


def whole_market_add(pools, amount):
    """addCpmmMultiLiquidityAnswersSumToOneV2 (calculate-cpmm.ts:897-932): merge a
    Delta = create(q_current, amount) reservewise, re-price each p to hold prob."""
    q = [cpmm_prob(pl["Y"], pl["N"], pl["p"]) for pl in pools]
    delta = sum_to_one_pools(q, amount)
    out = []
    for pl, dl, qi in zip(pools, delta, q):
        Y, N = pl["Y"] + dl["Y"], pl["N"] + dl["N"]
        denom = qi * Y + (1 - qi) * N
        p = (qi * Y / denom) if denom != 0 else float("nan")
        out.append({"Y": Y, "N": N, "p": p, "prob": qi})
    return out


def other_split_v2(Yo, No, pOther, answerCost):
    """createAnswerAndSumAnswersToOneV2 pool surgery (create-answer-cpmm.ts:490-508):
    targetProb = probOther/2, a = answerCost/2, both new pools {YES: Yo+a, NO: a},
    each repriced via weight_for_prob to targetProb."""
    probOther = cpmm_prob(Yo, No, pOther)
    targetProb = probOther / 2
    a = answerCost / 2
    poolY, poolN = Yo + a, a
    newAnswerP = weight_for_prob(poolY, poolN, targetProb)
    newOtherP = weight_for_prob(poolY, poolN, targetProb)
    return (
        {"Y": poolY, "N": poolN, "p": newAnswerP},
        {"Y": poolY, "N": poolN, "p": newOtherP},
        targetProb,
    )


# ----------------------------------------------------------------------------
# Sanity predicate + helpers
# ----------------------------------------------------------------------------
def sane_answer(pl):
    return pl["Y"] > 0 and pl["N"] > 0 and 0 < pl["p"] < 1


def sane(pools):
    return all(sane_answer(pl) for pl in pools)


def feasible(q):
    """Creation feasibility: the sqrt-variance creation at probs q is sane.
    Homogeneous in ante (GP19a.1), so ante = 1 WLOG."""
    return sane(sum_to_one_pools(q, 1.0))


def _z(val, scale=1.0, tol=1e-9):
    return abs(val) <= tol * max(1.0, abs(scale))


def _bisect(f, lo, hi, iters=100):
    """f(lo) and f(hi) have opposite signs; returns the sign-change point."""
    flo = f(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if (f(mid) > 0) == (flo > 0):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ============================================================================
# GP19a — creation feasibility lemma
# ============================================================================
def GP19a_creation_feasibility():
    print("=" * 76)
    print("GP19a  creation feasibility: F = {(q,n) : all poolYes>0, all p in (0,1)}")
    print("=" * 76)

    # --- a.1 (symbolic) homogeneity deg-1 in ante => feasibility depends only on (q,n).
    # Build N_i symbolically with the depth profile abstracted (s_i = sqrtC_i, m = mean):
    # every ante-dependence sits in D0, Wbar (both linear in ante).
    ante, lam, s_i, m, q_i, nn = symbols("ante lam s_i m q_i n", positive=True)
    D0 = ante * (nn - 2) / (2 * (nn - 1))
    Wbar = ante * nn / (4 * (nn - 1))
    Wi = Wbar * s_i / m
    b = D0 - Wi
    Ni = (-b + sqrt(b**2 + 4 * Wi * q_i * D0)) / 2
    # N_i(lam*ante) == lam * N_i(ante)  (needs n>=2 so D0>=0; sympy holds it symbolically)
    homog = simplify(Ni.subs(ante, lam * ante) - lam * Ni)
    print(f"  N_i(lam*ante) - lam*N_i(ante) = {homog}  (expect 0)")
    assert homog == 0
    # Then D = ante - Sigma N_j and poolYes = N_i + D scale by lam too (sums/differences of
    # deg-1 terms), and p = qY/(qY+(1-q)N) is a ratio of deg-1 terms => invariant.
    # => the SIGN pattern of (poolYes, poolNo, p-in-(0,1)) is ante-free: feasibility = f(q, n).
    print("  => D, poolYes ~ ante; p invariant => feasibility depends only on (q, n). OK")
    # numeric spot-check: same feasibility verdict across antes, feasible + infeasible q
    for qv in ([0.9] + [0.1 / 29] * 29, [0.4, 0.3, 0.2, 0.1]):
        verdicts = {a: sane(sum_to_one_pools(qv, a)) for a in (1.0, 7.3, 1000.0, 1e8)}
        assert len(set(verdicts.values())) == 1, f"feasibility must be ante-free: {verdicts}"
    print("  numeric: verdict identical at ante in {1, 7.3, 1000, 1e8} ... OK")

    # --- a.2 (symbolic + fuzz) poolNo_i = N_i > 0 ALWAYS (n >= 2).
    # N_i = (-b + s)/2 with s = sqrt(b^2 + c), c = 4 W_i q_i D0 >= 0 (D0 >= 0 for n >= 2).
    # s^2 - b^2 = c (symbolic identity below); s >= 0 => s >= |b| >= b => N_i >= 0,
    # strict when c > 0 (n >= 3); at n = 2, D0 = 0, b = -W_i < 0 => N_i = W_i > 0.
    bb, cc = symbols("bb cc", real=True, positive=None)
    s_expr = sqrt(bb**2 + cc)
    assert simplify(s_expr**2 - bb**2 - cc) == 0
    print("  s^2 - b^2 = 4 W q D0 >= 0 and s >= 0 => s >= |b| => N_i = (s-b)/2 >= 0.")
    n2 = sum_to_one_pools([0.7, 0.3], 100.0)  # n=2: D0=0 corner
    assert all(pl["N"] > 0 for pl in n2)
    rng = np.random.default_rng(19)
    for _ in range(500):
        n = int(rng.integers(2, 101))
        w = rng.exponential(1.0, n) + 1e-12
        qv = list(w / w.sum())
        assert all(pl["N"] > 0 for pl in sum_to_one_pools(qv, 1.0)), "poolNo must be > 0"
    print("  fuzz 500 random q (n in [2,100]): poolNo_i > 0 in every case ... OK")

    # --- a.3 (characterization) sane <=> min poolYes > 0 <=> D > -min N_i
    #         <=> Sigma_j N_j(q,1) - min_i N_i(q,1) < 1.
    # Given N > 0 and q in (0,1): p = qY/(qY+(1-q)N) is in (0,1) iff Y > 0
    # (Y>0: ratio x/(x+y) of positives; Y<=0: numerator <= 0 while p in (0,1) needs
    # numerator and denominator positive — check both branches numerically below).
    mismatches = 0
    for _ in range(4000):
        n = int(rng.integers(3, 101))
        style = rng.random()
        if style < 0.4:
            Q = float(rng.uniform(1.0 / n, 0.999))
            qv = [Q] + [(1 - Q) / (n - 1)] * (n - 1)
        elif style < 0.7:
            w = rng.exponential(1.0, n) + 1e-12
            qv = list(w / w.sum())
        else:
            r = float(rng.uniform(0.05, 1.0))
            w = np.array([r**i for i in range(n)])
            qv = list(w / w.sum())
        pools = sum_to_one_pools(qv, 1.0)
        Ns = [pl["N"] for pl in pools]
        direct = sane(pools)
        char = (sum(Ns) - min(Ns)) < 1.0  # <=> D > -min N_i at ante=1
        y_test = min(pl["Y"] for pl in pools) > 0
        p_test = all(0 < pl["p"] < 1 for pl in pools)
        if not (direct == char == y_test == p_test):
            mismatches += 1
    assert mismatches == 0, f"{mismatches} characterization mismatches"
    print("  fuzz 4000 q-vectors: sane <=> minY>0 <=> p-sane <=> SigmaN - minN < ante")
    print("  => F = {(q,n) : Sigma_j N_j(q,1) - min_i N_i(q,1) < 1}  (exact, algebraic). OK")

    # --- a.4 (repro) the discovery case: ante=1000, n=30, q = [0.90, 29 x 0.1/29].
    qd = [0.9] + [0.1 / 29] * 29
    pools = sum_to_one_pools(qd, 1000.0)
    D = pools[0]["Y"] - pools[0]["N"]  # poolYes = N_i + D for every i
    minY = min(pl["Y"] for pl in pools)
    minP = min(pl["p"] for pl in pools)
    print("  DISCOVERY REPRO n=30, dominant q=0.90, ante=1000:")
    print(f"    D = {D:.2f} (< 0), longshot poolYes = {minY:.2f} (< 0), min p = {minP:.3f}")
    assert D < 0 and minY < 0 and not (0 < minP < 1)
    assert _z(D - (-171.68), 1, tol=1e-2) and _z(minY - (-170.21), 1, tol=1e-2)
    print("    => negative pools + p outside (0,1): creation is INFEASIBLE here. OK")

    # --- a.5 (boundary map) rest-uniform family q = [Q, (1-Q)/(n-1) x (n-1)]:
    # scan confirms a SINGLE feasible->infeasible transition in Q (or none), then bisect.
    def qmax_rest_uniform(n):
        def qf(Q):
            return [Q] + [(1 - Q) / (n - 1)] * (n - 1)

        lo, hi = 1.0 / n, 1.0 - 1e-12
        assert feasible(qf(lo)), f"uniform must be feasible (n={n})"
        # single-transition check on a coarse scan
        flips = 0
        prev = True
        for k in range(1, 401):
            f = feasible(qf(lo + k * (hi - lo) / 400))
            flips += f != prev
            prev = f
        assert flips <= 1, f"non-monotone feasibility in Q at n={n}"
        if feasible(qf(hi)):
            return None  # feasible for ALL Q
        return _bisect(lambda Q: 1.0 if feasible(qf(Q)) else -1.0, lo, hi, 200)

    anchors = {30: 0.5859, 50: 0.3231, 100: 0.1567}
    print("  rest-uniform boundary q_max(n)  [dominant prob; rest uniform]:")
    print("    n    q_max(n)")
    for n in (3, 5, 10, 15, 20, 30, 50, 100):
        qm = qmax_rest_uniform(n)
        print(f"    {n:<4} {'1.0  (feasible for ALL q)' if qm is None else f'{qm:.4f}'}")
        if n in anchors:
            assert _z(qm - anchors[n], 1, tol=2e-3), f"q_max({n}) drifted: {qm}"
        else:
            assert qm is None, f"n={n} should be feasible everywhere in this family"
    # critical n for this family: n=20 has no boundary, n=21 does.
    assert qmax_rest_uniform(21) is not None
    print("    (boundary first appears at n = 21 in this family)")

    # non-dominant shapes:
    def qmax_two_big(n):
        def qf(Q):
            return [Q / 2, Q / 2] + [(1 - Q) / (n - 2)] * (n - 2)

        if feasible(qf(1.0 - 1e-12)):
            return None
        return _bisect(lambda Q: 1.0 if feasible(qf(Q)) else -1.0, 2.0 / n, 1.0 - 1e-12, 200)

    def rmin_geometric(n):
        def qf(r):
            w = np.array([r**i for i in range(n)])
            return list(w / w.sum())

        assert feasible(qf(1.0))
        if feasible(qf(1e-6)):
            return None
        return _bisect(lambda r: 1.0 if feasible(qf(r)) else -1.0, 1e-6, 1.0, 200)

    print("    two-big answers (combined mass Q_max):   "
          + ", ".join(f"n={n}: {qmax_two_big(n):.3f}" for n in (10, 30, 100)))
    print("    geometric decay (min feasible ratio r):  "
          + ", ".join(f"n={n}: {rmin_geometric(n):.3f}" for n in (10, 30, 100)))
    assert _z(qmax_two_big(30) - 0.445, 1, tol=2e-3)
    assert _z(rmin_geometric(30) - 0.8954, 1, tol=2e-3)
    print("  => the boundary is SHAPE-dependent (n=10 is safe rest-uniform but breaks under")
    print("     geometric decay r < 0.563); the exact test is the a.3 characterization. OK\n")


# ============================================================================
# GP19b — trading closure (unconditional)
# ============================================================================
def _buy_amount(Y, N, p, amount, outcome):
    """calculateCpmmShares mechanics (calculate-cpmm.ts:63-80): spend `amount`,
    new pool restores k = Y^p N^(1-p). Returns (shares, newY, newN)."""
    k = Y**p * N ** (1 - p)
    if outcome == "YES":
        nN = N + amount
        nY = (k / nN ** (1 - p)) ** (1 / p)
        return Y + amount - nY, nY, nN
    nY = Y + amount
    nN = (k / nY**p) ** (1 / (1 - p))
    return N + amount - nN, nY, nN


def _solve_u(P, Q, w, a, eta, iters=80):
    """Vectorized guarded Newton for the shares-in invariant (GP3: transcendental for
    general p, so numeric — GP5c licenses Newton). Solves, per element,
        w*ln(u + a) + (1-w)*ln(u) = w*ln(P) + (1-w)*ln(Q)
    for the SHRINKING reserve u, root in (max(0, -a), Q]. Substituting x = the
    boundary-hugging factor (x = u if a >= 0, else x = u + a) gives
        g(x) = alpha*ln(x) + beta*ln(x + c) = t,   c = |a| >= 0,
    and we solve in v = ln(x): both log arguments are then manifestly positive on the
    bracket (no cancellation at the domain edge) and the lower bracket
    v_lo = (t - beta*ln(x_hi + c))/alpha has g(v_lo) <= 0 PROVABLY (ln(e^v+c) is
    increasing and bounded by ln(x_hi+c) on the bracket)."""
    P, Q, w, a, eta = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (P, Q, w, a, eta))
    )
    t = w * np.log(P) + (1 - w) * np.log(Q)
    c = np.abs(a)
    swap = a < 0
    alpha = np.where(swap, w, 1 - w)  # coefficient on ln(x)
    beta = 1 - alpha
    # u = Q endpoint in the x coordinate: Q + a == P + eta ALGEBRAICALLY for both trade
    # directions (a is Q's deficit vs P plus eta) — use P + eta, which cannot cancel.
    x_hi = np.where(swap, P + eta, Q)
    assert np.all(x_hi > 0)
    hi = np.log(x_hi)  # g(hi) = w*ln((Q+|a| terms)/P) >= 0 (see callers)
    lo = np.minimum(hi, (t - beta * np.log(x_hi + c)) / alpha - 1e-9)
    v = hi.copy()
    with np.errstate(divide="raise", invalid="raise"):  # fail loudly on any NaN/inf
        for _ in range(iters):
            ev = np.exp(v)  # may underflow to 0.0 harmlessly (c > 0 there)
            arg = ev + c
            argp = arg > 0
            safe = np.where(argp, arg, 1.0)
            g = alpha * v + beta * np.where(argp, np.log(safe), v) - t
            pos = g > 0
            hi = np.where(pos, v, hi)
            lo = np.where(pos, lo, v)
            dg = alpha + beta * np.where(argp, ev / safe, 1.0)
            step = v - g / dg
            interior = (step > lo) & (step < hi)
            v = np.where(interior, step, 0.5 * (lo + hi))
    x = np.exp(v)
    u = np.where(swap, x - a, x)  # back to u (x - a = x + |a| when a < 0)
    # the GROWN reserve u + a: where a < 0 it equals x directly (cancellation-free —
    # recomputing u + a there can wipe out a tiny grown reserve); where a >= 0 both
    # terms are positive, so u + a is safe.
    grown = np.where(swap, x, u + np.maximum(a, 0.0))
    return u, grown


def _trade_shares(Y, N, p, eta, side):
    """Buy eta shares of `side` at general p; returns (newY, newN) on the invariant.
    Buy NO:  newN = N + C - eta shrinks; newY - newN = Y - N + eta =: a is constant.
    Buy YES: mirror (Y<->N, p<->1-p)."""
    Y, N, p = (np.asarray(x, dtype=float) for x in (Y, N, p))
    if side == "NO":
        u, grown = _solve_u(Y, N, p, Y - N + eta, eta)  # u = newN, grown = newY
        return grown, u
    u, grown = _solve_u(N, Y, 1 - p, N - Y + eta, eta)  # u = newY, grown = newN
    return u, grown


def _sell_yes(Y, N, p, d):
    """Convention-A sell of d YES shares == buy d NO + redeem d complete sets (GP9:
    identical cash and pool, any p). Returns (cash_received, newY, newN)."""
    nY, nN = _trade_shares(Y, N, p, d, "NO")
    cost = float(nN - N + d)  # C = newN - N + eta
    return d - cost, float(nY), float(nN)


def _auto_arb(pools, tol=1e-11):
    """Restore Sigma prob = 1 by buying equal shares eta of NO (Sigma>1) or YES (Sigma<1)
    in ALL answers — the vendor arb structure (GP5a: Sigma prob strictly monotone in eta
    => unique root). Returns new pools."""
    Y = np.array([pl["Y"] for pl in pools])
    N = np.array([pl["N"] for pl in pools])
    p = np.array([pl["p"] for pl in pools])
    S = float(np.sum(cpmm_prob(Y, N, p)))
    if abs(S - 1.0) < tol:
        return pools
    side = "NO" if S > 1.0 else "YES"

    def sigma_after(eta):
        nY, nN = _trade_shares(Y, N, p, eta, side)
        return float(np.sum(cpmm_prob(nY, nN, p)))

    # bracket eta: Sigma moves toward (and past) 1 monotonically as eta grows
    hi = 1e-6 * float(max(Y.max(), N.max()))
    for _ in range(200):
        if (sigma_after(hi) - 1.0) * (S - 1.0) <= 0:
            break
        hi *= 2.0
    else:
        raise AssertionError("auto-arb failed to bracket eta")
    eta = _bisect(lambda e: (sigma_after(e) - 1.0) * (1 if S > 1 else -1), 0.0, hi, 90)
    nY, nN = _trade_shares(Y, N, p, eta, side)
    return [
        {"Y": float(y), "N": float(nv), "p": float(pv)} for y, nv, pv in zip(nY, nN, p)
    ]


def GP19b_trading_closure():
    print("=" * 76)
    print("GP19b  trading closure — UNCONDITIONAL (k > 0 keeps reserves positive)")
    print("=" * 76)

    # --- b.1 (symbolic) buy either side, any amount A > 0, any p in (0,1):
    # newN = N + A > 0 trivially; newY = (k / newN^(1-p))^(1/p) is a positive power of
    # positive reals => > 0. Invariant preserved exactly (GP2). Sells are the convention-A
    # reverse (GP4) — the same one-parameter family along the invariant, so the same
    # positivity argument applies (numeric witness in the fuzz below).
    Y, N, A, p = symbols("Y N A p", positive=True)
    k = Y**p * N ** (1 - p)
    nN = N + A
    nY = (k / nN ** (1 - p)) ** (1 / p)
    resid = simplify(sp.powdenest(sp.together(simplify(nY**p * nN ** (1 - p) - k)), force=True))
    assert resid == 0, "invariant must be preserved exactly"
    assert nY.is_positive and nN.is_positive  # sympy positivity from positive symbols
    print("  buy: newN = N+A > 0, newY = (k/newN^(1-p))^(1/p) > 0, invariant residual 0.")
    print("  (NO-side buy is the (Y,N,p)->(N,Y,1-p) mirror; sells are the GP4 reverse.)")
    print("  => the CPMM never leaves the open positive quadrant: k > 0 is closed. OK")

    # --- b.2 (symbolic) the prob clamp bounds distance from the boundary.
    # rho := N/Y is pinned by prob: prob/(1-prob) = pN/((1-p)Y) => rho = ((1-p)/p) odds.
    # On the invariant, Y = k rho^(p-1), N = k rho^p — so prob in [0.01, 0.99]
    # (market orders are capped by an implicit limit at MIN/MAX_CPMM_PROB,
    #  calculate-cpmm.ts:250-257 via new-bet.ts:121) gives EXPLICIT reserve floors.
    prob = p * N / ((1 - p) * Y + p * N)
    rho = symbols("rho", positive=True)
    odds_id = simplify(prob / (1 - prob) - p * N / ((1 - p) * Y))
    assert odds_id == 0
    Y_of_rho = k.subs(N, rho * Y)  # = Y * rho^(1-p)
    assert simplify(Y_of_rho - Y * rho ** (1 - p)) == 0
    print("  prob/(1-prob) = (p/(1-p))(N/Y); on the invariant Y = k rho^(p-1), N = k rho^p.")
    rng = np.random.default_rng(20)
    for _ in range(300):
        pv = float(rng.uniform(0.02, 0.98))
        kv = 10 ** float(rng.uniform(-2, 3))
        pr = float(rng.uniform(MIN_CPMM_PROB, MAX_CPMM_PROB))
        rv = (1 - pv) / pv * pr / (1 - pr)
        Yv, Nv = kv * rv ** (pv - 1), kv * rv**pv
        r1 = (1 - pv) / pv * MIN_CPMM_PROB / (1 - MIN_CPMM_PROB)
        r2 = (1 - pv) / pv * MAX_CPMM_PROB / (1 - MAX_CPMM_PROB)
        assert Yv >= kv * r2 ** (pv - 1) * (1 - 1e-12), "Y floor"
        assert Nv >= kv * r1**pv * (1 - 1e-12), "N floor"
        assert _z(cpmm_prob(Yv, Nv, pv) - pr, 1)
    print("  clamped prob => Y >= k*rho_max^(p-1), N >= k*rho_min^p (300 samples) ... OK")

    # --- b.3 (numeric fuzz) random sane states, adversarial included; random
    # buy/sell sequences with the Sigma=1 auto-arb after each op. States are built
    # from (q_i, p_i, scale): rho = N/Y = ((1-p)/p)(q/(1-q)) pins prob_i = q_i exactly
    # (the b.2 identity), so Sigma q = 1 at start, p is adversarial by construction,
    # and reserves span tiny..huge. p in [0.02, 0.98] + relative trade sizes keep the
    # closed form inside float64 range (the REAL-arithmetic claim is unconditional;
    # see the underflow caveat demo below for what float64 adds).
    rng = np.random.default_rng(21)
    n_states, n_ops = 24, 3
    worst_sigma = 0.0
    for t in range(n_states):
        n = int(rng.choice([3, 5, 20, 50, 100]))
        # target probs: sometimes dominant (0.9 mass on one answer), else Dirichlet-ish
        if rng.random() < 0.4:
            rest = rng.exponential(1.0, n - 1) + 1e-9
            qv = np.concatenate([[9.0 * rest.sum()], rest])
        else:
            qv = rng.exponential(1.0, n) + 1e-9
        qv = qv / qv.sum()
        qv = np.clip(qv, 1e-4, 1 - 1e-4)
        qv = qv / qv.sum()
        pools = []
        for qi in qv:
            qi = float(qi)
            for _ in range(100):  # rejection-sample to keep reserves representable
                pv = float(rng.uniform(0.02, 0.98))
                Yv = 10 ** float(rng.uniform(-3, 3))
                Nv = Yv * ((1 - pv) / pv) * (qi / (1 - qi))  # prob_i = q_i exactly
                if 1e-6 <= Nv <= 1e6:
                    break
            pools.append({"Y": Yv, "N": Nv, "p": pv})
            assert _z(cpmm_prob(Yv, Nv, pv) - qi, 1)
        pools = _auto_arb(pools)  # construction sits on Sigma=1; this only tightens fp dust
        p_orig = [pl["p"] for pl in pools]
        held = {}  # answer -> YES shares we hold (for sells)
        for op in range(n_ops):
            i = int(rng.integers(0, n))
            pl = pools[i]
            if op == n_ops - 1 and i in held and held[i] > 0:
                # SELL half our held YES shares (convention-A reverse == GP9 buy-NO+redeem)
                d = held[i] / 2
                R, nY, nN = _sell_yes(pl["Y"], pl["N"], pl["p"], d)
                assert R > 0 and nY > 0 and nN > 0, "sell must keep reserves positive"
                pools[i] = {"Y": nY, "N": nN, "p": pl["p"]}
                held[i] -= d
            else:
                side = "YES" if rng.random() < 0.7 else "NO"
                amt = (pl["Y"] + pl["N"]) * 10 ** float(rng.uniform(-3, 0.5))
                # vendor market orders stop at the prob clamp (an implicit limit at
                # MIN/MAX_CPMM_PROB, new-bet.ts:121) — cap the taker fill faithfully.
                cap = calc_amount_to_prob(
                    pl["Y"], pl["N"], pl["p"],
                    MAX_CPMM_PROB if side == "YES" else MIN_CPMM_PROB, side)
                if cap <= 0:  # already at/past the clamp (arb legs are unclamped)
                    continue
                amt = min(amt, cap)
                sh, nY, nN = _buy_amount(pl["Y"], pl["N"], pl["p"], amt, side)
                assert nY > 0 and nN > 0 and sh > 0, "buy must keep reserves positive"
                pools[i] = {"Y": nY, "N": nN, "p": pl["p"]}
                if side == "YES":
                    held[i] = held.get(i, 0.0) + sh
            pools = _auto_arb(pools)
            assert all(pl["Y"] > 0 and pl["N"] > 0 for pl in pools), "arb reserves"
            assert [pl["p"] for pl in pools] == p_orig, "trades/arb never touch p"
            sig = sum(cpmm_prob(pl["Y"], pl["N"], pl["p"]) for pl in pools)
            worst_sigma = max(worst_sigma, abs(sig - 1.0))
            assert _z(sig - 1.0, 1, tol=1e-8), f"Sigma q = 1 restored (got {sig})"
    print(f"  fuzz {n_states} states x {n_ops} ops (n up to 100, reserves 1e-3..1e3+,")
    print("  dominant probs, p in [0.02,0.98], trades up to ~3x pool, taker fills")
    print("  clamp-capped as vendor's): reserves > 0 always, p untouched,")
    print(f"  worst |Sigma q - 1| after arb = {worst_sigma:.2e} ... OK")

    # --- b.4 (float64 caveat — found while proving) In REAL arithmetic closure is
    # unconditional; in float64 (vendor TypeScript Number too) the closed form
    # newY = (k/newN^(1-p))^(1/p) UNDERFLOWS TO EXACTLY 0 when p is extreme:
    # the 1/p exponent turns a ratio-just-below-1 into e^(-huge). Demonstrate:
    sh, nY, nN = _buy_amount(1000.0, 1000.0, 1e-9, 1000.0, "YES")
    assert nY == 0.0, "expected float64 underflow at p = 1e-9"
    print("  CAVEAT: at p = 1e-9 a pool-doubling buy underflows newY to exactly 0.0 in")
    print("  float64 (vendor shares this) — a representability failure, not a math one.")
    print("  Runtime guards should keep p away from {0,1}; the GP6a repricing only")
    print("  produces extreme p from extreme reserve/prob combinations.")
    print("  => TRADING IS UNCONDITIONALLY CLOSED on sane states (real arithmetic);")
    print("     float64 adds a p-representability caveat, demonstrated above.\n")


# ============================================================================
# GP19c — whole-market add: the CONDITIONAL closure (headline)
# ============================================================================
def GP19c_whole_market_add():
    print("=" * 76)
    print("GP19c  whole-market add: conditional closure with critical amount A*")
    print("=" * 76)

    # --- c.i  DeltaN_i >= 0 ALWAYS; only DeltaY_i can go negative, exactly when q
    #          is creation-infeasible. (Same root-sign argument as GP19a.2: the
    #          quadratic root is nonneg since sqrt(b^2+c) >= |b| for c >= 0.)
    rng = np.random.default_rng(22)
    saw_negY_infeasible, saw_all_posY_feasible = 0, 0
    for _ in range(2000):
        n = int(rng.integers(3, 101))
        if rng.random() < 0.5:
            Q = float(rng.uniform(1.0 / n, 0.999))
            qv = [Q] + [(1 - Q) / (n - 1)] * (n - 1)
        else:
            w = rng.exponential(1.0, n) + 1e-12
            qv = list(w / w.sum())
        delta = sum_to_one_pools(qv, 1.0)
        assert all(dl["N"] >= 0 for dl in delta), "DeltaN_i >= 0 must hold always"
        negY = any(dl["Y"] < 0 for dl in delta)
        assert negY == (not feasible(qv)), "DeltaY<0 somewhere <=> q infeasible"
        saw_negY_infeasible += negY
        saw_all_posY_feasible += not negY
    assert saw_negY_infeasible > 100 and saw_all_posY_feasible > 100  # both branches hit
    print("  fuzz 2000 q: DeltaN_i >= 0 in all; DeltaY_i < 0 somewhere <=> q infeasible")
    print(f"  (both branches exercised: {saw_negY_infeasible} infeasible / "
          f"{saw_all_posY_feasible} feasible) ... OK")

    # --- c.ii  A* = min_{i: dY_i < 0} Y_i / |dY_i(q, ante=1)|.
    # A sane market CAN sit at infeasible q — trades move probs freely (GP19b). Build one:
    # arbitrary positive reserves at the infeasible q from GP19a.4, p via GP6a repricing.
    n = 30
    qv = [0.9] + [0.1 / 29] * 29
    assert not feasible(qv)
    pools = []
    for i, qi in enumerate(qv):
        Yv = 10 ** float(rng.uniform(0.5, 2.5))
        Nv = 10 ** float(rng.uniform(0.5, 2.5))
        pools.append({"Y": Yv, "N": Nv, "p": weight_for_prob(Yv, Nv, qi)})
    assert sane(pools)
    unit = sum_to_one_pools(qv, 1.0)
    dY = [dl["Y"] for dl in unit]
    A_star = min(pl["Y"] / -dYi for pl, dYi in zip(pools, dY) if dYi < 0)
    print(f"  sane traded state at infeasible q (n=30, dominant 0.90): A* = {A_star:.3f}")

    below = whole_market_add(pools, 0.999 * A_star)
    assert sane(below), "A < A*: merged market must be sane"
    for pl, mg, qi in zip(pools, below, qv):
        assert _z(cpmm_prob(mg["Y"], mg["N"], mg["p"]) - qi, 1), "probs preserved below A*"
    above = whole_market_add(pools, 1.001 * A_star)
    assert min(pl["Y"] for pl in above) < 0, "A > A*: some merged poolYes < 0"
    assert not sane(above), "A > A*: sanity must break"
    print("  add(0.999*A*): all pools > 0, p in (0,1), every prob preserved ... OK")
    print("  add(1.001*A*): merged poolYes < 0 => p outside (0,1) => INSANE ... OK")

    # --- c.iii  homogeneity => the bound is on TOTAL added liquidity, independent of
    # drip size. Drizzle in 100 ticks == one lump (probs preserved each tick => every
    # tick's Delta has the same shape; GP17a additivity sums them exactly).
    T = 0.999 * A_star
    drizzle = [dict(pl) for pl in pools]
    for _ in range(100):
        drizzle = whole_market_add(drizzle, T / 100)
        assert sane(drizzle), "drizzle below A* must stay sane at every tick"
    lump = whole_market_add(pools, T)
    for dz, lp in zip(drizzle, lump):
        assert _z(dz["Y"] - lp["Y"], lp["Y"], tol=1e-7), "drizzle == lump (Y)"
        assert _z(dz["N"] - lp["N"], lp["N"], tol=1e-7), "drizzle == lump (N)"
        assert _z(dz["p"] - lp["p"], 1, tol=1e-7), "drizzle == lump (p)"
    print("  100-tick drizzle of 0.999*A* == one lump, reservewise (rel 1e-7) ... OK")

    # breaching drizzle: first insane tick k satisfies (k-1)*t <= A* < k*t — the SAME
    # total as the lump bound, for two very different drip sizes.
    for ticks in (7, 100):
        t = 1.001 * A_star / ticks
        state = [dict(pl) for pl in pools]
        breach_at = None
        for kk in range(1, ticks + 1):
            state = whole_market_add(state, t)
            if not sane(state):
                breach_at = kk
                break
        assert breach_at is not None, "1.001*A* total must breach"
        assert (breach_at - 1) * t <= A_star < breach_at * t, (
            f"breach tick {breach_at} inconsistent with A* (drip {t:.3f})"
        )
    print("  breaching drizzle (7 ticks and 100 ticks of 1.001*A* total): breach lands")
    print("  exactly where cumulative > A* — drip size is irrelevant, only the TOTAL. OK")

    # --- c.iv  corollary: feasible q => all dY_i > 0 => A* = infinity.
    qf = [0.4, 0.3, 0.2, 0.1]
    assert feasible(qf)
    pools_f = []
    for qi in qf:
        Yv, Nv = 10 ** float(rng.uniform(0, 2)), 10 ** float(rng.uniform(0, 2))
        pools_f.append({"Y": Yv, "N": Nv, "p": weight_for_prob(Yv, Nv, qi)})
    assert all(dl["Y"] > 0 for dl in sum_to_one_pools(qf, 1.0))
    assert sane(whole_market_add(pools_f, 1e9)), "feasible q: any A stays sane"
    print("  feasible q: dY_i > 0 for all i => A* = inf; add(1e9) still sane ... OK")
    print("  => ADD CLOSURE IS CONDITIONAL: sane for TOTAL added < A*(state); needs a guard.\n")


# ============================================================================
# GP19d — Other-split closure
# ============================================================================
def GP19d_other_split():
    print("=" * 76)
    print("GP19d  Other-split closure: sane per split, geometric walk to the 1% floor")
    print("=" * 76)

    # --- d.1 (symbolic) pools {YES: Yo+a, NO: a} positive for Yo > 0, a > 0 (trivial);
    # p* in (0,1) and displayed prob == targetProb by GP6a, targetProb = probOther/2
    # in (0, 1/2) whenever probOther in (0,1).
    Yo, a, q = symbols("Yo a q", positive=True)
    Yp, Np = Yo + a, a
    p_star = weight_for_prob(Yp, Np, q)
    assert simplify(cpmm_prob(Yp, Np, p_star) - q) == 0
    one_minus = simplify(1 - p_star - (1 - q) * Np / (q * Yp + (1 - q) * Np))
    assert one_minus == 0  # 1-p* is a ratio of positives => p* in (0,1)
    print("  pools (Yo+a, a) > 0; p* = GP6a weight: displayed prob == q exactly,")
    print("  p* and 1-p* both ratios of positives => p* in (0,1). Sane per split. OK")

    # numeric: one split on a concrete Other, incl. extreme probOther
    for Yov, Nov, pOv in ((500.0, 500.0, 0.5), (40.0, 4.0, 0.5), (1.0, 99.0, 0.5),
                          (30.0, 70.0, 0.83)):
        pa, po, tp = other_split_v2(Yov, Nov, pOv, answerCost=100.0)
        assert sane_answer(pa) and sane_answer(po)
        assert _z(cpmm_prob(pa["Y"], pa["N"], pa["p"]) - tp, 1)
        assert _z(cpmm_prob(po["Y"], po["N"], po["p"]) - tp, 1)
        assert _z(tp - cpmm_prob(Yov, Nov, pOv) / 2, 1)
    print("  numeric: A and Other' both sane, both at exactly probOther/2 ... OK")

    # --- d.2 (boundary walk) repeated splits halve probOther: 0.5 -> 0.25 -> ... .
    # Strict sanity (pools > 0, p in (0,1)) NEVER breaks — the walk violates the
    # MIN_CPMM_PROB = 0.01 floor (contract.ts:519): answers land below the market-order
    # clamp, i.e. below any tradeable price.
    Yv, Nv, pv = 500.0, 500.0, 0.5  # Other at 50%
    prob_other = cpmm_prob(Yv, Nv, pv)
    assert _z(prob_other - 0.5, 1)
    survived = 0
    child_probs = []
    while True:
        pa, po, tp = other_split_v2(Yv, Nv, pv, answerCost=100.0)
        assert sane_answer(pa) and sane_answer(po), "strict sanity holds every split"
        child_probs.append(tp)
        if tp < MIN_CPMM_PROB:
            break
        survived += 1
        Yv, Nv, pv = po["Y"], po["N"], po["p"]  # Other' becomes the next Other
    print("  walk from Other=0.50 (child prob per split): "
          + ", ".join(f"{x:.6f}" for x in child_probs))
    assert survived == 5, f"expected 5 surviving splits, got {survived}"
    assert _z(child_probs[5] - 0.5 / 2**6, 1)  # 0.0078125 < 0.01
    print(f"  survives {survived} splits; split 6 lands both children at "
          f"{child_probs[5]:.7f} < MIN_CPMM_PROB = {MIN_CPMM_PROB}")
    print("  => PRECONDITION for a split: probOther >= 2*MIN_CPMM_PROB (both children")
    print("     >= the floor). Equivalently k_max = floor(log2(probOther/MIN_CPMM_PROB))")
    print("     splits from a fresh Other. Needs a runtime floor check. OK\n")


# ============================================================================
# GP19e — per-answer independent add + v1 -> v2 conversion
# ============================================================================
def GP19e_per_answer_add_and_conversion():
    print("=" * 76)
    print("GP19e  per-answer lossless add + v1->v2 conversion: sane -> sane")
    print("=" * 76)

    # --- e.1 (symbolic) addCpmmLiquidity's floated p IS the GP6a weight of the grown
    # pool: newP = q(Y+a) / (q(Y+a) + (1-q)(N+a)) with q the pre-add prob. Hence
    # newP in (0,1) for ANY positive reserves/amount (GP17b's hypothesis, explicit),
    # and the displayed prob is exactly preserved.
    Y, N, a, p = symbols("Y N a p", positive=True)
    prob = cpmm_prob(Y, N, p)
    newP_vendor = (prob * (a + Y)) / (a - N * (prob - 1) + prob * Y)  # calculate-cpmm.ts:753-755
    newP_gp6a = weight_for_prob(Y + a, N + a, prob)
    assert simplify(newP_vendor - newP_gp6a) == 0
    assert simplify(cpmm_prob(Y + a, N + a, newP_vendor) - prob) == 0
    print("  newP(vendor) == GP6a weight of (Y+a, N+a) at the old prob (symbolic),")
    print("  => newP in (0,1) (ratio of positives) and prob preserved exactly. OK")

    # numeric fuzz incl. adversarial pools
    rng = np.random.default_rng(23)
    for _ in range(500):
        Yv = 10 ** float(rng.uniform(-4, 4))
        Nv = 10 ** float(rng.uniform(-4, 4))
        pv = float(rng.uniform(0.001, 0.999))
        amt = 10 ** float(rng.uniform(-3, 4))
        nY, nN, nP = add_cpmm_liquidity(Yv, Nv, pv, amt)
        assert nY > 0 and nN > 0 and 0 < nP < 1
        assert _z(cpmm_prob(nY, nN, nP) - cpmm_prob(Yv, Nv, pv), 1)
    print("  fuzz 500 adversarial pools (reserves 1e-4..1e4, p 0.001..0.999): sane. OK")

    # --- e.2 conversion v1 -> v2 is the identity embedding: a v1 answer stores positive
    # pools and no per-answer p; v2 reads p ?? 0.5 (e.g. create-answer-cpmm.ts:494).
    # p = 0.5 in (0,1) and reserves are unchanged, so sane -> sane trivially, and the
    # displayed prob is the v1 prob N/(Y+N) (GP1 reduction).
    for Yv, Nv in ((500.0, 100.0), (3.0, 3.0), (0.02, 40.0)):
        v2 = {"Y": Yv, "N": Nv, "p": 0.5}
        assert sane_answer(v2)
        assert _z(cpmm_prob(Yv, Nv, 0.5) - Nv / (Yv + Nv), 1)
    print("  v1->v2 = identity on pools + p=0.5: sane -> sane, prob = N/(Y+N). OK\n")


if __name__ == "__main__":
    GP19a_creation_feasibility()
    GP19b_trading_closure()
    GP19c_whole_market_add()
    GP19d_other_split()
    GP19e_per_answer_add_and_conversion()
    print("All GP19 sanity-closure theorems verified.")
