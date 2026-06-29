#!/usr/bin/env python3
"""
GP10 — the three auto-arb parameterizations trace ONE equilibrium curve.

The paper (auto-arb-algorithms.tex) gives three strategies that differ in what is
specified (cost / shares / target-prob) AND in their redemption decomposition:

  - Dollar-centric : arb NO in OTHERS (n-1); redemption eta*(n-2).
  - Share-centric  : arb NO in ALL (n);     redemption eta*(n-1).
  - Probability    : arb NO in OTHERS (n-1); redemption eta*(n-2).

GP10 claim: on the sum-to-one domain (Sigma p = 1), all three produce the SAME
(shares, cost, target-prob) triple AND the SAME final pools when given consistent
inputs. So they are three parameterizations of ONE trade, not three trades — the
"different redemption" bookkeeping is a relabeling, not a different equilibrium.

No vendor oracle exists for the closed forms beyond p=1/2 differential tests
(done in the library: tests/test_slice4_parameterizations.py); here we verify the
INTERNAL identity numerically over a fuzz of normalized markets, both positions,
n in 2..5. Self-contained (no project imports) so it lifts cleanly into the PR.

Run: python3 parameterization_equivalence.py
"""

import math
import random

_SIGMA_TOL = 1e-13
_MAX_BISECT = 200


# --- p=1/2 single-pool closed forms (fresh; CPMM invariant Y*N = k) ---------- #
def buy_amount(Y, N, A, pos):
    k = Y * N
    if pos == "YES":
        n = N + A
        y = k / n
        s = Y + A - y
    else:
        y = Y + A
        n = k / y
        s = N + A - n
    return s, {"YES": y, "NO": n}


def cost_for_shares(Y, N, s, pos):
    t = Y + N - s
    if pos == "YES":
        c = (s - Y - N + math.sqrt(t * t + 4.0 * N * s)) / 2.0
        return c, {"YES": Y + c - s, "NO": N + c}
    c = (s - Y - N + math.sqrt(t * t + 4.0 * Y * s)) / 2.0
    return c, {"YES": Y + c, "NO": N + c - s}


def shares_for_prob(Y, N, pt, pos):
    k = Y * N
    if pos == "YES":
        A = math.sqrt(pt * k / (1.0 - pt)) - N
        return Y + A - k / (N + A)
    A = math.sqrt((1.0 - pt) * k / pt) - Y
    return N + A - k / (Y + A)


def prob_yes(P):
    return P["NO"] / (P["YES"] + P["NO"])


def sigma(ps):
    return sum(prob_yes(p) for p in ps.values())


def bisect_sigma(state_fn, hint):
    """Find eta >= 0 with Sigma p(eta) = 1 by sign-bracketed bisection."""
    def f(e):
        return sigma(state_fn(e)[1]) - 1.0

    f0 = f(0.0)
    if abs(f0) < _SIGMA_TOL:
        return 0.0
    lo, hi = 0.0, max(1.0, abs(hint))
    fh = f(hi)
    grow = 0
    while f0 * fh > 0.0 and grow < 60:
        hi *= 2.0
        fh = f(hi)
        grow += 1
    if f0 * fh > 0.0:
        return 0.0
    eta = 0.0
    for _ in range(_MAX_BISECT):
        m = 0.5 * (lo + hi)
        if m in (lo, hi):
            break
        fm = f(m)
        if abs(fm) < _SIGMA_TOL:
            return m
        if (fm > 0.0) == (f0 > 0.0):
            lo = m
        else:
            hi = m
        eta = m
    return eta


# --- the three parameterizations -------------------------------------------- #
def dollar(P, k, A, pos):
    others = [a for a in P if a != k]
    n = len(P)
    arb = "NO" if pos == "YES" else "YES"

    def state(e):
        np = {}
        tot = 0.0
        for a in others:
            Y, N = P[a]["YES"], P[a]["NO"]
            if e > 0.0:
                c, p = cost_for_shares(Y, N, e, arb)
                np[a] = p
                tot += c
            else:
                np[a] = dict(P[a])
        red = e * (n - 2) if pos == "YES" else 0.0
        prim = A - (tot - red)
        Y, N = P[k]["YES"], P[k]["NO"]
        if prim > 0:
            ps, tp = buy_amount(Y, N, prim, pos)
        else:
            ps, tp = 0.0, {"YES": Y, "NO": N}
        np[k] = tp
        return ps + e, np

    e = bisect_sigma(state, abs(A))
    s, np = state(e)
    return s, A, prob_yes(np[k]), np


def share(P, k, s, pos):
    n = len(P)
    arb = "NO" if pos == "YES" else "YES"
    Y, N = P[k]["YES"], P[k]["NO"]
    cp, kp = cost_for_shares(Y, N, s, pos)
    base = {a: (kp if a == k else P[a]) for a in P}

    def state(e):
        np = {}
        tot = 0.0
        for a, pool in base.items():
            Y, N = pool["YES"], pool["NO"]
            if e > 0.0:
                c, p = cost_for_shares(Y, N, e, arb)
                np[a] = p
                tot += c
            else:
                np[a] = dict(pool)
        return tot, np

    e = bisect_sigma(lambda x: (None, state(x)[1]), abs(s))
    ca, np = state(e)
    red = e * (n - 1) if pos == "YES" else e
    return s, cp + ca - red, prob_yes(np[k]), np


def probc(P, k, pt, pos):
    others = [a for a in P if a != k]
    n = len(P)
    arb = "NO" if pos == "YES" else "YES"
    Y, N = P[k]["YES"], P[k]["NO"]
    d = shares_for_prob(Y, N, pt, pos)
    cp, kp = cost_for_shares(Y, N, d, pos)

    def state(e):
        np = {k: kp}
        tot = 0.0
        for a in others:
            Y, N = P[a]["YES"], P[a]["NO"]
            if e > 0.0:
                c, p = cost_for_shares(Y, N, e, arb)
                np[a] = p
                tot += c
            else:
                np[a] = dict(P[a])
        return tot, np

    e = bisect_sigma(lambda x: (None, state(x)[1]), 1.0)
    ca, np = state(e)
    red = e * (n - 2) if pos == "YES" else 0.0
    return d + e, cp + ca - red, prob_yes(np[k]), np


def theorem_GP10_parameterizations_agree(trials=5000, seed=10):
    print("=" * 70)
    print("GP10: dollar / share / prob parameterizations trace ONE curve (p=1/2)")
    print("=" * 70)
    rng = random.Random(seed)
    worst_triple = 0.0
    worst_pool = 0.0
    compared = 0
    for _ in range(trials):
        n = rng.randint(2, 5)
        raw = [rng.uniform(0.05, 0.95) for _ in range(n)]
        tot = sum(raw)
        probs = [r / tot for r in raw]
        P = {}
        for i, q in enumerate(probs):
            N = rng.uniform(50.0, 300.0)
            P[f"a{i}"] = {"YES": N * (1 - q) / q, "NO": N}
        k = f"a{rng.randrange(n)}"
        pos = rng.choice(["YES", "NO"])
        A = rng.uniform(1.0, 150.0)

        sd, cd, pd, npd = dollar(P, k, A, pos)
        if sd <= 1e-9 or not (0.02 < pd < 0.98):
            continue
        ss, cs, ps, nps = share(P, k, sd, pos)
        sp, cp, pp, npp = probc(P, k, pd, pos)
        compared += 1

        # Triple cross-agreement: share recovers (c, p); prob recovers (s, c).
        triple = max(abs(cs - cd), abs(ps - pd), abs(sp - sd), abs(cp - cd))
        # Final pools identical across all three.
        pool = 0.0
        for a in P:
            pool = max(
                pool,
                abs(nps[a]["YES"] - npd[a]["YES"]),
                abs(npp[a]["YES"] - npd[a]["YES"]),
            )
        worst_triple = max(worst_triple, triple)
        worst_pool = max(worst_pool, pool)

    print(f"  compared {compared} normalized markets (n in 2..5, both positions)")
    print(f"  worst (shares,cost,prob) cross-agreement dev = {worst_triple:.2e}")
    print(f"  worst final-pool dev across the three         = {worst_pool:.2e}")
    assert worst_triple < 1e-6, f"triple dev too large: {worst_triple}"
    assert worst_pool < 1e-6, f"pool dev too large: {worst_pool}"
    print("  => one trade, three parameterizations. GP10 holds.\n")


if __name__ == "__main__":
    theorem_GP10_parameterizations_agree()
    print("GP10 verified.")
