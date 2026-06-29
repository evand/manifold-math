#!/usr/bin/env python3
"""
Creation-liquidity theorems (cpmm-multi-2) — symbolic derivation.

GP13  Point liquidity (pre-auto-arb, one pool):
        a = dprob/dshares = p(1-p)·Y·N / W^3 = q(1-q)/W,  W = (1-p)Y + pN,
      and with p eliminated at fixed prob q:  a = q(1-q)(qY+(1-q)N)/(YN).
      Symmetric under (Y,N,p) -> (N,Y,1-p). ⇒ at fixed prob, max liquidity ⟺ max W.

GP14  Point liquidity (post-auto-arb, single-answer YES buy):
        a_i^eff = a_i (Σ_j a_j - a_i) / Σ_j a_j,
      from the η complete-set-of-NO redemption that restores Σ q = 1.

GP15  Basket-funded creation optimum:
        (a) all-winners-tight funding ⟹ Y_i - N_i = D (const);
        (b) minimize Σ c_i/W_i under a budget on Σ W_i ⟹ W_i ∝ √c_i, c_i = q_i(1-q_i)
            = Var(Bernoulli(q_i)) ⇒ the "depth ∝ outcome standard deviation" rule.

Mechanically verifiable: run it; each claim asserts its identity. Same idiom as
general_p_cost.py. Numerical accuracy of GP14/GP15 is checked in
creation_liquidity_optimum.py (a_eff to 7e-4) and creation_liquidity_solver.py
(√c-depth captures 95.5-99.7% of the gain).
"""

import sympy as sp
from sympy import Rational, powdenest, simplify, sqrt, symbols

Y, N, A = symbols("Y N A", positive=True, real=True)
p, q = symbols("p q", positive=True, real=True)


def _zero(expr):
    e = powdenest(sp.together(simplify(expr)), force=True)
    return simplify(powdenest(e, force=True))


def theorem_GP13_point_liquidity():
    print("=" * 70)
    print("GP13: point liquidity a = dprob/dshares = p(1-p)YN/W^3 = q(1-q)/W")
    print("=" * 70)
    # Buy YES with amount A (GP2 mechanic): N_new = N+A, Y_new restores invariant.
    k = Y**p * N ** (1 - p)
    N_new = N + A
    Y_new = (k / N_new ** (1 - p)) ** (1 / p)
    shares = A + (Y - Y_new)
    W_new = (1 - p) * Y_new + p * N_new
    prob_new = p * N_new / W_new

    # a = dprob/dshares at the margin (A -> 0) = (dprob/dA)/(dshares/dA).
    dprob = sp.diff(prob_new, A).subs(A, 0)
    dshares = sp.diff(shares, A).subs(A, 0)
    a = simplify(dprob / dshares)

    W = (1 - p) * Y + p * N
    a_form1 = p * (1 - p) * Y * N / W**3
    print("  claim a = p(1-p)YN/W^3 ...", end=" ")
    assert _zero(a - a_form1) == 0
    print("OK")

    # a = q(1-q)/W with q = pN/W.
    qexpr = p * N / W
    a_form2 = qexpr * (1 - qexpr) / W
    print("  claim a = q(1-q)/W      ...", end=" ")
    assert _zero(a - a_form2) == 0
    print("OK")

    # p eliminated at fixed prob: p = qY/(qY+(1-q)N) -> a = q(1-q)(qY+(1-q)N)/(YN).
    p_of_q = q * Y / (q * Y + (1 - q) * N)
    a_elim = a_form1.subs(p, p_of_q)
    a_qYN = q * (1 - q) * (q * Y + (1 - q) * N) / (Y * N)
    print("  claim a = q(1-q)(qY+(1-q)N)/(YN) ...", end=" ")
    assert _zero(a_elim - a_qYN) == 0
    print("OK")

    # p = 1/2 reduction: a = 2YN/(Y+N)^3.
    a_half = a_form1.subs(p, Rational(1, 2))
    print("  p=1/2 reduction = 2YN/(Y+N)^3 ...", end=" ")
    assert _zero(a_half - 2 * Y * N / (Y + N) ** 3) == 0
    print("OK")

    # YES/NO symmetry in shares: a(Y,N,p) == a(N,Y,1-p)  (buying NO is symmetric).
    a_swapped = a_form1.subs({Y: N, N: Y, p: 1 - p}, simultaneous=True)
    print("  shares symmetry a(Y,N,p)=a(N,Y,1-p) ...", end=" ")
    assert _zero(a_form1 - a_swapped) == 0
    print("OK")

    print("  ⇒ at fixed prob q, a = q(1-q)/W ⇒ MAX liquidity ⟺ MAX W=(1-p)Y+pN.\n")


def theorem_GP14_post_arb_effective():
    print("=" * 70)
    print("GP14: post-arb effective slope a_i^eff = a_i(Σa - a_i)/Σa")
    print("=" * 70)
    # A YES buy of s shares on answer i raises q_i by a_i*s. The auto-arb buys η
    # NO equally on all answers (complete-set redemption) to restore Σq=1; buying
    # NO lowers q_j by a_j*η (|dq/d(NO share)| = a_j by GP13 symmetry).
    n = 3
    a = symbols("a0:%d" % n, positive=True)
    s, eta = symbols("s eta", positive=True)
    i = 0
    # Net change per answer; Σ over all must be 0 (Σq stays 1).
    dq = [(-a[j] * eta) for j in range(n)]
    dq[i] += a[i] * s
    total = sum(dq)
    eta_sol = sp.solve(sp.Eq(total, 0), eta)[0]
    a_eff = simplify(dq[i].subs(eta, eta_sol) / s)
    S = sum(a)
    claim = a[i] * (S - a[i]) / S
    print(f"  η = {eta_sol}")
    print("  claim a_i^eff = a_i(Σa - a_i)/Σa ...", end=" ")
    assert simplify(a_eff - claim) == 0
    print("OK")
    print("  (numeric vs the auto-arb solver: 7e-4, see creation_liquidity_optimum.py)\n")


def theorem_GP15_optimum_structure():
    print("=" * 70)
    print("GP15: basket optimum — all-tight ⟹ Y_i-N_i const; depth ∝ √variance")
    print("=" * 70)
    # (a) All-winners-tight: worst_k = Y_k + Σ_{j≠k} N_j = ante for every k.
    #     Subtract scenario k from scenario m: (Y_k - N_k) - (Y_m - N_m) = 0.
    Yk, Nk, Ym, Nm, S_N, ante = symbols("Yk Nk Ym Nm S_N ante", positive=True)
    worst_k = Yk + (S_N - Nk)  # S_N = Σ N_j
    worst_m = Ym + (S_N - Nm)
    diff = simplify((worst_k - ante) - (worst_m - ante))  # both = 0 at the optimum
    print("  worst_k - worst_m = (Y_k-N_k) - (Y_m-N_m) =", diff)
    assert simplify(diff - ((Yk - Nk) - (Ym - Nm))) == 0
    print("  ⇒ all worst-cases equal ⟺ Y_i - N_i = const =: D.  OK")

    # (b) Minimize Σ c_i/W_i subject to Σ W_i = B (budget surrogate for capital).
    #     Lagrange: d/dW_i [ c_i/W_i + λ W_i ] = 0 ⇒ W_i = sqrt(c_i/λ) ∝ sqrt(c_i).
    c_i, W_i, lam = symbols("c_i W_i lam", positive=True)
    L = c_i / W_i + lam * W_i
    statio = sp.solve(sp.Eq(sp.diff(L, W_i), 0), W_i)
    W_star = [w for w in statio if w.is_positive][0]
    print(f"  stationary W_i = {W_star}   (∝ sqrt(c_i))")
    assert _zero(W_star - sqrt(c_i / lam)) == 0
    # c_i = q(1-q) is exactly the Bernoulli variance of the outcome.
    var_bernoulli = q * (1 - q)
    print(f"  c_i = q(1-q) = Var(Bernoulli(q)) = {var_bernoulli}")
    print("  ⇒ W_i ∝ √(q_i(1-q_i)) : allocate DEPTH ∝ outcome standard deviation.")
    print("  (numeric: captures 95.5-99.7% of the gain, creation_liquidity_solver.py)\n")


if __name__ == "__main__":
    theorem_GP13_point_liquidity()
    theorem_GP14_post_arb_effective()
    theorem_GP15_optimum_structure()
    print("All creation-liquidity theorems (GP13-GP15) verified.")
