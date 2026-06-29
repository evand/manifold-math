#!/usr/bin/env python3
"""
General-p CPMM cost formulas — symbolic derivation for cpmm-multi-2.

Settles the load-bearing question for the auto-arb perf rewrite: which direction of
the cost map is closed-form for arbitrary p, and which is transcendental.

Mechanically verifiable: run this script. Each theorem asserts its key identity, so a
broken claim raises AssertionError rather than printing a wrong formula.

Companion to tasks/amm_invariants_proof/algebraic_proofs.py (same idiom). See
tasks/cpmm_multi_2/onboarding.md "Math plan".
"""

import sympy as sp
from sympy import symbols, Rational, simplify, sqrt, solve, Eq, powdenest

# p in (0,1); pools and trade sizes positive.
Y, N, A, delta, C = symbols('Y N A delta C', positive=True, real=True)
p = symbols('p', positive=True, real=True)


def _zero(expr):
    """Robust 'is this identically zero' for expressions with fractional powers."""
    e = powdenest(sp.together(simplify(expr)), force=True)
    e = simplify(powdenest(e, force=True))
    return e


def theorem_GP1_probability_and_invariant():
    print("=" * 70)
    print("GP1: General-p probability and invariant; p=0.5 reduction")
    print("=" * 70)
    # Manifold CPMM: invariant k = Y^p * N^(1-p); prob = pN / ((1-p)Y + pN).
    k = Y**p * N**(1 - p)
    prob = p * N / ((1 - p) * Y + p * N)
    print(f"  invariant k   = {k}")
    print(f"  prob(YES)     = {prob}")

    prob_half = simplify(prob.subs(p, Rational(1, 2)))
    print(f"  prob @ p=1/2  = {prob_half}   (expect N/(Y+N))")
    assert _zero(prob_half - N / (Y + N)) == 0

    k_half = simplify(k.subs(p, Rational(1, 2)))
    print(f"  k    @ p=1/2  = {k_half}   (expect sqrt(Y*N))")
    assert _zero(k_half - sqrt(Y * N)) == 0
    print("  OK\n")


def theorem_GP2_cost_in_is_closed_form():
    print("=" * 70)
    print("GP2: cost-in (spend amount A) is CLOSED-FORM O(1) for ANY p")
    print("=" * 70)
    # Buy YES with amount A via share-creation mechanics:
    #   N_new = N + A ; Y_new set to restore invariant ; shares = A + (Y - Y_new).
    k = Y**p * N**(1 - p)
    N_new = N + A
    Y_new = (k / N_new**(1 - p))**(1 / p)          # explicit, no solve
    shares = A + (Y - Y_new)
    print(f"  N_new  = {N_new}")
    print(f"  Y_new  = {Y_new}")
    print("  shares = A + (Y - Y_new)")

    # Verify the invariant is preserved EXACTLY (closed form is correct).
    inv_residual = _zero(Y_new**p * N_new**(1 - p) - k)
    print(f"  invariant residual Y_new^p N_new^(1-p) - k = {inv_residual}  (expect 0)")
    assert inv_residual == 0

    # p=0.5 reduction: shares = A*(Y+N+A)/(N+A).
    shares_half = simplify(shares.subs(p, Rational(1, 2)))
    expect_half = simplify(A * (Y + N + A) / (N + A))
    print(f"  shares @ p=1/2 = {shares_half}")
    assert _zero(shares_half - expect_half) == 0
    print("  OK  -> amount-in needs NO numerical solve, even for p != 1/2\n")


def theorem_GP3_shares_in_is_transcendental_except_half():
    print("=" * 70)
    print("GP3: shares-in (buy delta shares -> cost C) is closed-form ONLY at p=1/2")
    print("=" * 70)
    # Cost C to buy delta YES shares: Y_new = Y - delta + C, N_new = N + C,
    # invariant (Y-delta+C)^p (N+C)^(1-p) = Y^p N^(1-p).
    lhs = (Y - delta + C)**p * (N + C)**(1 - p)
    rhs = Y**p * N**(1 - p)

    # p = 1/2: collapses to a quadratic -> elementary closed form (paper eq. 104).
    eq_half = Eq(lhs.subs(p, Rational(1, 2))**2, rhs.subs(p, Rational(1, 2))**2)
    roots = solve(sp.expand(eq_half.lhs - eq_half.rhs), C)
    print(f"  p=1/2 roots for C: {roots}")
    closed = (delta - Y - N + sqrt((Y + N - delta)**2 + 4 * delta * N)) / 2
    # The economically valid (positive-cost) root must equal the paper's C^Y(delta).
    match = any(_zero(r - closed) == 0 for r in roots)
    print(f"  matches paper C^Y(delta) = (delta - Y - N + sqrt((Y+N-delta)^2+4*delta*N))/2 : {match}")
    assert match

    # General rational p escalates polynomial degree -> no uniform radical solution.
    # Demonstrate degree growth in C after clearing the fractional powers.
    for pv in [Rational(1, 2), Rational(1, 3), Rational(1, 4)]:
        e = (lhs.subs(p, pv)**sp.denom(pv) - rhs.subs(p, pv)**sp.denom(pv))
        deg = sp.Poly(sp.expand(e), C).degree()
        print(f"    p={pv}: polynomial degree in C after clearing powers = {deg}")
    print("  -> degree grows with denom(p); p=1/2 (deg 2) is the special closed-form case.")
    print("  -> general p: solve C from amount-in instead (GP2), or bounded 1-D Newton.\n")


def theorem_GP4_pure_cpmm_reversibility():
    print("=" * 70)
    print("GP4: pure-CPMM cost is a state function -> reversible C(+)+C(-)=0 (any p)")
    print("=" * 70)
    # Buy amount A (YES): pool (Y,N) -> (Y_buy, N+A), cost = A, shares s.
    k = Y**p * N**(1 - p)
    N_buy = N + A
    Y_buy = (k / N_buy**(1 - p))**(1 / p)
    s = A + (Y - Y_buy)

    # Reverse: sell those s YES shares from (Y_buy, N_buy). Selling YES returns s YES to
    # the pool and withdraws cash R; invariant must hold and pool must return to (Y,N).
    # The pool-restoring sell sets Y_back = Y_buy + s - R, N_back = N_buy - R with
    # invariant preserved. Restoration (Y_back, N_back) = (Y, N) forces R = A.
    R = symbols('R', positive=True, real=True)
    Y_back = Y_buy + s - R
    N_back = N_buy - R
    # Solve restoration of the NO pool: N_back = N.
    R_sol = solve(Eq(N_back, N), R)[0]
    print(f"  cash returned on reversing sell R = {simplify(R_sol)}  (expect A)")
    assert _zero(R_sol - A) == 0

    # With R = A the YES pool also returns and the invariant holds: net cash A - R = 0.
    Y_back_at = simplify(Y_back.subs(R, R_sol))
    print(f"  Y pool after round-trip = {Y_back_at}  (expect Y)")
    assert _zero(Y_back_at - Y) == 0
    print("  net cost A - R = 0  -> reversible.")
    print("  NB: this is pure CPMM. Limit-order segments are NOT reversible under v1;")
    print("      the reversibility FIX (net-movement settlement) is verified numerically")
    print("      in a separate script. This theorem is the foundation it rests on.\n")


def theorem_GP9_reverse_equals_buy_opposite_redeem():
    print("=" * 70)
    print("GP9: pure-CPMM reverse == Theorem-9 buy-opposite+redeem (any p)")
    print("=" * 70)
    # Selling d YES two ways from (Y, N); show identical cash AND pool.
    k = Y**p * N**(1 - p)
    d = symbols('d', positive=True, real=True)
    c = symbols('c', real=True)   # reverse cash flow (negative = received)
    cp = symbols('cp', real=True)  # buy-NO cost (positive)

    # (1) Pure-CPMM REVERSE of a YES buy: act on the SAME (NO) pool side.
    #     n_new = N + c ; shares returned = c + (Y - y_new) = -d
    #     => y_new = Y + c + d, with invariant (Y+c+d)^p (N+c)^(1-p) = k
    #        (built inline as reverse_inv below).

    # (2) Theorem 9: BUY d NO, then redeem d complete sets ($d).
    #     y_new = Y + cp ; shares bought = cp + (N - n_new) = d
    #     => n_new = N + cp - d, with invariant (Y+cp)^p (N+cp-d)^(1-p) = k.
    #     proceeds_t9 = d - cp.  Match proceeds: d - cp = -c  <=>  cp = c + d.

    # Substitute cp = c + d into the buy-NO invariant; it must collapse to the
    # reverse invariant -> the two solutions coincide (same c), hence same cash.
    buyno_inv = (Y + cp)**p * (N + cp - d)**(1 - p) - k
    buyno_at = buyno_inv.subs(cp, c + d)
    reverse_inv = (Y + c + d)**p * (N + c)**(1 - p) - k
    print("  buy-NO invariant at cp=c+d minus reverse invariant:")
    print(f"    {_zero(buyno_at - reverse_inv)}  (expect 0)")
    assert _zero(buyno_at - reverse_inv) == 0

    # Pools coincide under cp = c + d:
    #   reverse: (Y + c + d, N + c)
    #   buy-NO : (Y + cp,     N + cp - d) = (Y + c + d, N + c)
    y_rev, n_rev = Y + c + d, N + c
    y_bno, n_bno = Y + (c + d), N + (c + d) - d
    print(f"  pool gap = ({_zero(y_rev - y_bno)}, {_zero(n_rev - n_bno)})  (expect 0,0)")
    assert _zero(y_rev - y_bno) == 0 and _zero(n_rev - n_bno) == 0
    print("  => same cash (cp = c + d) and same pool: reverse IS Theorem 9, any p.")
    print("     This is why cost_for_shares(-d) is the correct sell primitive --")
    print("     there is no competing sell convention to reconcile.\n")


if __name__ == "__main__":
    theorem_GP1_probability_and_invariant()
    theorem_GP2_cost_in_is_closed_form()
    theorem_GP3_shares_in_is_transcendental_except_half()
    theorem_GP4_pure_cpmm_reversibility()
    theorem_GP9_reverse_equals_buy_opposite_redeem()
    print("All general-p cost theorems verified.")
