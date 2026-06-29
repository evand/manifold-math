#!/usr/bin/env python3
"""
GP16 — Multi-target (m>1 basket) redemption share-split for cpmm-multi-2.

The 2025 auto-arb paper formalized only SINGLE-target auto-arbitrage: the "NO in others"
redemption credits "eta YES in target" (singular). The cpmm-multi-2 v2 multi-bet path
(calculateCpmmMultiArbitrageBetsYesV2) extended this to a basket of m>1 target answers and
had to split that credit across the basket. It used eta/m YES shares per basket answer.

That is WRONG for m>=2: it destroys mana at resolution to a basket answer. Found on the
LOCAL_ONLY dev instance (a sole participant's payout came up ~10% short; v1 conserves on the
same bet). Full write-up: tasks/cpmm_multi_2/findings-m1-basket-conservation-bug-2026-06-28.md.

This script proves the correct credit is eta YES shares in EACH basket answer (not eta/m),
that this is exactly what preserves sum-to-one resolution conservation at ANY per-answer p,
and reproduces the leak numerically.

Mechanically verifiable: run it. Each theorem asserts its identity (AssertionError on break).
Self-contained — sympy only, no project imports (portable for the upstream PR).

Companion to general_p_cost.py / monotone_equilibrium.py (same idiom). Indexed in
theorems_summary.md as GP16.
"""

import sympy as sp
from sympy import symbols, simplify


def _zero(expr):
    return simplify(sp.together(expr)) == 0


# ---------------------------------------------------------------------------
def theorem_GP16a_pool_mechanic():
    """The Manifold CPMM buy moves (Y - N) by exactly -shares (YES) / +shares (NO),
    independent of p, amount, and pool. (calculate-cpmm.ts calculateCpmmShares +
    calculateCpmmPurchase: YES -> newY=y+b-s, newN=n+b; NO -> newN=n+b-s, newY=y+b.)"""
    print("=" * 72)
    print("GP16a: CPMM buy shifts (Y - N) by -s (YES) / +s (NO), exactly, any p")
    print("=" * 72)
    y, n, b, s = symbols('y n b s', positive=True)

    # YES buy of amount b that yields s shares:
    newY_yes, newN_yes = y + b - s, n + b
    dYmN_yes = (newY_yes - newN_yes) - (y - n)
    print(f"  YES buy: Delta(Y - N) = {simplify(dYmN_yes)}   (expect -s)")
    assert _zero(dYmN_yes + s)

    # NO buy of amount b that yields s shares:
    newN_no, newY_no = n + b - s, y + b
    dYmN_no = (newY_no - newN_no) - (y - n)
    print(f"  NO  buy: Delta(Y - N) = {simplify(dYmN_no)}   (expect +s)")
    assert _zero(dYmN_no - s)
    print("  OK: the (Y-N) shift is the share count, sign by side; p/amount/pool drop out.\n")


# ---------------------------------------------------------------------------
def theorem_GP16b_residual_is_eta_per_basket():
    """Holding eta NO shares in each of the (n-m) OTHER answers pays eta*(n-m) iff a basket
    answer wins, eta*(n-m-1) iff an other wins. The guaranteed floor eta*(n-m-1) is redeemed;
    the residual 'pays eta iff ANY basket answer wins' is eta YES shares in EACH basket answer
    (NOT eta/m). With eta/m a winning basket answer pays only eta/m -> short by eta*(m-1)/m."""
    print("=" * 72)
    print("GP16b: redemption residual = eta YES shares in EACH basket answer (not eta/m)")
    print("=" * 72)
    eta = symbols('eta', positive=True)
    n, m = symbols('n m', positive=True, integer=True)

    # Payout of {eta NO in each of (n-m) others} as a function of the winner's location.
    pay_basket_wins = eta * (n - m)        # all (n-m) others lose -> all NO pay
    pay_other_wins = eta * (n - m - 1)     # winner is an other -> its NO pays 0
    floor = pay_other_wins                  # guaranteed minimum across outcomes
    print(f"  pays if basket wins = {pay_basket_wins};  if an other wins = {pay_other_wins}")
    print(f"  redeemable floor    = {floor}   (matches v2 'eta*(n-m-1)' redemption credit)")

    residual_if_basket = simplify(pay_basket_wins - floor)
    residual_if_other = simplify(pay_other_wins - floor)
    print(f"  residual: basket wins -> {residual_if_basket};  other wins -> {residual_if_other}")
    assert _zero(residual_if_basket - eta) and _zero(residual_if_other)
    # The residual claim "pays eta iff winner in basket". Express as YES shares s_i (i in basket):
    # when basket answer W wins, only s_W pays (others' YES pay 0). To pay eta for EVERY W in
    # basket -> s_i = eta for all i. The flat split s_i = eta/m pays only eta/m.
    s_correct = eta
    s_buggy = eta / m
    short = simplify(s_correct - s_buggy)
    print(f"  correct per-basket credit = {s_correct};  buggy = {s_buggy};  shortfall/leg = {short}")
    assert _zero(short - eta * (m - 1) / m)
    print("  OK: residual is eta-per-basket; eta/m underpays a basket winner by eta*(m-1)/m.\n")


# ---------------------------------------------------------------------------
def theorem_GP16c_conservation_forces_eta():
    """Sum-to-one resolution conservation. Define T_i^YES = poolYes_i + traderYES_i,
    T_i^NO = poolNo_i + traderNO_i. Resolve-to-W pays T_W^YES + sum_{j!=W} T_j^NO; this is
    constant across W  <=>  D_i := T_i^YES - T_i^NO is constant across answers. Tracing the
    basket buy (GP16a) shows D shifts by `credit` on basket answers and by `eta` on others;
    they match iff credit = eta. (General p — no p appears.)"""
    print("=" * 72)
    print("GP16c: conservation (D_i = T_i^YES - T_i^NO constant) forces credit = eta")
    print("=" * 72)
    eta, credit = symbols('eta credit', positive=True)
    aY, aN = symbols('a_Y a_N', positive=True)   # amount spent on a given leg (per answer)
    g = symbols('g', positive=True)              # YES shares bought per basket answer

    # Pre-bet, every answer sits on the invariant D_i = D0 (sum-to-one funding identity).
    D0 = symbols('D0', real=True)

    # Basket answer: buy g YES (pool shift -g by GP16a), then receive g + credit YES shares.
    #   T^YES += (amount aY - g)  [pool: y+aY-g]  + (g + credit)  [trader]
    #   T^NO  += aY               [pool: n+aY]    + 0             [trader]
    dD_basket = ((aY - g) + (g + credit)) - (aY)
    print(f"  basket: Delta D = {simplify(dD_basket)}   (expect credit)")
    assert _zero(dD_basket - credit)

    # Other answer: buy eta NO (pool shift +eta), trader keeps 0 (NO zeroed as internal arb).
    #   T^YES += aN              [pool: y+aN]      + 0
    #   T^NO  += (aN - eta)      [pool: n+aN-eta]  + 0
    dD_other = (aN) - (aN - eta)
    print(f"  other : Delta D = {simplify(dD_other)}   (expect eta)")
    assert _zero(dD_other - eta)

    # Uniform shift (invariant preserved) <=> credit = eta.
    print(f"  D stays uniform  <=>  credit - eta = {simplify(dD_basket - dD_other)} = 0  =>  credit = eta")
    assert _zero((dD_basket - dD_other).subs(credit, eta))
    # And the post-bet invariant value rises by exactly eta (the per-answer redemption residual).
    Dpost_basket = D0 + dD_basket.subs(credit, eta)
    Dpost_other = D0 + dD_other
    assert _zero(Dpost_basket - Dpost_other)
    print(f"  post-bet D = {simplify(Dpost_basket)} (all answers) -> resolution conserves for every winner.")
    print("  OK: credit = eta is the unique conservation-preserving split (m=1: eta/m = eta).\n")


# ---------------------------------------------------------------------------
def _cpmm_buy(y, n, p, amount, side):
    """General-p Manifold CPMM buy (calculate-cpmm.ts). Returns (shares, newY, newN)."""
    k = y ** p * n ** (1 - p)
    if side == 'YES':
        s = y + amount - (k * (amount + n) ** (p - 1)) ** (1 / p)
        return s, y + amount - s, n + amount
    s = n + amount - (k * (amount + y) ** (-p)) ** (1 / (1 - p))
    return s, y + amount, n + amount - s


def _prob(y, n, p):
    return (p * n) / ((1 - p) * y + p * n)


def theorem_GP16d_numerical_leak_and_fix():
    """End-to-end: build a sum-to-one v2 market, run an m=2 basket buy (g YES per basket +
    eta NO per other, solved for Sum prob == 1), and resolve to a basket answer. With the
    buggy eta/m credit mana is destroyed; with eta it conserves to machine precision. Mirrors
    the dev-instance differential (leak on basket winner, conserves on a non-basket winner)."""
    print("=" * 72)
    print("GP16d: numerical end-to-end — eta/m destroys mana, eta conserves (any p)")
    print("=" * 72)
    from scipy.optimize import brentq

    # 4-answer sum-to-one market, balanced pools so prob_i = p_i (valid v2 state); skewed probs.
    ps = [0.55, 0.25, 0.12, 0.08]
    L = 100.0
    pools = [[L, L, p] for p in ps]   # [Y, N, p]
    basket = [1, 2]
    others = [0, 3]
    n = 4

    # Conservation must hold for ANY basket buy size, so fix a modest g (YES shares per basket
    # answer) and pin eta (NO shares per other) so Sum prob == 1 — no budget solve needed.
    g = 12.0

    def amount_for_shares(i, side, shares):
        return brentq(lambda a: _cpmm_buy(pools[i][0], pools[i][1], pools[i][2], a, side)[0] - shares,
                      1e-12, 1e7)

    st = [row[:] for row in pools]
    for i in basket:
        _, ny, nn = _cpmm_buy(pools[i][0], pools[i][1], pools[i][2], amount_for_shares(i, 'YES', g), 'YES')
        st[i][0], st[i][1] = ny, nn
    target = 1 - sum(_prob(st[i][0], st[i][1], st[i][2]) for i in basket)

    def others_sum(eta):
        tot = 0.0
        for j in others:
            _, ny, nn = _cpmm_buy(pools[j][0], pools[j][1], pools[j][2],
                                  amount_for_shares(j, 'NO', eta), 'NO')
            tot += _prob(ny, nn, pools[j][2])
        return tot

    eta = brentq(lambda e: others_sum(e) - target, 1e-6, 1e4)
    for j in others:
        _, ny, nn = _cpmm_buy(pools[j][0], pools[j][1], pools[j][2],
                              amount_for_shares(j, 'NO', eta), 'NO')
        st[j][0], st[j][1] = ny, nn
    assert abs(sum(_prob(st[i][0], st[i][1], st[i][2]) for i in range(n)) - 1) < 1e-9

    # Trader YES shares per basket answer: g (bought) + credit (redemption). Pools = st.
    def payout_to_winner(W, credit):
        traderYES = {i: g + credit for i in basket}
        # winner pays its YES claims (pool + trader); every other answer pays its NO pool
        # (others' trader NO zeroed as internal arb; basket trader NO = 0).
        tYes_W = st[W][0] + traderYES.get(W, 0.0)
        return tYes_W + sum(st[j][1] for j in range(n) if j != W)

    for credit, label in [(eta / len(basket), 'BUGGY eta/m'), (eta, 'FIXED eta')]:
        pay = [payout_to_winner(W, credit) for W in range(4)]
        spread = max(pay) - min(pay)
        print(f"  {label:14}: payout by winner = {[round(p,3) for p in pay]}  spread={spread:.4f}")
        if 'FIXED' in label:
            assert spread < 1e-6, "fixed credit must conserve across all winners"
        else:
            assert spread > 1.0, "buggy credit must visibly leak"
    print("  OK: eta/m leaks on basket winners; eta conserves to machine precision.\n")


if __name__ == '__main__':
    theorem_GP16a_pool_mechanic()
    theorem_GP16b_residual_is_eta_per_basket()
    theorem_GP16c_conservation_forces_eta()
    theorem_GP16d_numerical_leak_and_fix()
    print("All GP16 multi-target-redemption theorems verified.")
