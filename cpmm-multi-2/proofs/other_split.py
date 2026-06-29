#!/usr/bin/env python3
"""
GP6 — lossless "Other"-split for cpmm-multi-2 (per-answer p).

When a sum-to-one market adds an answer, Manifold carves it out of the catch-all "Other".
The v1 mechanism (backend/api/src/create-answer-cpmm.ts:224-444, all p=0.5) cannot place
the new answer and the shrunken Other at exact target probabilities while reusing Other's
existing (generally imbalanced) reserves. So it: copies Other's excess-YES shares into both
new pools, DUMPS Other's excess-NO shares onto every listed answer's YES pool (perturbing
them), overshoots to Sigma prob > 1, then BETS IT DOWN (auto-arb) and reinserts the freed
mana as subsidy. The listed answers move and a cleanup arb is required.

The cpmm-multi-2 fix rests on one fact:

    With a per-answer weight p, ANY positive reserves (Y, N) can display ANY target
    probability q, by choosing  p* = qY / (qY + (1-q)N).

That decouples "reserves held" from "probability shown". So Other's reserves split
LOSSLESSLY between the new answer and the shrunken Other, each dialed to its exact target
prob via its own p, with:
  - every LISTED answer's pool untouched  => its probability is exactly invariant,
  - prob_A + prob_Other' = prob_Other(old)  => Sigma p stays exactly 1  => NO bet-down arb,
  - reserves conserved (mana-neutral apart from the answerCost subsidy).

Parts:
  GP6a (symbolic)  p* = qY/(qY+(1-q)N) hits prob q for any positive (Y,N); in (0,1);
                   reduces to 1/2 exactly when the target is the balanced prob N/(Y+N).
  GP6b (symbolic)  the lossless split: reserves conserved + Sigma p preserved + listed
                   answers untouched, for an arbitrary reserve partition.
  GP6c (numerical) faithful v1 construction on a concrete market: shows the Sigma prob > 1
                   overshoot, the forced bet-down, and (excess-NO case) the listed-answer
                   perturbation that v2 eliminates.

Run this script; each claim asserts its key identity. Companion to general_p_cost.py and
equilibrium.py; see tasks/cpmm_multi_2/vendor-map.md for the exact vendor lines.
"""

from sympy import symbols, Rational, simplify

Y, N, q = symbols('Y N q', positive=True)


def prob_yes(Yv, Nv, pv):
    return pv * Nv / ((1 - pv) * Yv + pv * Nv)


def weight_for_prob(Yv, Nv, qv):
    """Weight p* that makes pool (Y,N) display probability q."""
    return qv * Yv / (qv * Yv + (1 - qv) * Nv)


def theorem_GP6a_weight_hits_any_prob():
    print("=" * 72)
    print("GP6a: p* = qY/(qY+(1-q)N) makes pool (Y,N) show ANY target prob q")
    print("=" * 72)
    p_star = weight_for_prob(Y, N, q)
    print(f"  p* = {p_star}")

    # Substituting p* into the probability formula returns exactly q.
    got = simplify(prob_yes(Y, N, p_star))
    print(f"  prob_yes(Y, N, p*) = {got}   (expect q)")
    assert simplify(got - q) == 0

    # p* lies strictly in (0,1) for q in (0,1), Y,N > 0: it's a ratio x/(x+y) of positives.
    # (qY) and ((1-q)N) are both positive, so 0 < p* < 1. Show the complement 1 - p*:
    one_minus = simplify(1 - p_star)
    print(f"  1 - p* = {one_minus}  (manifestly positive => p* in (0,1))")
    assert simplify(one_minus - (1 - q) * N / (q * Y + (1 - q) * N)) == 0

    # Reduces to p*=1/2 exactly at the balanced target q = N/(Y+N) (the only prob a fixed
    # p=1/2 pool can show). Off that target, a p=1/2 pool CANNOT represent (Y,N) -- which
    # is the whole reason v1 must move shares around.
    p_at_balanced = simplify(p_star.subs(q, N / (Y + N)))
    print(f"  p* at q = N/(Y+N): {p_at_balanced}   (expect 1/2)")
    assert simplify(p_at_balanced - Rational(1, 2)) == 0
    print("  OK\n")


def theorem_GP6b_lossless_split_is_exact():
    print("=" * 72)
    print("GP6b: lossless split — reserves conserved, Sigma p preserved, listed untouched")
    print("=" * 72)
    # Old Other pool (Yo, No) at weight 1/2 => prob p_o = No/(Yo+No). answerCost subsidy `a`
    # injects balanced liquidity (a to each side). Total reusable reserves:
    Yo, No, a, alpha, beta = symbols('Yo No a alpha beta', positive=True)
    p_o = No / (Yo + No)                       # old Other probability (p=1/2 market)
    Y_tot = Yo + a
    N_tot = No + a

    # Partition reserves between the new answer A and the shrunken Other' by free fractions
    # alpha, beta in (0,1). Reserves are conserved by construction (the two pieces sum to
    # the whole) -- this is what "lossless" means: no shares created or destroyed.
    Y_A, Y_O = alpha * Y_tot, (1 - alpha) * Y_tot
    N_A, N_O = beta * N_tot, (1 - beta) * N_tot
    assert simplify((Y_A + Y_O) - Y_tot) == 0 and simplify((N_A + N_O) - N_tot) == 0
    print("  reserves conserved: Y_A + Y_O' = Yo + a,  N_A + N_O' = No + a  (by partition).")

    # Target probs: carve A at q, leave Other' at (p_o - q). Each pool gets its own weight
    # p* (GP6a) to display its target on whatever reserves it received.
    qA = symbols('qA', positive=True)          # target prob for the new answer, 0 < qA < p_o
    qO = p_o - qA                               # so the two pieces re-sum to old Other prob
    pA_star = weight_for_prob(Y_A, N_A, qA)
    pO_star = weight_for_prob(Y_O, N_O, qO)

    # Each piece shows EXACTLY its target (independent of alpha, beta).
    assert simplify(prob_yes(Y_A, N_A, pA_star) - qA) == 0
    assert simplify(prob_yes(Y_O, N_O, pO_star) - qO) == 0
    print("  prob_A = qA and prob_Other' = p_o - qA  for ANY reserve partition (alpha,beta).")

    # Sigma p preserved: listed answers untouched contribute Sigma_listed = 1 - p_o
    # (since the old market summed to 1). New contribution qA + (p_o - qA) = p_o. Total = 1.
    new_other_block = simplify(qA + qO)
    print(f"  new (A + Other') probability mass = {new_other_block}  (== old Other prob p_o)")
    assert simplify(new_other_block - p_o) == 0
    print("  => Sigma p = Sigma_listed + p_o = 1 exactly: NO bet-down arb, listed probs exact.")
    print("  OK\n")


def theorem_GP6c_v1_overshoots_and_perturbs():
    print("=" * 72)
    print("GP6c: faithful v1 construction overshoots Sigma>1 and can perturb listed answers")
    print("=" * 72)
    # Reproduce create-answer-cpmm.ts:261-304 numerically on a concrete sum-to-one market.
    def prob(pool):
        return pool['NO'] / (pool['YES'] + pool['NO'])      # p = 1/2

    def v1_split(listed, other, answerCost):
        """Returns (listed_after, newAnswerPool, newOtherPool, sigma_before_betdown)."""
        mana = answerCost + min(other['YES'], other['NO'])
        excessYes = max(0.0, other['YES'] - other['NO'])
        excessNo = max(0.0, other['NO'] - other['YES'])
        answerCostOrHalf = min(answerCost, mana / 2)
        newAnswerPool = {'YES': answerCostOrHalf + excessYes, 'NO': answerCostOrHalf}
        newOtherPool = {'YES': mana - answerCostOrHalf + excessYes,
                        'NO': mana - answerCostOrHalf}
        listed_after = [{'YES': a['YES'] + excessNo, 'NO': a['NO']} for a in listed]  # dump!
        sigma = (sum(prob(a) for a in listed_after)
                 + prob(newAnswerPool) + prob(newOtherPool))
        return listed_after, newAnswerPool, newOtherPool, excessNo, sigma

    # --- Case 1: low-prob Other (the common case): poolYes > poolNo => excess YES.
    # Listed NOT perturbed, but Sigma overshoots 1 (comment 257-258) => bet-down required.
    listed1 = [{'YES': 5.0, 'NO': 5.0}, {'YES': 6.0, 'NO': 4.0}]      # probs 0.5, 0.4
    other1 = {'YES': 36.0, 'NO': 4.0}                                  # prob 0.1
    s_before1 = sum(prob(a) for a in listed1) + prob(other1)
    la1, nap1, nop1, exNo1, sigma1 = v1_split(listed1, other1, answerCost=10.0)
    listed_shift1 = max(abs(prob(a) - prob(b)) for a, b in zip(la1, listed1))
    print(f"  case 1 (Other prob 0.10, excess-YES): Sigma before = {s_before1:.4f}, "
          f"after surgery = {sigma1:.4f} (>1), listed shift = {listed_shift1:.3e}")
    assert abs(s_before1 - 1.0) < 1e-9
    assert sigma1 > 1.0 + 1e-6                  # overshoot => bet-down arb needed
    assert listed_shift1 < 1e-12                # low-prob Other: listed untouched here
    assert exNo1 == 0.0

    # --- Case 2: high-prob Other (poolNo > poolYes) => excess NO dumped onto listed YES.
    # Listed answers ARE perturbed (their probs drop), forcing the cleanup arb to matter.
    listed2 = [{'YES': 19.0, 'NO': 1.0}, {'YES': 19.0, 'NO': 1.0}]     # probs 0.05, 0.05
    other2 = {'YES': 4.0, 'NO': 36.0}                                  # prob 0.9
    s_before2 = sum(prob(a) for a in listed2) + prob(other2)
    la2, nap2, nop2, exNo2, sigma2 = v1_split(listed2, other2, answerCost=10.0)
    listed_shift2 = max(abs(prob(a) - prob(b)) for a, b in zip(la2, listed2))
    print(f"  case 2 (Other prob 0.90, excess-NO):  Sigma before = {s_before2:.4f}, "
          f"after surgery = {sigma2:.4f}, excessNo dumped = {exNo2:.1f}, "
          f"listed shift = {listed_shift2:.4f}")
    assert abs(s_before2 - 1.0) < 1e-9
    assert exNo2 > 0.0                           # there IS excess NO
    assert listed_shift2 > 1e-3                  # listed answers measurably perturbed

    # v2 on the SAME case 2: keep listed fixed, split Other losslessly. Listed shift = 0,
    # Sigma = 1 immediately (GP6b), no bet-down. Carve A at half of Other's prob.
    p_o2 = prob(other2)
    qA = p_o2 / 2
    a = 10.0
    Y_tot, N_tot = other2['YES'] + a, other2['NO'] + a
    # any partition works; use proportional-to-target for a concrete witness
    frac = qA / p_o2
    YA, NA = frac * Y_tot, frac * N_tot
    YO, NO = (1 - frac) * Y_tot, (1 - frac) * N_tot
    pA = float(weight_for_prob(YA, NA, qA))
    pO = float(weight_for_prob(YO, NO, p_o2 - qA))
    sigma_v2 = (sum(prob(b) for b in listed2)            # listed untouched
                + (pA * NA) / ((1 - pA) * YA + pA * NA)
                + (pO * NO) / ((1 - pO) * YO + pO * NO))
    print(f"  v2 same case: listed shift = 0 (untouched), Sigma = {sigma_v2:.12f} "
          f"(== 1, no bet-down), reserves conserved Y/N exactly.")
    assert abs(sigma_v2 - 1.0) < 1e-12
    assert abs((YA + YO) - Y_tot) < 1e-12 and abs((NA + NO) - N_tot) < 1e-12
    print("  => per-answer p removes the excess-NO dump and the overshoot/bet-down entirely.")
    print("  OK\n")


if __name__ == "__main__":
    theorem_GP6a_weight_hits_any_prob()
    theorem_GP6b_lossless_split_is_exact()
    theorem_GP6c_v1_overshoots_and_perturbs()
    print("All GP6 Other-split theorems verified.")
