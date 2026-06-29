#!/usr/bin/env python3
"""
GP18 — the "Other" split is a REFINEMENT: invariant w.r.t. anything that can't
distinguish the new answer A from the shrunken Other'.

THE FRAMING (Evan).  Treat the catch-all "Other" as the event (A or Other').  Adding
answer A *refines* that event into its two parts.  Therefore the operation must be
invariant under every observable that does not distinguish A from Other':
  - every LISTED answer's probability and pool (price) is unchanged;
  - prob_A + prob_Other' == prob_Other(old)  ⇒  Σ prob stays 1;
  - every participant's PAYOUT is unchanged for any resolution that treats A and
    Other' alike (the coarse events {a listed wins} and {the Other-group wins}).
Creating the answer is then just a LIQUIDITY ADDITION to the Other group (answerCost),
plus the fair 50/50 split of Other's probability mass.  (It is NOT invariant about the
liquidity/price-impact *curve* — CPMM liquidity doesn't compose perfectly under
splitting — but that is depth, not probability or mana.)

WHY THIS IS CONSERVATIVE (the piece that dissolves the "emergent conservation" worry).
The grants in convertOtherAnswerShares are redemption-NEUTRAL relabels, by the identity

    GP18a (redemption identity):  in a sum-to-one market, one  NO-i  share pays exactly
    the same as holding one YES share in EVERY other answer  (both pay 1 iff i loses).

Since Other's complement is exactly the listed answers, NO-Other ≡ Σ_listed YES-listed.
So:
  - YES-Other (pays iff Other wins ≡ A or Other' wins)  →  YES-A + YES-Other'  (grant YES-A,
    keep the persisted YES-Other');
  - NO-Other  (pays iff a listed wins)  →  YES in each listed  (remove the persisted NO-Other',
    which after the refinement would wrongly also pay when A wins, and grant YES-listed).
Both are payout-identical to the original position for every coarse event ⇒ no mana created,
listed pools never need to move.  v1's excess-NO *dump onto listed pools* (which shifts listed
prices ~3pts, GP6c) is an ARTIFACT of the p=0.5 construction, not required by conservation.

SOLVENCY IN SHARE TERMS (the invariant the construction must preserve).
For each answer i let D_i = (total YES shares) − (total NO shares), summed over ALL holders
INCLUDING the pool.  A sum-to-one market is solvent ⟺ D_i is the same for every answer (then
locked mana M = D + Σ_i TN_i is the same for every winner).  D_i is trade-invariant (an AMM buy
adds the bet amount to both the pool's YES and NO, leaving TY_i−TN_i fixed).  The split must keep
D constant.  The uniform relabel does so automatically: the new answers get D_A=D_O'=TY_O and each
listed gets D_listed = D_old + TN_O = TY_O — all equal — so it is conservative with NO backing step.

WHAT THIS SCRIPT PROVES (run it; every claim asserts)
  GP18a  redemption identity NO-i ≡ Σ_{j≠i} YES-j (combinatorial, all winners).
  GP18b  the funded construction must keep D_i constant: naively partitioning Other's reserves
         into A,Other' HALVES D (insolvent); the correct construction copies Other's YES inventory
         into BOTH new pools and funds their NO from answerCost (balanced add, D-preserving), each
         re-priced to p_o/2.  Σ prob = 1, listed prices fixed.
  GP18c  refinement payout-invariance: for a panel of traders (YES-Other, NO-Other, YES-listed,
         NO-listed, mixed), every payout matches the coarse original under the A,Other'→Other
         coarsening, for EVERY resolution outcome.
  GP18d  the whole split is a single uniform share relabel applied to EVERY holder, POOL INCLUDED
         (no special 'backing' step): D stays constant, locked mana is conserved exactly, and the
         listed POOLS' reserves (hence prices) never move — the relabeled NO-Other shares live on
         the holders (e.g. the LP), not in the listed pools.  v1's excess-NO dump onto listed pools
         (which shifts listed prices, GP6c) is an artifact of the p=0.5 construction, not required.

Companion to other_split.py (GP6, the pool-split losslessness) and liquidity_add_split.py
(GP17). See tasks/cpmm_multi_2/findings-other-split-*.md for the write-up.
"""

import math


# ----------------------------------------------------------------------------
# CPMM helpers (vendor calculate-cpmm.ts)
# ----------------------------------------------------------------------------
def cpmm_prob(Y, N, p):
    return (p * N) / ((1 - p) * Y + p * N)


def reprice_p(Y, N, q):
    """Unique p making pool (Y,N) show prob q (GP6a)."""
    return (q * Y) / (q * Y + (1 - q) * N)


def add_cpmm_liquidity(Y, N, p, amount):
    """addCpmmLiquidity: +amount to both reserves, float p to hold prob. Returns (Y',N',p')."""
    prob = cpmm_prob(Y, N, p)
    newP = (prob * (amount + Y)) / (amount - N * (prob - 1) + prob * Y)
    return Y + amount, N + amount, newP


def sum_to_one_pools(q, ante):
    """Port of calculate-cpmm.ts cpmmMulti2SumToOnePools (√variance creation; D_i=Y_i−N_i const).
    Returns list of (poolYes, poolNo, p) per answer."""
    n = len(q)
    if n < 2:
        return [(ante, ante, q[0])]
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
        out.append((poolYes, poolNo, p))
    return out


TOL = 1e-9


def _z(v, scale=1.0):
    return abs(v) <= TOL * max(1.0, abs(scale))


# ----------------------------------------------------------------------------
# Resolution payout model (sum-to-one CHOOSE_ONE)
# ----------------------------------------------------------------------------
def payout(position, winner):
    """position: dict (answer, side) -> shares. side in {'YES','NO'}.
    CHOOSE_ONE winner: a YES-i share pays 1 iff i==winner; a NO-i share pays 1 iff i!=winner."""
    total = 0.0
    for (ans, side), sh in position.items():
        if side == 'YES':
            total += sh if ans == winner else 0.0
        else:  # NO
            total += sh if ans != winner else 0.0
    return total


def GP18a_redemption_identity():
    print("=" * 72)
    print("GP18a: redemption identity  NO-i ≡ Σ_{j≠i} YES-j  (pays 1 iff i loses)")
    print("=" * 72)
    answers = ['L1', 'L2', 'O']
    for i in answers:
        no_i = {(i, 'NO'): 1.0}
        yes_others = {(j, 'YES'): 1.0 for j in answers if j != i}
        for w in answers:
            assert _z(payout(no_i, w) - payout(yes_others, w)), f"NO-{i} vs ΣYES≠{i} at {w}"
    print("  NO-i and 'YES in every other answer' have identical payout in every scenario ... OK")


def build_market(listed_probs, other_prob, ante=1000.0):
    """A sum-to-one market with k listed answers + Other. Listed answers built balanced
    (Y=N), priced via p; Other built balanced too. Σ prob = 1 by construction of inputs."""
    assert _z(sum(listed_probs) + other_prob - 1.0), "inputs must sum to 1"
    L = ante
    listed = []
    for i, q in enumerate(listed_probs):
        listed.append({'id': f'L{i+1}', 'Y': L, 'N': L, 'p': reprice_p(L, L, q), 'isOther': False})
    other = {'id': 'O', 'Y': L * 0.4, 'N': L * 1.1, 'p': 0.5}  # deliberately imbalanced Y≠N
    other['p'] = reprice_p(other['Y'], other['N'], other_prob)
    return listed, other


def GP18b_funded_construction_keeps_D():
    print("=" * 72)
    print("GP18b: the funded construction must keep D_i = Y_i−N_i CONSTANT (solvency).")
    print("       naive 'partition the reserves' HALVES D (insolvent); YES-copy+balanced-fund holds.")
    print("=" * 72)
    # √variance market {L1,L2,O}: D_i = Y_i − N_i is constant across answers by construction.
    pools = sum_to_one_pools([0.5, 0.3, 0.2], 1000.0)
    (YL1, NL1), (YL2, NL2), (Yo, No) = pools[0][:2], pools[1][:2], pools[2][:2]
    D = Yo - No
    assert _z((YL1 - NL1) - D, D) and _z((YL2 - NL2) - D, D), "pre-split D constant"
    p_o = 0.2
    answerCost = 100.0

    # --- WRONG: deepen O by answerCost then equal-partition reserves into A, O'. ---
    Yd, Nd, _ = add_cpmm_liquidity(Yo, No, pools[2][2], answerCost)
    YA_bad, NA_bad = Yd / 2, Nd / 2
    D_A_bad = YA_bad - NA_bad
    # listed D unchanged = D; but A's D = (Yo−No)/2 = D/2 ≠ D ⇒ NOT solvent
    assert not _z(D_A_bad - D, D), "equal-partition wrongly preserves D?"
    print(f"  WRONG (equal-partition): D_A = {D_A_bad:.3f} vs D_listed = {D:.3f}  → insolvent (D not constant)")

    # --- RIGHT (from GP18d): YES-O copies into BOTH A and O' (YES inventory Yo each); the answerCost
    # is a BALANCED add to each new pool (a YES + a NO, a = answerCost/2), which preserves D; the old
    # poolNo_O relabels to an LP YES-listed claim (GP18a), not into A/O' NO reserves. Reprice to p_o/2.
    a = answerCost / 2
    YA = Yo + a   # copied YES inventory + balanced add
    NA = a        # balanced add only (NO from answerCost, not from poolNo_O)
    YO2, NO2 = Yo + a, a
    D_A, D_O2 = YA - NA, YO2 - NO2
    assert _z(D_A - Yo, Yo) and _z(D_O2 - Yo, Yo), "new-answer D should be Yo"
    # listed total D becomes Yo too (each gains No YES via the relabeled NO-O claim) — but in LP
    # HOLDINGS, not listed pool reserves; so listed prices stay. (Total-level solvency in GP18d.)
    assert _z((D + No) - Yo, Yo), "listed total D = D_old + No = Yo (matches new answers)"
    # both new pools price to p_o/2 via their own p (GP6a); together restore p_o
    pA = reprice_p(YA, NA, p_o / 2)
    pO2 = reprice_p(YO2, NO2, p_o / 2)
    assert _z(cpmm_prob(YA, NA, pA) - p_o / 2, 1) and _z(cpmm_prob(YO2, NO2, pO2) - p_o / 2, 1)
    sigma = cpmm_prob(YL1, NL1, pools[0][2]) + cpmm_prob(YL2, NL2, pools[1][2]) + p_o / 2 + p_o / 2
    assert _z(sigma - 1.0, 1), f"Σ prob = 1 (got {sigma})"
    print(f"  RIGHT (YES-copy + balanced fund): D_A = D_O' = {D_A:.3f} = D_listed(total) → solvent; "
          f"A=O'={p_o/2:.3f}; Σ={sigma:.6f} ... OK")


def GP18c_refinement_payout_invariance():
    print("=" * 72)
    print("GP18c: every trader's payout is invariant under the A,Other'→Other coarsening")
    print("=" * 72)
    listed_ids = ['L1', 'L2']
    # A panel of traders with diverse positions in the ORIGINAL market {L1, L2, O}.
    traders = {
        'yesO':   {('O', 'YES'): 10.0},
        'noO':    {('O', 'NO'): 7.0},
        'yesL1':  {('L1', 'YES'): 5.0},
        'noL1':   {('L1', 'NO'): 4.0},
        'mixed':  {('O', 'YES'): 3.0, ('O', 'NO'): 2.0, ('L2', 'YES'): 6.0},
    }

    def refine(pos):
        """Apply the v2 refinement grants to an original position. O -> {A, O'}.
        YES-O -> YES-A + YES-O' (persist O' YES, grant A YES).
        NO-O  -> YES in each listed (remove persisted NO-O', grant YES-listed) [GP18a]."""
        new = {}
        for (ans, side), sh in pos.items():
            if ans == 'O' and side == 'YES':
                new[("O'", 'YES')] = new.get(("O'", 'YES'), 0.0) + sh  # persisted
                new[('A', 'YES')] = new.get(('A', 'YES'), 0.0) + sh    # granted
            elif ans == 'O' and side == 'NO':
                for lid in listed_ids:                                  # NO-O ≡ Σ YES-listed
                    new[(lid, 'YES')] = new.get((lid, 'YES'), 0.0) + sh
            else:
                new[(ans, side)] = new.get((ans, side), 0.0) + sh       # listed untouched
        return new

    # coarse map for winners: A and O' both coarsen to O
    def coarse(w):
        return 'O' if w in ('A', "O'") else w

    refined_winners = ['L1', 'L2', 'A', "O'"]
    for name, pos in traders.items():
        rpos = refine(pos)
        for w in refined_winners:
            got = payout(rpos, w)
            want = payout(pos, coarse(w))
            assert _z(got - want, max(1.0, want)), \
                f"{name}: refined payout at {w}={got} != original at {coarse(w)}={want}"
        # and the refined position never pays differently for the two refinements of O
        if any(a == 'O' for (a, _s) in pos):
            assert _z(payout(rpos, 'A') - payout(rpos, "O'"), 1) or True  # may differ? check below
    print("  YES-O, NO-O, YES-listed, NO-listed, mixed: all payouts invariant ∀ outcome ... OK")

    # Sharper: a trader who CANNOT distinguish A from O' (no A-vs-O'-specific position) must
    # get the SAME payout whether A or O' wins. True for everyone here (none bet A vs O').
    for name, pos in traders.items():
        rpos = refine(pos)
        assert _z(payout(rpos, 'A') - payout(rpos, "O'"), 1), \
            f"{name}: payout differs A vs O' despite no A-vs-O' position"
    print("  no trader can tell A from Other' (equal payout both ways) — true refinement ... OK")


def _totals(holders, answers):
    """Sum holders' shares into TY_i, TN_i per answer."""
    TY = {a: 0.0 for a in answers}
    TN = {a: 0.0 for a in answers}
    for pos in holders.values():
        for (ans, side), sh in pos.items():
            (TY if side == 'YES' else TN)[ans] += sh
    return TY, TN


def _solvency(holders, answers):
    """Return (D_by_answer, M_by_winner). Solvent ⟺ D constant ⟺ M constant across winners.
    M_w = Σ_holders payout(holder, w) = TY_w + Σ_{i≠w} TN_i = (TY_w − TN_w) + Σ_i TN_i."""
    TY, TN = _totals(holders, answers)
    D = {a: TY[a] - TN[a] for a in answers}
    M = {w: sum(payout(p, w) for p in holders.values()) for w in answers}
    return D, M


def GP18d_pool_splits_like_users():
    print("=" * 72)
    print("GP18d: the split is a uniform share relabel (pool INCLUDED) — D stays constant,")
    print("       mana conserved, no 'backing' step. Listed POOLS (prices) never move.")
    print("=" * 72)
    # Sum-to-one market {L1, L2, O} via the √variance creation (gives D_i = Y_i−N_i = const).
    pools = sum_to_one_pools([0.5, 0.3, 0.2], 1000.0)  # L1, L2, O
    answers = ['L1', 'L2', 'O']
    # Holders: each answer's POOL is a holder of its reserves (poolYes YES, poolNo NO);
    # plus a few traders. (Pools included — that's the whole point.)
    # A valid SOLVENT state has D_i = TY_i−TN_i constant. The √variance pools give that; a
    # complete-set holder (1 YES of each answer, bought for ~1 mana/set) is D-preserving (adds the
    # same amount to every TY_i) and exercises a USER YES-O alongside the pool's shares.
    cs = 8.0
    holders = {
        'pool_L1': {('L1', 'YES'): pools[0][0], ('L1', 'NO'): pools[0][1]},
        'pool_L2': {('L2', 'YES'): pools[1][0], ('L2', 'NO'): pools[1][1]},
        'pool_O':  {('O', 'YES'): pools[2][0], ('O', 'NO'): pools[2][1]},
        'completeSetHolder': {('L1', 'YES'): cs, ('L2', 'YES'): cs, ('O', 'YES'): cs},
    }
    D0, M0 = _solvency(holders, answers)
    spread0 = max(D0.values()) - min(D0.values())
    assert _z(spread0, max(abs(v) for v in D0.values())), f"D not constant pre-split: {D0}"
    Mvals0 = list(M0.values())
    assert _z(max(Mvals0) - min(Mvals0), Mvals0[0]), f"M not constant pre-split: {M0}"
    print(f"  pre-split: D≡{D0['O']:.3f} (constant), locked mana M≡{M0['O']:.3f} (constant) ... OK")

    # THE SPLIT: apply to EVERY holder uniformly (pool_O included, no special casing).
    listed_ids = ['L1', 'L2']
    new_answers = ['L1', 'L2', 'A', "O'"]

    def split_holder(pos):
        new = {}
        for (ans, side), sh in pos.items():
            if ans == 'O' and side == 'YES':
                new[('A', 'YES')] = new.get(('A', 'YES'), 0.0) + sh
                new[("O'", 'YES')] = new.get(("O'", 'YES'), 0.0) + sh
            elif ans == 'O' and side == 'NO':
                for lid in listed_ids:                       # NO-O ≡ Σ YES-listed (held by THIS holder)
                    new[(lid, 'YES')] = new.get((lid, 'YES'), 0.0) + sh
            else:
                new[(ans, side)] = new.get((ans, side), 0.0) + sh
        return new

    split_holders = {h: split_holder(p) for h, p in holders.items()}
    D1, M1 = _solvency(split_holders, new_answers)
    spread1 = max(D1.values()) - min(D1.values())
    assert _z(spread1, max(abs(v) for v in D1.values())), f"D not constant post-split: {D1}"
    Mvals1 = list(M1.values())
    assert _z(max(Mvals1) - min(Mvals1), Mvals1[0]), f"M not constant post-split: {M1}"
    # mana exactly conserved by the relabel alone (answerCost not yet added)
    assert _z(M1["A"] - M0['O'], M0['O']), f"mana not conserved: {M1['A']} vs {M0['O']}"
    print(f"  post-split (relabel only): D≡{D1['A']:.3f} constant across L1,L2,A,O'; "
          f"M≡{M1['A']:.3f} == pre-split (conserved) ... OK")
    # the listed answers' POOL reserves are untouched (pool_L1/L2 holders unchanged) → prices fixed.
    assert split_holders['pool_L1'] == holders['pool_L1'], "listed pool L1 reserves moved!"
    assert split_holders['pool_L2'] == holders['pool_L2'], "listed pool L2 reserves moved!"
    # the relabeled NO-O shares live on the HOLDERS (e.g. pool_O now holds YES-listed claims),
    # NOT in the listed CPMM pools — so listed PRICES are invariant while payouts still balance.
    assert split_holders['pool_O'].get(('L1', 'YES'), 0) == holders['pool_O'][('O', 'NO')], \
        "pool_O's NO-O should relabel to its own YES-listed claim"
    print("  listed POOL reserves byte-identical (prices fixed); pool_O's NO-O → its YES-listed "
          "claim ... OK")
    print("  ⇒ no 'backing' gymnastics: pool shares split by the SAME identities as user shares.")


def _W(Y, N, p):
    """Marginal-liquidity depth metric W = (1-p)Y + p N (GP13: dprob/dshares = q(1-q)/W)."""
    return (1 - p) * Y + p * N


def GP18e_liquidity_curve_dof():
    print("=" * 72)
    print("GP18e: a liquidity-curve DOF REMAINS after fixing all probs (incl prob_A=prob_O')")
    print("       and solvency (D). Two constructions agree on every invariant, differ in depth.")
    print("=" * 72)
    pools = sum_to_one_pools([0.5, 0.3, 0.2], 1000.0)
    Yo = pools[2][0]
    p_o = 0.2
    answerCost = 100.0
    D_new = Yo  # the copy-construction's constant (GP18d)

    def build(xA, xO):
        """Fund A's NO with xA, Other's NO with xO (xA+xO = answerCost). YES = Yo copied into
        both ⇒ D_A = (Yo+xA)-xA = Yo, D_O' = Yo. Reprice each to p_o/2. Same probs/D/mana, any (xA,xO)."""
        YA, NA = Yo + xA, xA
        YO, NO = Yo + xO, xO
        pA = reprice_p(YA, NA, p_o / 2)
        pO = reprice_p(YO, NO, p_o / 2)
        return (YA, NA, pA), (YO, NO, pO)

    cons = {
        'symmetric':      build(answerCost / 2, answerCost / 2),
        'A-deep/O-shallow': build(answerCost * 0.8, answerCost * 0.2),
    }
    Wsets = {}
    for name, ((YA, NA, pA), (YO, NO, pO)) in cons.items():
        # invariants identical across constructions:
        assert _z(cpmm_prob(YA, NA, pA) - p_o / 2, 1), f"{name} prob_A"
        assert _z(cpmm_prob(YO, NO, pO) - p_o / 2, 1), f"{name} prob_O'"
        assert _z((YA - NA) - D_new, D_new) and _z((YO - NO) - D_new, D_new), f"{name} D"
        # total NO funded by answerCost is the same (xA+xO=answerCost) ⇒ same mana
        assert _z((NA + NO) - answerCost, answerCost), f"{name} mana"
        WA, WO = _W(YA, NA, pA), _W(YO, NO, pO)
        Wsets[name] = (WA, WO)
    # ...but the liquidity curves (W) differ between the two constructions:
    assert not _z(Wsets['symmetric'][0] - Wsets['A-deep/O-shallow'][0], Wsets['symmetric'][0]), \
        "constructions should differ in W_A"
    print(f"  symmetric:        W_A={Wsets['symmetric'][0]:.2f}  W_O'={Wsets['symmetric'][1]:.2f}")
    print(f"  A-deep/O-shallow: W_A={Wsets['A-deep/O-shallow'][0]:.2f}  W_O'={Wsets['A-deep/O-shallow'][1]:.2f}")
    print("  same probs, same D, same mana — different curves ⇒ depth is a genuine free DOF ... OK")
    # HOW MANY DOF? Count it. Free params of the A,O' pools = 6 (PY,PN,p each). The copy of Other's
    # YES inventory (Yo into both) is FORCED by the refinement of the pool's YES-O holding; the
    # answerCost funding of each new pool must be a BALANCED add (else D_A≠Yo, insolvent). So each
    # new pool is pinned except its depth, and the two depths share one mana budget:
    #   prob_A, prob_O' fixed (2) + D_A=D_O'=D_new (2, and D_new=Yo is itself forced by the listed
    #   relabel) + mana=answerCost (1)  ⇒  6−5 = 1 free parameter.
    # That 1 DOF is exactly build(xA, answerCost−xA): the split of the balanced add between A and O'.
    # Verify the witness samples really are one continuous family (a 3rd interior point exists):
    (Ya, Na, pa), _ = build(answerCost * 0.35, answerCost * 0.65)
    assert _z(cpmm_prob(Ya, Na, pa) - p_o / 2, 1) and _z((Ya - Na) - D_new, D_new)
    print("  ⇒ in the TIGHT family (listed pools fixed, conservation, refinement relabels): exactly")
    print("    ONE DOF — the A↔Other' depth split (build(xA, answerCost−xA)); symmetric is xA=c/2.")
    # Relaxing 'listed pools fixed' to merely 'listed PROBS fixed' reopens more: each listed answer
    # then has its own depth DOF (k more) plus the NO-O routing (LP-claim vs into-listed-pools) — but
    # those move listed curves/prices, sacrificing refinement-invariance, so we don't take them.
    decline = math.sqrt((p_o / 2) * (1 - p_o / 2)) / math.sqrt(p_o * (1 - p_o))
    print(f"  (note) the √variance 'in-between': O' depth at p_o/2 wants ~{decline:.3f}× its p_o depth "
          f"(not 0.5×) — a specific interior point of that 1 DOF; symmetric is the chosen one")


def GP18f_pool_no_must_relabel_not_duplicate():
    print("=" * 72)
    print("GP18f: the pool's NO inventory must RELABEL to YES-listed (conserves); DUPLICATING")
    print("       it into both new pools is solvent (D const) but CREATES exactly `No` mana.")
    print("=" * 72)
    pools = sum_to_one_pools([0.5, 0.3, 0.2], 1000.0)  # L1, L2, O
    Yo, No = pools[2][0], pools[2][1]
    answers0 = ['L1', 'L2', 'O']
    holders0 = {
        'pool_L1': {('L1', 'YES'): pools[0][0], ('L1', 'NO'): pools[0][1]},
        'pool_L2': {('L2', 'YES'): pools[1][0], ('L2', 'NO'): pools[1][1]},
        'pool_O':  {('O', 'YES'): Yo, ('O', 'NO'): No},
    }
    _, M0 = _solvency(holders0, answers0)
    M_old = M0['O']
    c = 100.0
    a = c / 2
    new_answers = ['L1', 'L2', 'A', "O'"]

    # (RELABEL) pool_O: YES-O copies to A,O'; NO-O → YES in each listed; +balanced a to each new pool.
    relabel = {
        'pool_L1': dict(holders0['pool_L1']),
        'pool_L2': dict(holders0['pool_L2']),
        'pool_A':  {('A', 'YES'): Yo + a, ('A', 'NO'): a},
        "pool_O'": {("O'", 'YES'): Yo + a, ("O'", 'NO'): a},
        # pool_O's No relabeled to YES-listed claims (held by the LP), NOT into listed pools:
        'lp_claim': {('L1', 'YES'): No, ('L2', 'YES'): No},
    }
    _, Mr = _solvency(relabel, new_answers)
    assert _z(max(Mr.values()) - min(Mr.values()), Mr['A']), "relabel: M constant"
    assert _z(Mr['A'] - (M_old + c), M_old + c), f"relabel: M_new should be M_old+answerCost ({Mr['A']} vs {M_old + c})"
    print(f"  RELABEL No→YES-listed: M_new = {Mr['A']:.3f} = M_old({M_old:.3f}) + answerCost({c}) ... OK")

    # (DUPLICATE) keep No in BOTH new pools (the tempting shortcut). Still D-constant/solvent...
    dup = {
        'pool_L1': dict(holders0['pool_L1']),
        'pool_L2': dict(holders0['pool_L2']),
        'pool_A':  {('A', 'YES'): Yo + a, ('A', 'NO'): No + a},
        "pool_O'": {("O'", 'YES'): Yo + a, ("O'", 'NO'): No + a},
    }
    _, Md = _solvency(dup, new_answers)
    assert _z(max(Md.values()) - min(Md.values()), Md['A']), "duplicate: still solvent (D const)"
    # ...but it creates exactly `No` extra mana — NOT conservative.
    assert _z(Md['A'] - (M_old + c + No), M_old + c + No), "duplicate: M_new should be M_old+c+No"
    print(f"  DUPLICATE No into both:  M_new = {Md['A']:.3f} = M_old + answerCost + No({No:.3f}) "
          f"→ creates {No:.3f} mana (WRONG)")
    print("  ⇒ conservation PINS the choice: relabel the pool's NO to YES-listed, don't duplicate it.")


def GP18g_two_routes_for_the_pool_no():
    print("=" * 72)
    print("GP18g: two valid routes for the pool's NO-O shares (Evan). In sum-to-one CHOOSE_ONE,")
    print("       YES-i + NO-i = 1 M$, and YES-O = YES-A + YES-O', so per share:")
    print("       NO-A + NO-O' = NO-O + 1 M$.  ⇒ (a) relabel NO-O → ΣYES-listed is FREE (payout-")
    print("       identical); (b) duplicate NO-O → NO-A + NO-O' COSTS 1 M$/share.")
    print("=" * 72)
    answers = ['L1', 'L2', 'A', "O'"]   # post-split answers; 'listed' = L1,L2

    def mana_of(bundle, listed):
        """A bundle is solvent-priced at M$ = its payout in ANY single outcome iff that payout is
        constant across outcomes (YES-i+NO-i=1 each, complete sets =1). Return (min,max) payout."""
        ws = [payout(bundle, w) for w in answers]
        return min(ws), max(ws)

    # (a) FREE relabel: 1 NO-O  vs  ΣYES-listed — identical payout in every outcome (mana-equal).
    # NO-O pays 1 iff a listed wins; ΣYES-listed pays 1 iff a listed wins → identical.
    no_o_equiv = {('L1', 'YES'): 1.0, ('L2', 'YES'): 1.0}
    for w in answers:
        pay_no_o = 1.0 if w in ('L1', 'L2') else 0.0
        assert _z(payout(no_o_equiv, w) - pay_no_o), f"NO-O ≡ ΣYES-listed at {w}"
    print("  (a) NO-O ≡ ΣYES-listed: identical payout every outcome ⇒ FREE relabel (what we ship)")

    # (b) DUPLICATE: NO-A + NO-O' pays 2 on a listed win, 1 on A or O' → = NO-O + 1 M$ (a complete
    # set pays 1 everywhere). Verify NO-A+NO-O' − (constant 1) reproduces NO-O's payout.
    dup = {('A', 'NO'): 1.0, ("O'", 'NO'): 1.0}
    for w in answers:
        pay_no_o = 1.0 if w in ('L1', 'L2') else 0.0
        # subtract one complete set (pays 1 everywhere) from the duplicate ⇒ NO-O's payout
        assert _z((payout(dup, w) - 1.0) - pay_no_o), f"NO-A+NO-O' − 1 == NO-O at {w}"
    print("  (b) NO-A + NO-O' = NO-O + 1 M$ (per share): duplicating the pool's NO COSTS No M$ —")
    print("      a valid alternative (deeper NO in the new pools), just not free. Both conserve.")


if __name__ == "__main__":
    GP18a_redemption_identity()
    GP18b_funded_construction_keeps_D()
    GP18c_refinement_payout_invariance()
    GP18d_pool_splits_like_users()
    GP18e_liquidity_curve_dof()
    GP18f_pool_no_must_relabel_not_duplicate()
    GP18g_two_routes_for_the_pool_no()
    print("\nGP18 — Other-split refinement invariance verified.")
