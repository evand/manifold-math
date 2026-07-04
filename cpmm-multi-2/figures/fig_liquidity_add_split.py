#!/usr/bin/env python3
"""Figure C2 — WHERE a whole-market liquidity add lands, pool by pool.

Companion to fig_liquidity_add_burn (HOW MUCH survives): the burn figure shows
v1 destroys value at any non-uniform profile; this one shows where the
surviving value sits — and answers the LP's practical question, "if one answer
has been ruled out, am I wasting liquidity by subsidizing the whole market?"

Setup: n = 5 sum-to-one market, one featured answer at probability q, the
others splitting the remainder equally. An M1,000 whole-market add. Every
reserve delta is valued IN EXPECTATION at the add-time prices (q·ΔY +
(1−q)·ΔN per answer) — a decomposition that sums to exactly M1,000 for v2
(GP17c) and to M1,000 minus the discarded EV for v1, so the two panels share
one accounting. "Destroyed" is likewise the discarded shares' EV at those
prices; the REALIZED loss depends on which answer wins (everything-or-nothing
by resolution — the realized range is fig_liquidity_add_burn's band, e.g.
[M0, M897] at a 90% favorite). v2's zero is stronger than an EV statement:
the merge contributes exactly Δ to EVERY resolution (GP17c all-winners-tight),
so "nothing destroyed" holds outcome-by-outcome. Both policies' deltas are
pure functions of the profile (independent of pool depth), so there is no
ante caveat.

"RULED OUT" here means an answer the market has traded down to q = 0.5%.
There is no hard floor: direct bets on an answer clamp at MIN_CPMM_PROB = 1%
(vendor common/src/contract.ts), but selling shares or buying the OTHER
answers pushes it lower — genuinely dead answers commonly sit at 0.1%–0.5%
on production markets. The sweep starts at 0.1%; headline numbers are quoted
at 0.5%, and they only shrink toward the bottom of that band (v2 parks
under M1 at q = 0.1%).

LEFT — v1 (`addCpmmMultiLiquidityAnswersSumToOne`): equal per-answer rounds
with partial recycling. The landed mix is distorted toward the longshots and
the rest is destroyed outright — at a 90% favorite only M96 of M1,000 reaches
the favorite's pool while M807 is discarded; even with a RULED-OUT answer
(the "safe-looking" direction) M140 burns.

RIGHT — v2 (√variance merge, GP17): everything lands, and the allocation
tracks probability — the featured pool's share ≈ q. A ruled-out answer
attracts ~nothing (M2 of M1,000 at q = 0.5%): adding to a market with a dead
answer is, in value terms, adding to the (n−1)-answer market. Not wasted.

(The dead pool does receive a large YES reserve — but YES-on-a-dead-answer is
nearly worthless, which is exactly why the EV-valued share is ~0. The v2 add
is also depth-consistent with creation: GP17e.)

All numbers computed through the vendor ports in scenarios.py; asserts pin
the conservation identity per panel, the burn figure's M807, the uniform-zero,
and both edge headlines.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import figstyle as st
import scenarios as sc

N = 5
ADD = 1000.0
# "Ruled out" = traded down to 0.5% — the upper end of where genuinely dead
# answers sit on production markets (0.1%–0.5%; direct bets clamp at
# MIN_CPMM_PROB = 1% but sells / buying other answers go below it).
Q_DEAD = 0.005
Q_LO = 0.001  # sweep from the bottom of the observed dead-answer band


def mix_curves():
    """Sweep featured q; return per-policy (featured, per-other, waste) shares."""
    # linspace(Q_DEAD, ...) puts Q_DEAD and the uniform point 1/N exactly on
    # the grid (the asserts hit them via interp); prepend Q_LO for the sweep.
    qs = np.concatenate([[Q_LO], np.linspace(Q_DEAD, 0.95, 190)])
    v1_feat, v1_other, v1_waste = [], [], []
    v2_feat, v2_other = [], []
    for q in qs:
        prof = sc.featured_profile(float(q), N)
        r1 = sc.v1_sum_to_one_add(prof, ADD)
        l1 = [x / ADD for x in r1["landed_ev_by_answer"]]
        w1 = r1["ev_loss"] / ADD
        assert abs(sum(l1) + w1 - 1.0) < 1e-7  # v1 conservation: landed + burned = add
        v1_feat.append(l1[0])
        v1_other.append(l1[1])
        v1_waste.append(w1)

        l2 = [x / ADD for x in sc.v2_merge_landed(prof, ADD)["landed_ev_by_answer"]]
        assert abs(sum(l2) - 1.0) < 1e-9  # v2: everything lands (GP17c)
        v2_feat.append(l2[0])
        v2_other.append(l2[1])
    return qs, map(np.array, (v1_feat, v1_other, v1_waste, v2_feat, v2_other))


def stack(ax, qs, feat, other, color, waste=None):
    """Stacked composition: featured pool, then the n−1 other pools, then waste."""
    lo = np.zeros_like(qs)
    hi = feat
    ax.fill_between(qs, lo, hi, color=color, alpha=0.75, lw=0)
    for k in range(1, N):  # the other pools, separated by hairlines
        lo, hi = hi, hi + other
        ax.fill_between(qs, lo, hi, color=color, alpha=0.28, lw=0)
        ax.plot(qs, lo, color=st.SURFACE, lw=0.7)
    if waste is not None:
        ax.fill_between(qs, hi, hi + waste, facecolor=st.SURFACE,
                        edgecolor=color, hatch="///", lw=0)
        ax.plot(qs, hi, color=color, lw=1.2)
    ax.plot(qs, feat, color=color, lw=1.2)


def main() -> None:
    st.apply()
    qs, (v1f, v1o, v1w, v2f, v2o) = mix_curves()

    # ---- headline asserts --------------------------------------------------
    at = lambda arr, q: float(np.interp(q, qs, arr))  # noqa: E731
    assert abs(at(v1w, 0.9) - 0.807) < 2e-3           # matches fig_liquidity_add_burn
    assert at(v1w, 1 / N) < 1e-6                      # uniform profile: v1 lossless
    assert abs(at(v1f, 1 / N) - 1 / N) < 1e-6 and abs(at(v2f, 1 / N) - 1 / N) < 1e-6
    # ruled-out band: v2 parks ~M2 at the quoted q = 0.5%, under M1 at 0.1%;
    # v1 burns >M130 across the whole band
    assert at(v2f, Q_DEAD) * ADD < 3.0 and at(v2f, Q_LO) * ADD < 1.0
    assert at(v1w, Q_DEAD) > 0.13 and at(v1w, Q_LO) > 0.13
    assert np.max(np.abs(v2f - qs)) < 0.025           # v2 featured share tracks ≈ q
    # ruled-out ⇒ ≈ the (n−1)-answer market: each live pool gets ≈ 1/(n−1)
    assert abs(at(v2o, Q_DEAD) - 1 / (N - 1)) < 0.002

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.6, 4.6), sharey=True)

    # ---------------- left: v1 ----------------
    stack(axl, qs, v1f, v1o, st.V1, waste=v1w)
    axl.annotate("destroyed\n(in expectation — realized\nis outcome-dependent)",
                 xy=(0.66, 0.80), color=st.V1,
                 fontsize=10, fontweight="bold", ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.25", fc=st.SURFACE, ec="none"))
    axl.annotate(f"ruled-out answer (q = 0.5%):\nstill burns M{at(v1w, Q_DEAD) * ADD:.0f}",
                 xy=(0.02, 0.93), xytext=(0.07, 0.70), va="top", **st.ANNOT_KW,
                 arrowprops=dict(arrowstyle="-", color=st.INK_2, lw=0.8))
    axl.annotate(f"favorite at 90%: its pool\ngets M{at(v1f, 0.9) * ADD:.0f} of M1,000",
                 xy=(0.9, at(v1f, 0.9) / 2), xytext=(0.63, 0.26),
                 ha="center", **st.ANNOT_KW,
                 arrowprops=dict(arrowstyle="-", color=st.INK_2, lw=0.8))
    axl.annotate("featured pool", xy=(0.44, 0.155), color=st.SURFACE,
                 fontsize=9.5, fontweight="bold", ha="center")
    axl.annotate("other pools", xy=(0.13, 0.44), color=st.INK,
                 fontsize=9.5, ha="center")
    axl.set_title("v1 — equal rounds + recycling\n(the rest is thrown away)",
                  fontsize=10.5)

    # ---------------- right: v2 ----------------
    stack(axr, qs, v2f, v2o, st.V2)
    axr.annotate("featured pool's share ≈ its probability", xy=(0.60, 0.38),
                 color=st.SURFACE, fontsize=9.5, fontweight="bold",
                 ha="center", va="center", rotation=36)
    axr.annotate("other pools", xy=(0.36, 0.76), color=st.INK, fontsize=9.5,
                 ha="center")
    axr.annotate(f"ruled-out answer (q = 0.5%) gets\nM{at(v2f, Q_DEAD) * ADD:.0f} "
                 "of M1,000 — like adding\nto the 4-answer market",
                 xy=(0.015, 0.03), xytext=(0.05, 0.50), va="top",
                 **st.ANNOT_KW,
                 arrowprops=dict(arrowstyle="-", color=st.INK_2, lw=0.8))
    axr.annotate("everything lands — nothing destroyed, at every\nprofile "
                 "(and in every resolution, not just EV)",
                 xy=(0.93, 0.965), ha="right", va="top", color=st.V2,
                 fontsize=9, fontweight="bold")
    axr.set_title("v2 — √variance merge (GP17)\n(allocation tracks probability)",
                  fontsize=10.5)

    for ax in (axl, axr):
        ax.axvline(1 / N, color=st.INK_2, lw=0.8, ls=(0, (2, 3)))
        ax.annotate("uniform\n1/n", xy=(1 / N + 0.012, 0.03), fontsize=8,
                    color=st.SURFACE)
        st.pct(ax, "x")
        st.despine(ax)
        ax.set_xlim(0, 0.95)
        ax.set_ylim(0, 1)
        ax.set_xlabel("featured answer's probability q (others equal)")
        ax.grid(False)
    st.pct(axl, "y")
    axl.set_ylabel("share of the M1,000 add (EV at add-time prices)")

    fig.suptitle("Where a whole-market M1,000 liquidity add lands (n = 5) — "
                 "v2 self-allocates; a ruled-out answer takes ~nothing",
                 fontsize=12, fontweight="bold", color=st.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    st.save(fig, "fig_liquidity_add_split")

    print(f"ruled out (q={Q_DEAD:.1%}):  v2 parks M{at(v2f, Q_DEAD) * ADD:.0f} in the dead pool "
          f"(each live pool M{at(v2o, Q_DEAD) * ADD:.0f} ≈ M1,000/4); "
          f"v1 burns M{at(v1w, Q_DEAD) * ADD:.0f}")
    print(f"favorite (q=90%):   v2 puts M{at(v2f, 0.9) * ADD:.0f} with the favorite, all lands; "
          f"v1 lands M{at(v1f, 0.9) * ADD:.0f} there and burns M{at(v1w, 0.9) * ADD:.0f}")
    print(f"v2 featured-pool share tracks q: max |share − q| = "
          f"{np.max(np.abs(v2f - qs)):.3f} over the sweep")


if __name__ == "__main__":
    main()
