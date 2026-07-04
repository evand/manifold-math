#!/usr/bin/env python3
"""Figure C2 — WHERE a whole-market liquidity add lands, pool by pool.

Companion to fig_liquidity_add_burn (HOW MUCH survives): the burn figure shows
v1 destroys value at any non-uniform profile; this one shows where the
surviving value sits — and answers the LP's practical question, "if one answer
has been ruled out, am I wasting liquidity by subsidizing the whole market?"

Setup: n = 5 sum-to-one market, one featured answer at probability q, the
others splitting the remainder equally. An M1,000 whole-market add. Every
reserve delta is valued at the market's own prices (q·ΔY + (1−q)·ΔN per
answer) — a decomposition that sums to exactly M1,000 for v2 (GP17c) and to
M1,000 minus the discarded EV for v1, so the two panels share one accounting.
Both policies' deltas are pure functions of the profile (independent of pool
depth), so there is no ante caveat.

LEFT — v1 (`addCpmmMultiLiquidityAnswersSumToOne`): equal per-answer rounds
with partial recycling. The landed mix is distorted toward the longshots and
the rest is destroyed outright — at a 90% favorite only M96 of M1,000 reaches
the favorite's pool while M807 is discarded; even with a RULED-OUT answer
(q → 0, the "safe-looking" direction) M140 burns.

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
Q_DEAD = 0.005  # the "ruled out" featured probability


def mix_curves():
    """Sweep featured q; return per-policy (featured, per-other, waste) shares."""
    qs = np.linspace(Q_DEAD, 0.95, 190)
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
    assert v2f[0] < 0.003 and v1w[0] > 0.13           # ruled-out edge: v2 ~0, v1 burns
    assert np.max(np.abs(v2f - qs)) < 0.025           # v2 featured share tracks ≈ q
    # ruled-out ⇒ ≈ the (n−1)-answer market: each live pool gets ≈ 1/(n−1)
    assert abs(v2o[0] - 1 / (N - 1)) < 0.002

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.6, 4.6), sharey=True)

    # ---------------- left: v1 ----------------
    stack(axl, qs, v1f, v1o, st.V1, waste=v1w)
    axl.annotate("destroyed", xy=(0.66, 0.80), color=st.V1,
                 fontsize=11, fontweight="bold", ha="center",
                 bbox=dict(boxstyle="round,pad=0.25", fc=st.SURFACE, ec="none"))
    axl.annotate(f"ruled-out answer:\nstill burns M{at(v1w, Q_DEAD) * ADD:.0f}",
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
    axr.annotate(f"ruled-out answer gets M{at(v2f, Q_DEAD) * ADD:.0f}\n"
                 "of M1,000 — like adding to\nthe 4-answer market",
                 xy=(0.015, 0.03), xytext=(0.05, 0.50), va="top",
                 **st.ANNOT_KW,
                 arrowprops=dict(arrowstyle="-", color=st.INK_2, lw=0.8))
    axr.annotate("everything lands — nothing destroyed,\nat every profile",
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
    axl.set_ylabel("share of the M1,000 add, at market prices")

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
