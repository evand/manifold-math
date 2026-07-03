#!/usr/bin/env python3
"""Figure C1 — what v1 does to added liquidity (and v2 doesn't).

Two vendor paths, two panels:

LEFT — one pool (the `addCpmmLiquidityFixedP` primitive: any independent-answers
multi-choice add; also exactly the n=2 sum-to-one case). Holding p = 1/2 forces
discarding shares on one side. Valued at the answer's probability the EV loss is
|2q−1| of the amount added; the REALIZED loss depends on the resolution:
the discarded shares are all on the abundant side, worth |2q−1|/max(q,1−q) if
that side wins and NOTHING if it loses. (Binary cpmm-1 markets don't use this
primitive — their add floats p and is lossless. v2 brings that same float-p add
to multi-choice.)

RIGHT — sum-to-one whole-market add (`addCpmmMultiLiquidityAnswersSumToOne`).
The vendor partially rescues itself: thrown NO shares convert to YES-on-others
and the min across answers is reinvested as complete sets; only the excess is
dropped. The discard is therefore a pure function of the probability PROFILE:
zero at uniform (1/n each) and growing with skew — for n > 2 the safe point is
NOT 50%, it's 1/n. v2's float-p add is identically lossless at every profile.

Both panels computed from ports of the vendor functions (scenarios.py);
asserts pin the closed form, the n=2 reduction, and the uniform-zero.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import figstyle as st
import scenarios as sc


def one_pool_ev_loss(q: float) -> float:
    """|2q−1|: EV of discarded shares per unit added (fixed-p add, one pool)."""
    return abs(2 * q - 1)


def one_pool_worst_loss(q: float) -> float:
    """Realized loss per unit added if the abundant (discarded) side wins."""
    return abs(2 * q - 1) / max(q, 1 - q)


def main() -> None:
    st.apply()
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)

    # ---------------- left: one pool ----------------
    qs = np.linspace(0.01, 0.99, 393)
    ev = np.array([one_pool_ev_loss(q) for q in qs])
    worst = np.array([one_pool_worst_loss(q) for q in qs])

    # asserts: closed form == vendor port via the n=2 sum-to-one reduction
    r2 = sc.v1_sum_to_one_add(sc.featured_profile(0.7, 2), 1000.0)
    assert abs(r2["ev_loss"] / 1000 - one_pool_ev_loss(0.7)) < 1e-9
    assert abs(r2["max_loss"] / 1000 - one_pool_worst_loss(0.7)) < 1e-9
    assert r2["rounds"] == 1  # n=2: nothing to recycle

    axl.fill_between(qs, 0, worst, color=st.V1, alpha=st.BAND_ALPHA,
                     label="v1 realized loss range (by resolution)")
    axl.plot(qs, ev, color=st.V1, label="v1 expected loss")
    axl.plot(qs, worst, color=st.V1, lw=1.0, ls=(0, (3, 2)))
    axl.plot(qs, np.zeros_like(qs), color=st.V2, label="v2 (float-p add)")

    axl.annotate("if the discarded side wins", xy=(0.145, 0.885),
                 xytext=(0.34, 0.875), **st.ANNOT_KW,
                 arrowprops=dict(arrowstyle="-", color=st.INK_2, lw=0.8))
    axl.annotate("expected: |2q−1|", xy=(0.335, 0.44), color=st.V1,
                 fontsize=10, fontweight="bold", ha="right")
    axl.annotate("if it loses: 0 — and v2: 0 always", xy=(0.37, 0.045),
                 color=st.V2, fontsize=9.5, fontweight="bold")
    axl.set_title("One pool\n(independent answers; = n = 2 sum-to-one)",
                  fontsize=10.5)
    axl.set_xlabel("answer probability q when liquidity is added")
    axl.set_ylabel("loss as share of mana added")
    axl.legend(loc="lower left", bbox_to_anchor=(0.01, 0.06), fontsize=8.5)

    # ---------------- right: sum-to-one whole-market add ----------------
    for n, ls in ((3, (0, (5, 2))), (5, "solid"), (20, (0, (1, 1.5)))):
        grid = np.linspace(0.01, 0.95, 55)
        ev_n = np.array(
            [sc.v1_sum_to_one_add(sc.featured_profile(float(q), n), 1000.0)["ev_loss"] / 1000
             for q in grid]
        )
        # uniform profile is exactly lossless
        assert sc.v1_sum_to_one_add([1 / n] * n, 1000.0)["ev_loss"] < 1e-6 * 1000
        axr.plot(grid, ev_n, color=st.V1, ls=ls, label=f"v1, n = {n}")
        i0 = int(np.argmin(ev_n))
        axr.plot([grid[i0]], [0], marker="o", ms=5, color=st.V1, zorder=4)

    axr.plot([0, 1], [0, 0], color=st.V2, label="v2 (any n, any profile)")
    axr.annotate("lossless only at the uniform profile\nq = 1/n (marked dots — not 50%)",
                 xy=(0.53, 0.17), va="top", **st.ANNOT_KW)
    axr.set_title("Sum-to-one market, whole-market add\n(one featured answer at q, others equal)",
                  fontsize=10.5)
    axr.set_xlabel("featured answer's probability q")
    axr.legend(loc="upper left", fontsize=8.5)

    for ax in (axl, axr):
        st.pct(ax, "x")
        st.despine(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.04, 1.02)
    st.pct(axl, "y")

    fig.suptitle("Adding liquidity to a v1 multi-choice market destroys value — "
                 "v2's float-p add is lossless everywhere",
                 fontsize=12, fontweight="bold", color=st.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    st.save(fig, "fig_liquidity_add_burn")

    r = sc.v1_sum_to_one_add(sc.featured_profile(0.9, 5), 1000.0)
    print(f"n=5, featured 90%: M1,000 added -> EV loss M{r['ev_loss']:.0f}, "
          f"realized [M{r['min_loss']:.0f}, M{r['max_loss']:.0f}]")


if __name__ == "__main__":
    main()
