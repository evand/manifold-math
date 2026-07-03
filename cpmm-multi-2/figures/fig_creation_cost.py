#!/usr/bin/env python3
"""Figure A1 — what it costs a creator to OPEN a market at the prices they believe.

SCENARIO: sum-to-one multi-choice, ante M1,000, one featured answer opened at q
(the rest split the remainder equally). Two panels: n = 5 and n = 20.

  v2: creation takes a probability profile directly (cpmmMulti2SumToOnePools).
      Outlay = ante; the creator holds no position, so the final cost after
      resolution (pool exhausted, no liquidity returned) is the ante regardless
      of the outcome.

  v1: creation is pinned to uniform 1/n. To open at q the creator must ALSO
      self-trade the featured answer up through auto-arb — extra cash, and it
      leaves them holding YES shares, a forced position. Final cost after
      resolution: ante + spend if the featured answer loses; ante + spend −
      shares if it wins. The shaded band is outcome risk the mechanism imposes.

All v1 numbers come from the reference implementation's auto-arb simulator via
scenarios.v1_self_trade_to_profile. For custom profiles (not just one featured
answer), run scenarios.py directly:  python scenarios.py --profile 0.55,0.25,...
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import figstyle as st
import scenarios as sc

ANTE = 1000.0


def panel(ax, n: int, q_max: float) -> dict:
    q_grid = np.linspace(1 / n, q_max, 24)
    outlay, fmin, fmax = [], [], []
    for q in q_grid:
        if abs(q - 1 / n) < 1e-9:
            r = {"outlay": ANTE, "final_min": ANTE, "final_max": ANTE}
        else:
            r = sc.v1_self_trade_to_profile(sc.featured_profile(float(q), n), ANTE)
        outlay.append(r["outlay"])
        fmin.append(r["final_min"])
        fmax.append(r["final_max"])
    outlay, fmin, fmax = map(np.array, (outlay, fmin, fmax))

    # v2 sanity at a few profiles: funding identity holds
    for q in (0.3, 0.6, 0.9):
        prof = sc.featured_profile(q, n)
        pools = sc.v2_sum_to_one_pools(prof, ANTE)
        assert abs(sc.worst_case_payout(pools) - ANTE) < 1e-6

    ax.fill_between(q_grid, fmin, fmax, color=st.V1, alpha=st.BAND_ALPHA,
                    label="v1 final cost range (forced position resolves)")
    ax.plot(q_grid, outlay, color=st.V1, label="v1 cash outlay")
    ax.plot(q_grid, fmin, color=st.V1, lw=1.0, ls=(0, (3, 2)))
    ax.axhline(ANTE, color=st.V2, label="v2 — outlay AND final cost")
    ax.set_title(f"n = {n}", fontsize=10.5)
    ax.set_xlabel("featured answer's opening probability q")
    st.pct(ax, "x")
    st.despine(ax)

    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"M{v:,.0f}"))
    return {"q": q_grid, "outlay": outlay, "fmin": fmin, "fmax": fmax}


def main() -> None:
    st.apply()
    fig, (ax5, ax20) = plt.subplots(1, 2, figsize=(10.6, 4.5))

    d5 = panel(ax5, 5, 0.95)
    d20 = panel(ax20, 20, 0.95)

    ax5.set_ylabel("creator cost  (ante = M1,000)")
    ax5.legend(loc="upper left", fontsize=8.5)

    ax5.annotate("v1: ante + self-trade to q", xy=(0.52, 1620), color=st.V1,
                 fontsize=10, fontweight="bold")
    ax5.annotate("v2: flat M1,000 — no position, no outcome risk",
                 xy=(0.94, 935), color=st.V2, fontsize=9.5, fontweight="bold",
                 ha="right", va="top")
    ax20.annotate("if the featured answer wins\n(the forced bet pays off)",
                  xy=(0.40, float(np.interp(0.40, d20["q"], d20["fmin"]))),
                  xytext=(0.08, 1950), **st.ANNOT_KW,
                  arrowprops=dict(arrowstyle="-", color=st.INK_2, lw=0.8))

    fig.suptitle("Opening a market at the prices you believe: v1 makes you bet, v2 doesn't",
                 fontsize=12, fontweight="bold", color=st.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    st.save(fig, "fig_creation_cost")

    for n, d in ((5, d5), (20, d20)):
        i = int(np.argmin(np.abs(d["q"] - 0.9)))
        print(f"n={n}: q={d['q'][i]:.2f} v1 outlay M{d['outlay'][i]:,.0f} "
              f"final [M{d['fmin'][i]:,.0f}, M{d['fmax'][i]:,.0f}]  v2 M{ANTE:,.0f}")

    overlay(d5, d20)


def overlay(d5: dict, d20: dict) -> None:
    """Companion: v1 outlay only, n = 3/5/20 overlaid — the n-insensitivity claim."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    # n=3 computed fresh; 5 and 20 reuse the panel data
    q3 = np.linspace(1 / 3, 0.95, 24)
    o3 = np.array([ANTE if abs(q - 1 / 3) < 1e-9 else
                   sc.v1_self_trade_to_profile(sc.featured_profile(float(q), 3), ANTE)["outlay"]
                   for q in q3])
    ax.plot(q3, o3, color=st.V1, ls=(0, (5, 2)), label="v1 outlay, n = 3")
    ax.plot(d5["q"], d5["outlay"], color=st.V1, label="v1 outlay, n = 5")
    ax.plot(d20["q"], d20["outlay"], color=st.V1, ls=(0, (1, 1.5)), label="v1 outlay, n = 20")
    ax.axhline(ANTE, color=st.V2, label="v2 (any n)")
    ax.annotate("the v1 penalty is about the target price,\nnot the answer count",
                xy=(0.28, 2200), **st.ANNOT_KW)
    ax.set_xlabel("featured answer's opening probability q")
    ax.set_ylabel("creator cash outlay  (ante = M1,000)")
    ax.set_title("v1 creation overhead barely depends on n")
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"M{v:,.0f}"))
    st.pct(ax, "x")
    st.despine(ax)
    ax.legend(loc="upper left", fontsize=8.5)
    st.save(fig, "fig_creation_cost_overlay")


if __name__ == "__main__":
    main()
