#!/usr/bin/env python3
"""Figure E2 — Other-split bystander damage as a function of Other's price.

Companion to fig_other_split (the concrete dumbbell example). Same construction
— v1 pool surgery (create-answer-cpmm.ts port) + the GP5d-shaped bet-down to
Σ = 1 — swept over Other's pre-split probability q_O. Manifold BASE-TIER
defaults: 3-answer market (2 listed bystanders splitting the remainder +
Other), ante = max(3 x 25, 100) = M100, self-traded to the profile, add-answer
cost M25 (answerCostTiers[0], common/src/tier.ts). Magnitudes scale with
answerCost/pool-depth — the M1,000 tier (answerCost M100, deeper pools) peaks
at −5.7pt instead of −8.3pt; shapes are identical. v2 is identically zero /
exactly-as-chosen at every q_O (GP6b).

What the sweep shows (n_listed = 2, base tier):
  * ABSOLUTE bystander shift peaks in the MID-RANGE (≈ −8.3pt around
    q_O ≈ 0.3–0.5, where the Σ-overshoot maxes at 1.5) — not at extreme q_O.
  * RELATIVE shift grows toward high q_O (≈ −38% at 0.9): small bystanders
    next to a dominant Other lose over a third of their price.
  * The new answer LANDS anywhere from ~12% to ~47% as q_O varies — a pure
    artifact of the surgery; the creator never chooses it.

Where exactly is the absolute worst case? Two stacked effects: the pure
surgery/overshoot mechanism peaks near q_O = 0.5 (fixed-equal-depth control:
peak at 0.48), but a TRADED market's pools are shallowest at the uniform point
(no self-trade capital yet: total reserves 225 at q_O = 1/3 vs 442 at 0.9), so
damage per answerCost is maximized near uniform — the realistic worst case
sits at q_O ≈ 1/n (measured: 0.33 for n = 3, 0.21 for n = 5).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import figstyle as st
import scenarios as sc
from fig_other_split import v1_split, bet_down, prob

ANTE = 100.0        # base tier: max(3 answers x M25, M100)
COST = 25.0         # answerCostTiers[0]
N_LISTED = 2


def sweep(qo: float) -> tuple[float, float, float, float]:
    ql = (1 - qo) / N_LISTED
    pools = sc.traded_v1_pools([ql] * N_LISTED + [qo], ANTE)
    listed, other = pools[:N_LISTED], pools[N_LISTED]
    la, na, no_ = v1_split(listed, other, COST)
    surgery = la + [na, no_]
    sigma = sum(map(prob, surgery))
    after = bet_down(surgery)
    return ql, prob(after[0]) - ql, prob(after[-2]), sigma


def main() -> None:
    qo_grid = np.linspace(0.05, 0.95, 46)
    rows = [sweep(float(q)) for q in qo_grid]
    ql = np.array([r[0] for r in rows])
    d_abs = np.array([r[1] for r in rows])
    c_land = np.array([r[2] for r in rows])
    sigma = np.array([r[3] for r in rows])
    d_rel = d_abs / ql

    # pin the shape claims
    i_peak = int(np.argmax(np.abs(d_abs)))
    assert 0.25 <= qo_grid[i_peak] <= 0.60               # abs damage peaks mid-range
    assert abs(sweep(0.5)[3] - 1.5) < 1e-9               # overshoot peaks at 1.5 (q_O = 0.5)
    assert np.abs(d_rel[-1]) > np.abs(d_rel[0])           # relative damage grows with q_O
    assert c_land.min() < 0.12 and c_land.max() > 0.45   # landing spans wildly
    i_mid = int(np.argmin(np.abs(qo_grid - 0.5)))

    st.apply()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.8, 3.7))

    ax1.plot(qo_grid, d_abs * 100, color=st.V1, label="cpmm-multi-1")
    ax1.axhline(0, color=st.V2, label="cpmm-multi-2")
    ax1.set_ylabel("bystander shift (prob points)")
    ax1.set_title("absolute damage peaks mid-range", fontsize=10)
    ax1.annotate("Σ overshoots to 1.5 here", xy=(0.5, d_abs[i_mid] * 100),
                 xytext=(0.42, -5.2), **st.ANNOT_KW,
                 arrowprops=dict(arrowstyle="-", color=st.INK_2, lw=0.8))
    ax1.legend(loc="lower right", fontsize=8)

    ax2.plot(qo_grid, d_rel * 100, color=st.V1)
    ax2.axhline(0, color=st.V2)
    ax2.set_ylabel("bystander shift (% of its price)")
    ax2.set_title("relative damage worst at extremes", fontsize=10)
    ax2.annotate("small bystanders lose\nover a third of their price",
                 xy=(0.55, -35), **st.ANNOT_KW)

    ax3.plot(qo_grid, c_land, color=st.V1, label="v1: where C lands")
    ax3.fill_between(qo_grid, 0.001, qo_grid - 0.001, color=st.V2, alpha=0.12,
                     label="v2: anywhere the creator chooses")
    ax3.set_ylabel("new answer C's starting prob")
    ax3.set_title("v1 picks the new answer's price for you", fontsize=10)
    ax3.legend(loc="upper left", fontsize=8)
    from matplotlib.ticker import FuncFormatter
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("Other's probability before the split")
        st.pct(ax, "x")
        st.despine(ax)

    fig.suptitle("Splitting an answer out of “Other”, swept over Other's price — "
                 "v1 damage by regime (base tier: ante M100, answerCost M25)",
                 fontsize=11.5, fontweight="bold", color=st.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    st.save(fig, "fig_other_split_sweep")

    for qo in (0.3, 0.5, 0.9):
        i = int(np.argmin(np.abs(qo_grid - qo)))
        print(f"q_O={qo_grid[i]:.2f}: bystander {d_abs[i]*100:+.2f}pt ({d_rel[i]*100:+.1f}%), "
              f"C lands {c_land[i]:.1%}, sigma {sigma[i]:.3f}")


if __name__ == "__main__":
    main()
