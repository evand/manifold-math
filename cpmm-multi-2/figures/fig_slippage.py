#!/usr/bin/env python3
"""Figure B1 — the liquidity a trader actually experiences at an asymmetric price.

QUESTION: "The market shows 60% — how far does my M100 move it?"

Scenario as in fig_creation_cost (n = 5, featured answer at q, others equal),
with the creator's TOTAL outlay fixed at M1,000 for the headline comparison:

  v1 (same budget): uniform creation + forced self-trade to q, ante scaled so
      ante + self-trade = M1,000 (degree-1 homogeneity makes this exact);
  v2: sqrt-variance profile creation, ante M1,000 — same total capital;
  v1 (overspend, reference): ante M1,000 + self-trade paid ON TOP, i.e. the
      creator spends up to M2.4k more; shown dotted. This was round-1's solid
      line, and its extra capital is what produced the misleading down-side
      "hump" — at equal budget the v1 disadvantage is monotone in q.

Then a trader arrives and spends M100 on the featured answer — once buying it
up (YES), once betting it down (NO) — through full auto-arb. We plot how far
the price moves (less movement = deeper market).

Considered and deferred (clutter): "expected funding cost" normalization
(outlay net of the forced position's EV payout) — a third budget convention
between the two shown.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reference")))
from manifold.market_simulator import MarketSimulator  # noqa: E402

import figstyle as st  # noqa: E402
import scenarios as sc  # noqa: E402

ANTE = 1000.0
N_ANSWERS = 5
TRADE = 100.0


def v1_self_trade(q: float) -> tuple[float, float]:
    r = sc.v1_self_trade_to_profile(sc.featured_profile(q, N_ANSWERS), ANTE)
    return r["spend"], r["shares"]["a0"]


def v1_state_at(q: float) -> MarketSimulator:
    """v1, ante M1,000 + self-trade paid on top (creator overspends)."""
    return sc.v1_self_trade_to_profile(sc.featured_profile(q, N_ANSWERS), ANTE)["sim"]


def v1_budget_state_at(q: float) -> MarketSimulator:
    """v1 at the SAME total outlay as v2's ante (budget = M1,000)."""
    pools, _ = sc.v1_fixed_outlay_pools(sc.featured_profile(q, N_ANSWERS), ANTE)
    return sc.make_sim(pools)


def v2_state_at(q: float, ante: float = ANTE) -> MarketSimulator:
    return sc.make_sim(sc.v2_sum_to_one_pools(sc.featured_profile(q, N_ANSWERS), ante))


def impact(sim: MarketSimulator, position: str) -> float:
    p0 = sim.get_probability("a0")
    sim.simulate_buy("a0", TRADE, position)
    return sim.get_probability("a0") - p0


def main() -> None:
    q_grid = np.linspace(0.20, 0.92, 19)

    rows = {("v1b", "YES"): [], ("v1b", "NO"): [], ("v2", "YES"): [], ("v2", "NO"): [],
            ("v1x", "YES"): [], ("v1x", "NO"): []}
    for q in q_grid:
        q = float(q)
        makers = {
            "v1b": lambda: v1_budget_state_at(q),   # equal total outlay (headline)
            "v2": lambda: v2_state_at(q),
            "v1x": lambda: v1_state_at(q),          # ante + self-trade on top
        }
        for mech, mk in makers.items():
            for pos in ("YES", "NO"):
                rows[(mech, pos)].append(abs(impact(mk(), pos)))

    for k in rows:
        rows[k] = np.array(rows[k])

    st.apply()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)

    for ax, pos, title in zip(axes, ("YES", "NO"),
                              ("M100 buying the answer UP", "M100 betting the answer DOWN")):
        ax.plot(q_grid, rows[("v1b", pos)], color=st.V1,
                label="cpmm-multi-1, total outlay M1,000")
        ax.plot(q_grid, rows[("v2", pos)], color=st.V2,
                label="cpmm-multi-2, ante M1,000 (same outlay)")
        ax.plot(q_grid, rows[("v1x", pos)], color=st.V1, lw=1.2, ls=(0, (1, 1.5)),
                label="cpmm-multi-1 overspending: M1,000 + self-trade on top")
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("featured answer's price q")
        st.pct(ax, "x")
        st.despine(ax)

    axes[0].set_ylabel("price movement caused (prob points)")
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    axes[0].yaxis.set_major_locator(MultipleLocator(0.05))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}pt"))
    axes[0].legend(loc="upper right")

    fig.suptitle("Price impact of an M100 trade at an asymmetric price — "
                 "equal creator budget of M1,000 (n = 5)",
                 fontsize=11, fontweight="bold", color=st.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    st.save(fig, "fig_slippage")

    # console summary at the three reference asymmetries (equal-budget pair)
    print(f"{'q':>5} | {'v1 up':>7} {'v2 up':>7} | {'v1 down':>8} {'v2 down':>8}")
    for q in (0.30, 0.60, 0.90):
        i = int(np.argmin(np.abs(q_grid - q)))
        print(f"{q_grid[i]:5.2f} | {rows[('v1b', 'YES')][i]*100:6.2f}pt {rows[('v2', 'YES')][i]*100:6.2f}pt |"
              f" {rows[('v1b', 'NO')][i]*100:7.2f}pt {rows[('v2', 'NO')][i]*100:7.2f}pt")


if __name__ == "__main__":
    main()
