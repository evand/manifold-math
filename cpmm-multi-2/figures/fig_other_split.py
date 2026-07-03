#!/usr/bin/env python3
"""Figure E1 — what splitting an answer out of "Other" does to the bystanders.

SCENARIO — Manifold base-tier defaults throughout: a 3-answer sum-to-one market
(2 listed + Other) created at the base liquidity tier (ante = max(3 x 25, 100)
= M100, common/src/economy.ts + tier.ts), self-traded to listed 20% / 20%,
Other 60% (dramatic but not the worst case — see fig_other_split_sweep; the
absolute worst sits near the uniform point q_O = 1/n, where a traded market's
pools are shallowest), then the creator carves a new answer out of Other at
the base-tier answerCost = M25 (answerCostTiers[0]).

v1 (create-answer-cpmm.ts, construction follows proofs/other_split.py GP6c):
Other's excess-NO shares are DUMPED onto every listed answer's YES pool, then
the Σ > 1 overshoot is bet down by auto-arb. The bet-down is modeled as the
equal-η NO buy across all answers bisected to Σ = 1 (the GP5d equilibrium
characterization); the arb-profit is asserted ≥ 0. Net effect: bystander
answers nobody traded lose about a third of their price, and the new answer
LANDS wherever the surgery puts it — the creator has no say.

v2 (GP6a/b): with per-answer p, Other's reserves split losslessly and each
piece is dialed to its exact target probability (p* = qY/(qY+(1−q)N)). Listed
answers are byte-identical; Σ = 1 exactly, no bet-down; the new answer starts
at the chosen probability (here 30%, half of Other).
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reference")))
from manifold.amm_core import cost_for_shares, pool_after_trade  # noqa: E402

import figstyle as st  # noqa: E402
import scenarios as sc  # noqa: E402

ANTE = 100.0        # base tier: max(3 answers x M25, M100)
ANSWER_COST = 25.0  # answerCostTiers[0]


def prob(pl: dict) -> float:
    return pl["NO"] / (pl["YES"] + pl["NO"])  # p = 1/2


def v1_split(listed, other, answer_cost):
    """Port of create-answer-cpmm.ts pool surgery (proofs/other_split.py GP6c)."""
    mana = answer_cost + min(other["YES"], other["NO"])
    excess_yes = max(0.0, other["YES"] - other["NO"])
    excess_no = max(0.0, other["NO"] - other["YES"])
    half = min(answer_cost, mana / 2)
    new_a = {"YES": half + excess_yes, "NO": half}
    new_o = {"YES": mana - half + excess_yes, "NO": mana - half}
    listed_after = [{"YES": a["YES"] + excess_no, "NO": a["NO"]} for a in listed]
    return listed_after, new_a, new_o


def bet_down(pools):
    """Equal-η NO buy on every answer, bisected to Σ prob = 1 (GP5d shape)."""
    def sigma_at(eta):
        s = 0.0
        for pl in pools:
            c = cost_for_shares(pl["YES"], pl["NO"], eta, "NO")
            y2, n2 = pool_after_trade(pl["YES"], pl["NO"], c, "NO")
            s += n2 / (y2 + n2)
        return s

    lo, hi = 0.0, 500.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sigma_at(mid) > 1 else (lo, mid)
    eta = (lo + hi) / 2
    out, cost_total = [], 0.0
    for pl in pools:
        c = cost_for_shares(pl["YES"], pl["NO"], eta, "NO")
        cost_total += c
        y2, n2 = pool_after_trade(pl["YES"], pl["NO"], c, "NO")
        out.append({"YES": y2, "NO": n2})
    redeem = (len(pools) - 1) * eta
    assert redeem - cost_total > -1e-9  # the bet-down is an arb, never a cost
    return out


def main() -> None:
    q_listed, q_other = 0.20, 0.60
    pools = sc.traded_v1_pools([q_listed, q_listed, q_other], ANTE)
    listed, other = pools[:2], pools[2]
    assert abs(sum(map(prob, pools)) - 1) < 1e-3

    la, new_a, new_o = v1_split(listed, other, ANSWER_COST)
    surgery = la + [new_a, new_o]
    sigma_surgery = sum(map(prob, surgery))
    assert sigma_surgery > 1 + 1e-6
    after = bet_down(surgery)
    assert abs(sum(map(prob, after)) - 1) < 1e-9

    v1_listed = [prob(p) for p in after[:2]]
    v1_new, v1_other = prob(after[2]), prob(after[3])
    shift = v1_listed[0] - q_listed
    assert shift < -0.015                 # bystanders measurably damaged
    assert shift / q_listed < -0.30       # >30% of their price

    # v2: listed untouched; Other split at the creator's chosen 30/30.
    v2_new = v2_other = q_other / 2

    # ---------------- chart: dumbbells per answer ----------------
    st.apply()
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    rows = [
        ("Answer A (listed)", q_listed, v1_listed[0], q_listed),
        ("Answer B (listed)", q_listed, v1_listed[1], q_listed),
        ("New answer C\n(carved from Other)", None, v1_new, v2_new),
        ("Other, after split", None, v1_other, v2_other),
    ]
    for y, (label, before, v1v, v2v) in enumerate(reversed(rows)):
        if before is not None:
            ax.plot([before], [y + 0.12], "o", mfc="none", mec=st.INK_2, ms=8)
            ax.annotate("", xy=(v1v, y + 0.12), xytext=(before, y + 0.12),
                        arrowprops=dict(arrowstyle="->", color=st.V1, lw=1.8))
        ax.plot([v1v], [y + 0.12], "o", color=st.V1, ms=8)
        ax.plot([v2v], [y - 0.12], "o", color=st.V2, ms=8)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=9.5)
    ax.set_xlim(0, 1)
    st.pct(ax, "x")
    st.despine(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("probability")

    ax.annotate(f"v1 drags bystanders {q_listed:.0%} → {v1_listed[0]:.1%}\n"
                f"({shift / q_listed:+.0%} on answers nobody traded)",
                xy=(0.27, 2.42), color=st.V1, fontsize=9.5, fontweight="bold")
    ax.annotate("v2: byte-identical", xy=(0.27, 2.78), color=st.V2,
                fontsize=9.5, fontweight="bold")
    ax.annotate(f"v1: lands at {v1_new:.1%} — creator has no say",
                xy=(0.47, 1.30), color=st.V1, fontsize=9.5, fontweight="bold")
    ax.annotate(f"v2: starts at the chosen {v2_new:.0%}",
                xy=(0.47, 0.62), color=st.V2, fontsize=9.5, fontweight="bold")

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc="none", mec=st.INK_2, label="before split"),
        Line2D([], [], marker="o", ls="", color=st.V1, label="after, cpmm-multi-1"),
        Line2D([], [], marker="o", ls="", color=st.V2, label="after, cpmm-multi-2"),
    ], loc="lower right", fontsize=8.5)

    ax.set_title("Splitting a new answer out of “Other” — who moves?\n"
                 f"(base-tier market, ante M{ANTE:.0f}, answerCost M{ANSWER_COST:.0f}; "
                 f"v1 overshoots to Σ = {sigma_surgery:.3f}, then auto-arb bets everyone down — "
                 "v2: Σ = 1 exactly, no arb)",
                 fontsize=10)
    st.save(fig, "fig_other_split")
    print(f"bystanders {q_listed:.0%} -> {v1_listed[0]:.2%} ({shift/q_listed:+.1%}), "
          f"C lands {v1_new:.1%}, sigma {sigma_surgery:.3f}")


if __name__ == "__main__":
    main()
