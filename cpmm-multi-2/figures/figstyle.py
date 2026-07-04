"""Shared visual language for the cpmm-multi-2 evidence-bundle figures.

One rule above all: v1 and v2 keep the same two colors in EVERY figure.
v1 (the incumbent baseline) is a recessive warm gray; v2 (the mechanism the
bundle argues for) is the blue accent. Color follows the entity, never the
figure. Palette values are from a CVD-validated categorical set (worst
adjacent DeltaE 24.2; blue clears 3:1 on the light surface).
"""

from __future__ import annotations

import matplotlib as mpl

# -- entities ---------------------------------------------------------------
V1 = "#6b6a66"  # cpmm-multi-1 — neutral warm gray (baseline)
V2 = "#2a78d6"  # cpmm-multi-2 — blue accent (proposed)
V1_FILL = "#6b6a66"
V2_FILL = "#2a78d6"
BAND_ALPHA = 0.18  # for min/max outcome bands

# -- surfaces & ink ----------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"  # text-primary
INK_2 = "#52514e"  # text-secondary (axis labels, captions)
GRID = "#e5e4e0"  # recessive grid

ANNOT_KW = dict(color=INK_2, fontsize=9)


def apply() -> None:
    """Set the shared rcParams. Call once, before creating any figure."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "svg.fonttype": "none",  # keep text as text in SVG
        }
    )


def despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, stem: str) -> None:
    """Save SVG (canonical, text stays text) + PNG (GitHub-friendly preview)."""
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.png", bbox_inches="tight", dpi=160)
    print(f"wrote {stem}.svg / .png")


def pct(ax, axis: str = "y") -> None:
    from matplotlib.ticker import FuncFormatter

    fmt = FuncFormatter(lambda v, _: f"{v:.0%}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)
