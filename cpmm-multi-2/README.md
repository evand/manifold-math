# cpmm-multi-2 — evidence bundle

The math, machine-checked proofs, and Python reference implementation behind the upstream
`cpmm-multi-2` mechanism for [Manifold Markets](https://github.com/manifoldmarkets/manifold).

`cpmm-multi-1` (Manifold's sum-to-one multi-choice market) has three long-standing limitations
that all trace to one missing degree of freedom — a multi-choice answer has no per-answer `p`,
the way a binary `cpmm-1` market does. Adding it (defaulting to `0.5`, so every existing market
is unchanged) unlocks: **non-uniform initialization**, **lossless liquidity addition**,
**reversible-limit auto-arb** (a resting order fills when price *reaches* its limit, not on
transient overshoot), and a **direct-formula `O(n log N)` cost core**. The new behavior rides in
a new, opt-in `cpmm-multi-2` mechanism; `cpmm-multi-1` is frozen bug-for-bug.

**Start with [`rationale.md`](rationale.md)** — the full pitch: what it fixes, why it's safe, and
the evidence behind every claim.

## Layout

| Path | What |
|---|---|
| [`rationale.md`](rationale.md) | The upstream pitch + evidence map. Read this first. |
| [`paper/`](paper/) | `auto-arb-algorithms.pdf` (+ TeX source) — derives the three direct-formula auto-arb algorithms and the general-`p` cost core. |
| [`proofs/`](proofs/) | 15 standalone scripts (sympy + numpy) proving GP1–GP18. Index + status in [`proofs/theorems_summary.md`](proofs/theorems_summary.md). |
| [`reference/manifold/`](reference/manifold/) | The Python reference implementation (`closed_form_arb.py` + the AMM core it needs) — the executable oracle the TypeScript port is checked against. |
| [`tests/`](tests/) | The GP8 differential anchor: the reference path reproduces Manifold's own solver **and a real captured Manifold bet** (`data/test_fixtures/`) to ~`1e-13`. |
| [`repro/`](repro/) | Live-instance reproducibility extras (advanced) — a lifecycle harness and the v1-liquidity-defect probes. Need a local Manifold instance / API access. |

## Running it

```bash
pip install -r requirements.txt          # numpy, sympy, pytest

# Every proof — each is standalone and prints its own checks.
# (A few creation-liquidity numerical companions import the reference package.)
PYTHONPATH=reference python proofs/general_p_cost.py
for f in proofs/*.py; do PYTHONPATH=reference python "$f" || break; done

# The GP8 differential anchor (reproduces a real captured Manifold bet):
PYTHONPATH=reference:. python -m pytest tests/ -q
```

All 15 proofs pass; `tests/` is 140/140 (incl. the real-bet reproduction).

The `repro/` harness (`validate_lifecycle.py`, `probes/`) talks to a running Manifold instance
over HTTP and is **not** part of the out-of-the-box run — see comments in those files for the
API base / auth a local instance needs.

## License

MIT — see [../LICENSE](../LICENSE).
