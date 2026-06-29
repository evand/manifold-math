# cpmm-multi-2 — Theorems Summary (proof index)

House pattern mirrors `tasks/amm_invariants_proof/theorems_summary.md`. Each entry: claim,
script, status. Feeds `docs/auto-arb-proof-coverage.md` once mature. Run scripts to verify.

## General-p cost core — `proofs/general_p_cost.py` ✅ (symbolic, verified)
- **GP1** — invariant `k=Y^p·N^(1−p)`, `prob=pN/((1−p)Y+pN)`; both reduce correctly at p=1/2.
- **GP2** — **cost-in (spend amount A) is closed-form O(1) for any p.** Invariant residual
  is exactly 0. This is the operation the perf rewrite should build on.
- **GP3** — **shares-in (δ→cost) is closed-form only at p=1/2** (quadratic, recovers paper
  eq. 104). Cleared-power polynomial degree = denom(p) (2,3,4,…) ⇒ transcendental for
  general p. ⇒ parameterize auto-arb in the cost-in direction; residual shares-in steps =
  bounded 1-D Newton.
  - **How transcendental? "Awkwardly," not a packaged special function.** The defining
    relation in log space is `p·log(Y−δ+C) + (1−p)·log(N+C) = log k` — a sum of *two*
    logarithms with distinct shifts and an irrational weight. That is **not** `x·e^x`
    form, so **no Lambert-W** (or other single std special function) closes it. For
    rational p=a/b it's an algebraic equation of degree b: p=1/3,1/4 give cubic/quartic
    (radical solutions exist but are Cardano/Ferrari-ugly); **p=1/5 already has Galois
    group S₅ (order 120) ⇒ no radical solution at all** (verified via `Poly.galois_group`).
    Irrational p isn't even a finite polynomial. Practical upshot: there is nothing to
    "look up" — invert numerically. GP5c shows Newton from the p=½ closed-form seed
    clears it in ≤5 steps.
- **GP4** — pure-CPMM cost is a state function ⇒ reversible `C(+)+C(−)=0` for any p. Limits
  are the *only* irreversibility source (handled separately). The proof prescribes
  **convention A**: the reverse acts on the *same* pool side as the buy (`N_back = N_buy − R`).
  **Slice 3 finding:** `amm_core.pool_after_trade` had diverged from this — it switched pool
  sides on the sign of `cost`, so a buy/sell round-trip drifted off the start *even at p=½*
  (cost cancelled, pool didn't). Fixed in Slice 3; the GP4 fuzz test
  (`tests/test_slice3_reversibility.py`) now holds against the implementation.
- **GP9** — **the pure-CPMM reverse IS Theorem 9 (buy-opposite + redeem), at any p.** Selling
  d of one side via the convention-A reverse gives *identical* cash and pool to buying d of the
  opposite side and redeeming d complete sets (symbolic: the buy-opposite invariant at
  `cp = c + d` collapses to the reverse invariant; pools coincide). So `cost_for_shares(−d)` is
  the correct sell primitive and there is **no second sell convention to reconcile** — Theorem 9
  (previously stated at p=½) generalizes off p=½. Numerically anchored in
  `tests/test_slice3_reversibility.py::TestReverseEqualsTheorem9`.

## Equilibrium core — `proofs/equilibrium.py` ✅ (symbolic + numerical, verified)
- **GP5a** — buying NO strictly lowers an answer's prob for **any p** (symbolic: the
  dprob/dC numerator factors `N·p·(C+Y)·(p−1)·Y^(p/(p−1))`, sign-negative by inspection;
  invariant preservation log-verified). Worked in the cost coordinate (closed form, GP2)
  and lifted to the share coordinate by cost↔shares monotonicity ⇒ `Σp(η)` strictly
  monotone ⇒ the equilibrium η **exists and is unique**, `sign(η)=sign(Σp−1)`. Extends
  **S3** off p=½.
- **GP5b** — at p=½ the NO shares-in cost is closed-form (quadratic), the mirror of GP3's
  `C^Y`. The one p where each arb leg is truly O(1).
- **GP5c** — general-p arb leg (shares→cost) is a bounded, monotone, convex 1-D inversion;
  Newton from the p=½ seed clears it in **≤5 steps**. Full equilibrium solver (single outer
  search + per-leg Newton) hits Σp=1 to <1e-10 on mixed-p configs.
- **GP5d** — the `η·(n−1)` NO-in-all redemption value is **outcome- and p-independent**
  (risk-free constant), so folding it into `c_net` is exact (S4 off p=½). The equilibrium
  is one monotone root ⇒ a single bisection reaches it to machine precision (≈53 steps,
  |err|=0). **v1's `>0.01` outer leftover-mana loop is dissolved, not merely tightened** —
  it was an artifact of the iterate-buy-arb-reinvest fixed point, not a real tolerance.

## Other-split core — `proofs/other_split.py` ✅ (symbolic + numerical, verified)
- **GP6a** — **`p* = qY/(qY+(1−q)N)` makes any positive reserves `(Y,N)` show any target
  prob `q`** (symbolic: substituting `p*` returns exactly `q`; `p*∈(0,1)`; reduces to ½
  exactly at the balanced target `q=N/(Y+N)`). This is the DOF a fixed-½ pool lacks.
- **GP6b** — the lossless split: carve A at `qA`, leave Other′ at `p_o−qA`, on an
  arbitrary partition of Other's reserves `+answerCost`. Reserves conserved by partition;
  each piece shows its exact target (GP6a); `qA+(p_o−qA)=p_o` ⇒ **Σp stays exactly 1, no
  bet-down, listed probs exact** — all symbolic.
- **GP6c** — faithful v1 construction (vendor `create-answer-cpmm.ts:261-304`) on concrete
  markets: low-prob Other overshoots to Σ≈1.20 (bet-down forced even though listed
  untouched); **high-prob Other dumps 32 excess-NO shares, shifting listed answers ~3 pts**
  before cleanup. v2 on the same case: listed shift 0, Σ=1 to 1e-12, reserves conserved.

## Reversibility-with-limits — `proofs/reversibility_limits.py` ✅ (numerical fuzz)
- **Where the break is.** NOT the pool (GP4 reversible). Vendor maker fills
  (calculate-cpmm-arbitrage.ts:281-300) don't move `cpmmState`; only pool fills do. The
  ratchet is `applyMakersToWorkingState` (149-183) mutating each maker's filled
  amount/shares *in place* (169-170) and never un-filling. The inner binary search brackets
  the target from BOTH sides, so an overshoot probe consumes makers the final answer never
  crosses → the computed result depends on the probe trajectory.
- **Trade vs. search (the distinction that matters).** A *real* round-trip trade (buy up,
  later sell down) crosses different makers each way and is legitimately irreversible — real
  spread paid, correct. What must be reversible is the auto-arb SEARCH's internal trajectory:
  it computes ONE net bet, so its result must depend only on net δ, never on how it probed.
- **The fix = temporary reverse limits** (docs/amm-invariants.md §8, lines 119-125): crossing
  a maker at ρ during the atomic op leaves a reverse order at the same price, which a
  down-probe re-crosses FIRST (price priority) for an exact refund. This keeps the real limit
  book live in the search (limit-accurate intermediate pricing — unlike a pure-CPMM proxy)
  while making the consumed amount of every maker a **pure function of current price**, so
  taker cash is a potential `Φ(p)` and `cash(a→b)=Φ(b)−Φ(a)`. Reversibility and
  path-independence are then identities, not approximations.
- **GP7a** — round trip `0.3→0.7→0.3`: reverse-limit cash = exactly 0; naive in-place leaks
  −7.74 and leaves 3 orders consumed. `C(+δ)+C(−δ)=0` restored across limits.
- **GP7b** — fuzz 400 random books × trajectories: reverse-limit cost spread across
  trajectories = exactly 0 (Φ is a state function); naive diverges from the direct cost in
  **383/400**.
- **GP7c** — on monotone (no-overshoot) searches, reverse-limit == naive to 8.9e-16:
  **conservative**, corrective only on excursions.
- **GP7d** — at commit the reverse limits vanish, leaving exactly the makers crossed by the
  net δ (`consumed(·, p_end)`); a wild probe path's total cost == the direct start→end cost
  (300/300, dev 0). Search+commit reproduces one clean net bet — a real later round-trip is a
  separate atomic op with fresh scaffolding, hence still legitimately irreversible.
- **Scope.** Pool modeled as pure CPMM (GP4) and omitted — maker consumption is the sole
  irreversibility source. This validates the reverse-limit *design*; GP8 anchors the
  numerics against vendor directly.
- **GP7 / our Python — already path-independent, no reverse-limit machinery needed (Slice 3).**
  The reverse-limit fix is a *vendor* (PR2/TypeScript) concern. **Our** auto-arb search reads a
  fresh `order_book = raw − _limit_consumption` on every probe, and `_limit_consumption` mutates
  **only** in `apply_trade`, never during a read-only calc (`market_simulator.py:111-116`). So a
  probe never consumes makers a later probe sees — the search result is a pure function of the
  net bet, by construction. Vendor's in-place `applyMakersToWorkingState` ratchet has no analogue
  here. Verified (not assumed) in `tests/test_slice3_reversibility.py::TestSearchPathIndependence`:
  a large limit-crossing probe between two identical small probes leaves the small result
  byte-identical; read calc leaves consumption state untouched; clone == original. This is the
  Slice-3 "verify-then-minimal" deliverable: we confirmed the property rather than building
  machinery to enforce it.

## v1 equivalence anchor — `manifold/closed_form_arb.py` + `tests/test_gp8_direct_equivalence.py` ✅ (numerical/differential, Slice 2)
- **GP8** — the direct closed-form dollar-centric path (`calculate_purchase_with_arbitrage_direct`,
  formulas written **fresh** from GP3/GP5b, not via `amm_core`) reproduces the production
  nested-search solver (`calculate_purchase_with_arbitrage`) on **p=0.5/no-limit** configs to
  **machine precision** — across a YES/NO × n∈{2,3,4,5} × amount∈{1,10,50,200} sweep the worst
  disagreement was **2.8e-13 shares / 2.4e-11 pools**. Tolerances set ~1e3× the measured floor
  (1e-9 shares, 1e-7 pools), far below any real algorithmic drift. **External anchor:** the same
  direct path matches a *real captured Manifold bet* (`tests/data/multi_choice_bet_1`: a $1 buy
  on a 10-answer market that executed for 6.185468080296236 shares) to **rel 6.6e-13** — vendor
  ground truth, not just our own solver. This licenses the upstream O(n log²N)→O(n log N) rewrite
  and is the PR1 regression seed. (Only at p=0.5 is vendor a valid oracle — it throws on p≠0.5.)
  - **The 2025 "3.7% large-bet" discrepancy is FIXED, not open.** That gap was a real bug in an
    older *partially-equivalent* arb algorithm; it was resolved by adopting Manifold's own
    direct-inverse techniques (`calculate_shares_exact` rewrite 2026-05-07, see
    `docs/auto-arb-proof-coverage.md`) plus limit-order-consumption awareness — especially with
    limits present. Production now matches vendor; direct == production to machine precision here
    confirms the closed-form path inherits that fidelity. The stale `test_multichoice_calculation_vs_api.py`
    (skipped, snapshot gone) documented the *old* bug and was removed.

## Three-parameterization API — `proofs/parameterization_equivalence.py` + `tests/test_slice4_parameterizations.py` ✅ (numerical/differential, Slice 4)
- **GP10** — the paper's three auto-arb strategies (dollar / share / probability-centric) are
  **one trade in three coordinates.** Despite different redemption decompositions — dollar &
  prob arb "NO in others" (redemption η(n−2) for a YES buy); share arbs "NO in all" (redemption
  η(n−1)) — feeding one parameterization's output as another's input recovers an **identical
  (shares, cost, target-prob) triple AND identical final pools**, on the whole sum-to-one domain,
  both positions, n∈{2..5}. Numeric fuzz: worst cross-agreement / pool dev **5.4e-9 / 5.0e-9**
  over ~5000 normalized markets. So "three algorithms" is safe — the redemption bookkeeping is a
  relabeling, not a different equilibrium. **NO-purchase redemption** (not in the paper's
  YES-only pseudocode) derived + confirmed against production: share-centric η·1 (YES-in-all,
  only the winner pays); dollar/prob 0 (mirrors the GP8-validated dollar path).
- **Each parameterization independently anchored to production** at p=0.5 (GP8-style): share ==
  `calculate_shares_exact`, prob == `buy_to_probability` (both to ~1e-10, both positions), dollar
  == `calculate_purchase_with_arbitrage` (GP8). The fresh closed forms live in
  `manifold/closed_form_arb.py` as the PR1 reference surface.

## v1 truncation residual + exact-root correctness — `proofs/truncation_residual.py` ✅ (numerical, GP11)
- **GP11** — **v2 computes the exact equilibrium; v1 truncates it.** Vendor's multi-answer arb
  (`calculateCpmmMultiArbitrageBetsYes`) is an iterate-buy-arb-reinvest loop
  `while (amountToBet > 0.01)` (`calculate-cpmm-arbitrage.ts:203`) that exits with ≤ **0.01 mana**
  of redemption left unreinvested. v2's direct path drives `Σp = 1` to machine precision — it
  **is** the `0.01 → 0` limit of that same iteration (so v2 is *more correct*, not merely
  different). The script models **both** algorithms in self-contained p=0.5 closed forms (no
  project imports) and proves two things:
  - **GP11a (exact-root correctness)** — across the fixture sweep, v1(τ)'s total acquired shares
    increase **monotonically** as τ→0 and converge to v2's single-solve value; v2 lands `Σp = 1`
    to **2.2e-16**. v1's `0.01` stop is dissolved, not merely tightened.
  - **GP11b (residual bound → gate tolerance)** — the v1(0.01)-vs-v2 share gap equals the
    unreinvested tail's value, `gap = leftover/(price_sum·(1−ρ_tail))`. Measured **gap/(0.01/price)
    = 1.000** on every fixture: the geometric recycle factor `1/(1−ρ_tail)→1` because the arb
    surplus funding the next iteration is **second-order in the overshoot**, which → 0 at the
    tail. So `0.01/price` is not just leading-order — it's **asymptotically exact**. ⇒ the PR2
    v1↔v2 no-limit equivalence gate is **per-case** `|shares_v1 − shares_v2| ≤ threshold /
    (YES price sum) [= 0.01/price]` — data-dependent (Decision #5), bounded, NOT an arbitrary
    epsilon. The **cost** field is exact (both pay `B`); only acquired shares / pools differ.
    Worst gap on this sweep: **4.96e-3 shares** (the skewed-pool / cheap-YES cases — small price
    ⇒ wider tail). *(Distinct from GP8, which anchors v2 vs. our own already-exact solver — no
    0.01 loop — to machine precision.)* Rationale: `pr2-plan.md` Decision #5.

## Limit-aware multi-buy equilibrium — `proofs/monotone_equilibrium.py` ✅ (symbolic + numerical, GP12)
- **GP12** — **the limit-aware sum-to-one multi-buy equilibrium EXISTS, is UNIQUE, and each
  answer's price is MONOTONE in the bet size `t`.** This is the correctness foundation for 2b.2
  (the reverse-limit fix on the one buggy path, `calculateCpmmMultiArbitrageBetsYes`). If
  monotone, the v1 overshoot (a traded answer settles 0.399 but *peaks 0.677* mid-iteration) is a
  provable **algorithm artifact**, not an equilibrium property — so a single net move crosses each
  resting maker exactly once, the ordinary single-move fill (already correct for a monotone
  single-answer buy) pins right, and the fix is licensed. Approaches **A** (reverse-limit
  injection) and **C** (direct monotone solve) target this **one** equilibrium ⇒ the choice is
  implementation style, not correctness. Modeled at **p=½** (the bug + the entire vendor spec are
  p=½). Four parts:
  - **GP12a (symbolic)** — the whole result rests on one identity: at p=½ the marginal YES-buy and
    NO-buy rates are **equal**, `a_i := dprob_i/d(YES share) = 2YᵢNᵢ/(Yᵢ+Nᵢ)³ = −dprob_i/d(NO
    share)`. From it (implicit-function theorem on `Σp(g,η)=1`, g = equal YES shares per basket
    answer, η = equal NO shares in all) every comparative statics is **signed in closed form**:
    `dη/dg = (Σ_basket a)/(Σ_all a) ∈ (0,1)`; basket `dprob_i/dg = aᵢ·(Σ_{non-basket} a)/(Σ_all a)
    > 0`; non-basket `dprob_j/dg = −aⱼ·(Σ_basket a)/(Σ_all a) < 0`; and net spend `dt/dg =
    Σ_basket prob_i > 0` (the `η(n−1)` redemption **exactly cancels** the arb's marginal cost —
    this is precisely vendor's `yesSharePriceSum`). ⇒ no-limit, every price is strictly monotone in
    `t`; `t` strictly increasing in `g` ⇒ the equilibrium at any budget is unique. *(`a_i=b_i` is
    special to p=½ — see GP12 scope note.)*
  - **GP12b (numerical)** — those closed-form derivatives match finite differences of the actual
    `(g,η)` solver to **2.2e-7** on a skewed 5-answer config (g∈{10,40,90}); the symbolic law
    governs the real mechanism, not an idealization.
  - **GP12c (numerical/constructive)** — the **cheapest-monotone-flow oracle** (Evan's "buy the
    cheapest mix holding Σp=1 infinitesimally, then redeem via identity": spend the budget in tiny
    chunks, each a hair of basket-YES + an arb to Σp=1, so prices creep monotonically with no
    overshoot) (1) reproduces the vendor fixed point **and** its **0.677 peak to the digit** (so it
    models the real path); (2) under limits **pins correctly** — large in-path NO-ask@0.30 → a0
    pins at 0.30; past-final ask@0.50 → **unfilled**; small ask crossed; a2 large YES-bid pins —
    i.e. the **two vendor bug cases resolve and the controls hold**; (3) is monotone in `t` under
    limits — a **two-grid Richardson** check over 40 fuzzed configs (with limit books) shows every
    residual reversal **halves when the step halves** (ratio mean 1.99), i.e. it is O(dg)
    discretization error → 0, so the **continuum** equilibrium price is monotone.
  - **GP12d (numerical)** — side-by-side on the canonical pinning case: the **monotone oracle**
    crosses the maker once and pins at 0.30 (fill 24.3); the **vendor iteration** drags a0 *below*
    0.30 (settles 0.172) over-consuming the maker in place (fill 61.2), and on the past-final
    ask@0.50 keeps a **25.8 transient fill** the net move never reaches. The reverse-limit fix ==
    single net-move fills == the GP12 equilibrium.
  - **Scope / general p** — `a_i=b_i` is the p=½ YES/NO marginal symmetry; for general per-answer
    `p` the rates differ, so GP12a's clean individual-basket sign argument is p=½-specific. What
    carries to general p: **GP5a** (Σp strictly monotone in η ⇒ existence+uniqueness of the arb,
    any p), the **non-basket + basket-sum** monotonicity (p-agnostic), and the
    **net-path/single-crossing** limit argument (GP12c/d, p-agnostic). The only open piece for a
    fully general-p GP12 is individual-basket `dprob_i/dg > 0` without `a_i=b_i` — and that is
    **numerically confirmed** (general-p no-limit fuzz: 120 configs, per-answer p∈(0.12,0.88),
    308 basket-answer checks, worst basket-prob drop **4.4e-16**), formalization pending (needs
    the general-p analog of the GP12a sign argument, an `a_i/b_i` ratio bound). The 2b.2 bug, the
    spec, and the immediate fix are all p=½, so p=½ is load-bearing; general p is a noted
    follow-on.

## Creation liquidity — `proofs/creation_liquidity_proofs.py` ✅ (symbolic, verified) + `creation_liquidity{,_optimum,_solver}.py` (numerical)
How a v2 market should choose its creation pools (the onboarding "DEFERRED MATH"). Full
write-up + numbers: `tasks/cpmm_multi_2/creation-liquidity-findings.md`.
- **GP13** — **point liquidity (pre-auto-arb, one pool):** `a = dprob/dshares =
  p(1−p)YN/W³ = q(1−q)/W`, `W=(1−p)Y+pN`; with `p` eliminated `a = q(1−q)(qY+(1−q)N)/(YN)`.
  Reduces to `2YN/(Y+N)³` at p=½; symmetric under `(Y,N,p)→(N,Y,1−p)` (shares is the YES/NO-
  symmetric coordinate; mana is not). ⇒ **at fixed prob, max liquidity ⟺ max `W=(1−p)Y+pN`**
  — the true depth metric (corrects the stored `√(Y·N)`). Symbolic, derived from the GP2 buy
  mechanic; numeric finite-diff anchor to 4e-5.
- **GP14** — **point liquidity (post-auto-arb, single-answer YES buy):** `aᵢᵉᶠᶠ =
  aᵢ(Σⱼaⱼ − aᵢ)/Σⱼaⱼ`, from the `η` complete-set-of-NO redemption restoring `Σq=1`. Symbolic
  (linear solve); numeric vs the auto-arb solver to 7e-4. The optimum is the same whether
  optimized on `a` or `aᵉᶠᶠ` (gains within ~1–3pp), so the pre-arb solve is a faithful proxy.
- **GP15** — **basket-funded creation optimum.** (a) all-winners-tight funding ⟹ `Yᵢ−Nᵢ=D`
  (const). (b) `min Σ cᵢ/Wᵢ` under a budget on `ΣWᵢ` ⟹ `Wᵢ ∝ √cᵢ`, and `cᵢ=qᵢ(1−qᵢ)=
  Var(Bernoulli(qᵢ))` ⇒ the **"depth ∝ outcome standard deviation"** rule. Numeric: the √c-depth
  rule captures **95.5–99.7%** of the balanced→optimum gain (gains are 11–55%, growing with n);
  the reduced unconstrained solve (funding baked in via `D=ante−ΣN`) finds the exact optimum
  robustly. **Set/independent markets:** at fixed per-answer risk `max(Y,N)`, balanced `p=q` is
  the verified optimum = exactly Manifold binary CPMM (balanced pool, `p=initialProb`).

## Multi-target (m>1 basket) redemption — `proofs/multi_target_redemption.py` ✅ (symbolic + numerical, GP16)
The redemption share-split for a multi-answer YES **basket** buy (the `/v0/multi-bet` path,
`calculateCpmmMultiArbitrageBetsYesV2`). The 2025 paper formalized only the **single**-target
"NO in others" redemption ("η YES in target"); the v2 basket extension split that credit as
**η/m per basket answer**, which **destroys mana** at resolution to a basket answer (found on the
dev instance: sole-participant payout ~10% short; v1 conserves). Full write-up +
dev-instance differential: `tasks/cpmm_multi_2/findings-m1-basket-conservation-bug-2026-06-28.md`.
- **GP16a** — the Manifold CPMM buy shifts `(Y−N)` by **−s** (YES) / **+s** (NO) exactly, for any
  `p`/amount/pool (from the `newY=y+b−s, newN=n+b` mechanic; symbolic).
- **GP16b** — holding `η` NO in each of the `(n−m)` other answers pays `η(n−m)` iff a basket
  answer wins, `η(n−m−1)` iff an other wins. Floor `η(n−m−1)` is redeemed (matches v2's credit);
  the **residual "pays η iff ANY basket answer wins" is η YES shares in EACH basket answer**, not
  η/m. η/m underpays a winning basket answer by `η(m−1)/m`. (symbolic)
- **GP16c** — sum-to-one resolution conserves **iff** `D_i := T_i^YES − T_i^NO` is constant across
  answers (`T_i^YES=poolYes_i+traderYES_i`, etc.). Tracing the basket buy via GP16a, `D` shifts by
  `credit` on basket answers and by `η` on others ⇒ uniform **iff credit = η**. General-p (no `p`
  appears). For m=1, `η/m = η` ⇒ the single-answer path was always correct. (symbolic)
- **GP16d** — numerical end-to-end: a v2 sum-to-one basket buy with `η/m` leaks on basket winners
  (payout spread > 0, mirrors the dev-instance pattern), with `η` conserves to machine precision.
- **Fix:** `shares: eta / m` → `shares: eta` in `calculateCpmmMultiArbitrageBetsYesV2` (vendor;
  mana split stays `netOthers/m` so total cost == betAmount). jest regression
  (`calculate-cpmm-arbitrage.test.ts` "m>1 basket conserves at resolution") asserts the
  payout-constant-across-winners invariant; verified to FAIL under η/m, pass under η.

### GP17 — whole-market √variance liquidity-add split (`liquidity_add_split.py`)
The current v2 whole-market add equal-splits the subsidy (`amount/n`, lossless float-`p` binary add
per answer) — **LMSR/balanced-shaped at the margin**, inconsistent with the √variance *creation*
rule (depth `W_i ∝ √(q_i(1−q_i))`). The creation-consistent add (Evan: "apply creation's allocation
to the *added* mana at current probs; don't rearrange existing depth") is the **MERGE rule**: for an
add of `Δ` to a market currently pricing at `q` (current probs, Σq=1),
`new_pool_i = existing_i + cpmmMulti2SumToOnePools(q, Δ)[i]` (reservewise), then re-price each `p_i`
to hold prob = `q_i`. Verified claims (float64, machine-precision residuals):
- **GP17a (additivity)** — `cpmmMulti2SumToOnePools` is **homogeneous degree-1 in ante** (reserves
  ∝ ante, `p` invariant) ⇒ `pools(q,A)+pools(q,Δ) = pools(q,A+Δ)`, so the merge is well-defined.
- **GP17b (prob preservation)** — a **unique** `p_i` re-prices any positive merged pool to `q_i`
  (odds `(p/(1−p))(N/Y)` strictly ↑ in `p`); Σ=1 is **inherited** (each prob individually held),
  not imposed — shown on an arbitrary non-basket reserve state too.
- **GP17c (conservation)** — the Δ-creation satisfies the all-winners-tight identity
  `Y_i+Σ_{j≠i}N_j = Δ` (locks exactly Δ); CHOOSE_ONE payout is **linear in reserves** ⇒ merging
  **superposes** two independently-conservative markets, so the add contributes exactly the LP's Δ
  (re-pricing `p` moves no reserves — mana-free).
- **GP17d (reductions)** — on an **untraded** market merge(Δ) = create(A+Δ) = scale by (A+Δ)/A
  (the "probs unchanged ⇒ just scale" case); at **n=2** creation is balanced so merge = the current
  equal-split exactly (no behaviour change); **Set/independent** creation is balanced ⇒ merge =
  equal-split, so Set is **left unchanged** — only sum-to-one moves to √variance.
- **GP17e (shape)** — reproduces `findings-liquidity-add-split-2026-06-28` numbers: equal-split adds
  near-flat depth (`Δk≈[119,115,108,102]`, ratio 1.17), merge concentrates in uncertain answers
  (`Δk=[274,159,80,55]`, ratio 4.98). **Drizzle inherits this** (it calls the whole-market add) —
  a feature: drizzle keeps the market on the √variance manifold instead of drifting toward balanced.
Full write-up: `tasks/cpmm_multi_2/findings-liquidity-add-split-2026-06-28.md` (§Derivation).

### GP18 — "Other" split as a REFINEMENT (`other_split_refinement.py`)
Adding an answer A to a sum-to-one market carves it out of the catch-all "Other". The right model
(Evan): treat Other as the event `(A or Other')` and *refine* it — the operation must be invariant
under anything that can't distinguish A from Other'. Proved via **share identities** (the conservation
is automatic, not emergent — no "backing" gymnastics):
- **GP18a (redemption identity)** — in sum-to-one, `NO-i ≡ Σ_{j≠i} YES-j` (both pay 1 iff i loses).
  Since Other's complement is the listed answers, `NO-Other ≡ Σ YES-listed`.
- **Solvency in share terms** — `D_i = (ΣYES)−(ΣNO)` over ALL holders incl the pool; solvent ⟺ D_i
  constant across answers (M = D + ΣTN_i then constant per winner). D_i is trade-invariant.
- **GP18b (construction)** — naively partitioning Other's reserves into A,Other' HALVES D
  (insolvent); the correct construction copies Other's YES inventory into BOTH new pools and funds
  their NO from `answerCost` (balanced, D-preserving), each re-priced to `p_o/2`. Σ prob = 1.
- **GP18c (payout invariance)** — every trader (YES/NO-Other, YES/NO-listed, mixed) gets the
  coarse-original payout under the A,Other'→Other coarsening, for every outcome; no trader can tell
  A from Other'.
- **GP18d (uniform relabel, pool included)** — the whole split is ONE relabel applied to every
  holder, the pool too: `YES-O→YES-A+YES-O'`, `NO-O→ΣYES-listed`. D stays constant, mana conserved
  exactly, and the **listed pools' reserves/prices never move** (the relabeled NO-O shares live on
  the holders/LP, not in listed pools). v1's excess-NO dump onto listed pools (GP6c, ~3pt shift) is
  a p=0.5 artifact, not required.
- **GP18e (liquidity DOF)** — fixing all probs (incl prob_A=prob_O') + solvency + conservation
  leaves EXACTLY one continuous DOF: the A↔Other' depth split at fixed added mana (per-answer p
  decouples reserves from price). Symmetric (50/50) chosen; the √variance-graded depth (O' at p_o/2
  ≈ 0.75× its p_o depth) is a valid interior point.
- **GP18f (No must relabel)** — keeping the pool's NO in both new pools is solvent (D const) but
  CREATES exactly `No` mana; relabeling NO→YES-listed conserves (M_new = M_old + answerCost). Pins
  the construction.
- **GP18g (two NO routes)** — `YES-i+NO-i=1 M$` ⇒ `NO-A+NO-O' = NO-O + 1 M$`/share: (a) relabel
  `NO-O→ΣYES-listed` is FREE (shipped); (b) duplicate `NO-O→NO-A+NO-O'` COSTS `No` (deeper NO).
  `poolNo` IS the pool's directional shares — no special object needed.

GP6 (`other_split.py`) covers the pool-split losslessness via per-answer p; GP18 adds the
refinement-invariance + conservation framing. **Implemented + instance-validated** (vendor
`ef4101099`; `createAnswerAndSumAnswersToOneV2`; net=0 across all resolves).

*Running the proofs:* `creation_liquidity_proofs.py` needs `sympy`; the numerical companions need
`numpy`/`scipy` (plus the project `manifold` package for the auto-arb oracle).
`multi_target_redemption.py` needs `sympy` + `scipy`.
