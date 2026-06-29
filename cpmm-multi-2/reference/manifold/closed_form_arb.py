"""Closed-form direct auto-arbitrage — cpmm-multi-2 PR1 reference (p=0.5/no-limit).

The **reference implementation** of the auto-arb-algorithms.tex three
parameterizations, written fresh from the GP theorems. PR1 ports these to
upstream TypeScript to replace vendor's O(n log² N) nested binary search with an
O(n log N) direct-formula path. Three public entry points, one per input axis:

  - `calculate_purchase_with_arbitrage_direct`     (dollar-centric — cost in)
  - `calculate_shares_with_arbitrage_direct`       (share-centric  — shares in)
  - `calculate_probability_with_arbitrage_direct`  (prob-centric   — target prob in)

They are three parameterizations of ONE equilibrium trade: feed one's output as
another's input and recover an identical (shares, cost, prob) triple and final
pools (GP10, `proofs/parameterization_equivalence.py`).

Validation (GP8-style differential, each independent):
  - dollar  vs `calculate_purchase_with_arbitrage`  (`test_gp8_direct_equivalence.py`)
  - share   vs `calculate_shares_exact`             (`test_slice4_parameterizations.py`)
  - prob    vs `MarketSimulator.buy_to_probability` (`test_slice4_parameterizations.py`)

Scope (deliberately narrow): **pure CPMM, p = 0.5, no limit orders** — the regime
where vendor v1 and the direct path provably agree (math plan item 3 /
`tasks/cpmm_multi_2/proofs/`). General per-answer p and reversible limits are
later slices. The closed forms here are written **fresh** from the GP theorems
(not routed through `amm_core`), so the differential tests cross-check two
*independent* implementations of the same math rather than one wrapper of the other.

Closed forms (p = 1/2, CPMM invariant Y·N = k):
  - Buy YES with amount A in (Y, N):  shares S = Y + A - Y·N/(N+A);
        pool → (Y·N/(N+A), N+A).                          [GP3 / paper eq. 104]
  - Buy NO  with amount A in (Y, N):  shares S = N + A - Y·N/(Y+A);
        pool → (Y+A, Y·N/(Y+A)).                          [GP3, Y↔N mirror]
  - Buy η YES shares in (Y, N):  cost Cʸ(η) =
        (η - Y - N + sqrt((Y+N-η)² + 4·N·η)) / 2; pool → (Y+C-η, N+C).
  - Buy η NO  shares in (Y, N):  cost Cᴺ(η) =
        (η - Y - N + sqrt((Y+N-η)² + 4·Y·η)) / 2; pool → (Y+C, N+C-η).   [GP5b]
  - prob_YES = N / (Y + N).                                [GP1 at p=1/2]

Dollar-centric Algorithm 1 (Strategy 1), matching `execute_arbitrage_trades`:
  buy `amount` of `position` in the target answer; arbitrage = η shares of the
  OPPOSITE position in each of the (n-1) other answers. For a YES purchase the
  redemption identity returns η YES in the target + η·(n-2) cash; the leftover
  budget (amount - net_arb) is the primary purchase in the target. η is the
  unique root of Σ p(η) = 1 — Σp is strictly monotone in η (GP5a) so a single
  bisection finds it, and GP5d dissolves v1's leftover-mana loop so we can drive
  it to machine precision (no `MANIFOLD_ARB_EPSILON` clamp).

Returns the same `ArbitrageResult` shape as `calculate_purchase_with_arbitrage`,
so the two are directly comparable field-by-field.
"""

import math

from manifold.multi_choice import ArbitrageResult

Pool = dict[str, float]
Pools = dict[str, Pool]

# Drive the η bisection to machine precision. Unlike production (which stops at
# MANIFOLD_ARB_EPSILON to mirror vendor's floatingArbitrageEqual), the direct
# path has no leftover-mana loop (GP5d) so the root is reachable to ~1e-13.
_SIGMA_TOL = 1e-13
_MAX_BISECT = 200


# --------------------------------------------------------------------------- #
# Single-pool closed forms (p = 1/2). Written fresh from the GP theorems.
# --------------------------------------------------------------------------- #
def _prob_yes(Y: float, N: float) -> float:
    """P(YES) for a balanced (p=1/2) pool."""
    return N / (Y + N)


def _buy_amount(Y: float, N: float, amount: float, position: str) -> tuple[float, Pool]:
    """Spend `amount` buying `position`. Returns (shares_out, new_pool).

    YES: pool → (Y·N/(N+A), N+A);  NO: pool → (Y+A, Y·N/(Y+A)).
    """
    k = Y * N
    if position == "YES":
        n_new = N + amount
        y_new = k / n_new
        shares = Y + amount - y_new
    else:  # NO
        y_new = Y + amount
        n_new = k / y_new
        shares = N + amount - n_new
    return shares, {"YES": y_new, "NO": n_new}


def _cost_for_shares(Y: float, N: float, shares: float, position: str) -> tuple[float, Pool]:
    """Cost to buy `shares` of `position` (η shares). Returns (cost, new_pool).

    Buying η YES:  C = (η - Y - N + sqrt((Y+N-η)² + 4Nη))/2, pool → (Y+C-η, N+C).
    Buying η NO:   C = (η - Y - N + sqrt((Y+N-η)² + 4Yη))/2, pool → (Y+C, N+C-η).
    """
    s = Y + N - shares
    if position == "YES":
        cost = (shares - Y - N + math.sqrt(s * s + 4.0 * N * shares)) / 2.0
        return cost, {"YES": Y + cost - shares, "NO": N + cost}
    else:  # NO
        cost = (shares - Y - N + math.sqrt(s * s + 4.0 * Y * shares)) / 2.0
        return cost, {"YES": Y + cost, "NO": N + cost - shares}


def _shares_for_prob(Y: float, N: float, target_prob: float, position: str) -> float:
    """Shares of `position` to buy in (Y, N) to move P(YES) to `target_prob`.

    Inverts p = f(Y, N) for the post-buy pool (SharesForProb in the paper,
    Algorithm 3). Buying YES *raises* P(YES); buying NO *lowers* it. With
    Y·N = k held, a YES buy of amount A gives pool (k/(N+A), N+A) so
    P(YES) = (N+A)²/(k+(N+A)²) = target ⇒ N+A = sqrt(target·k/(1-target)).
    The NO case is the Y↔N, target↔(1-target) mirror.
    """
    k = Y * N
    if position == "YES":
        n_new = math.sqrt(target_prob * k / (1.0 - target_prob))
        amount = n_new - N
        return Y + amount - k / n_new  # shares = Y + A - Y_new
    else:  # NO: lower P(YES) -> raise P(NO) = 1 - target_prob
        y_new = math.sqrt((1.0 - target_prob) * k / target_prob)
        amount = y_new - Y
        return N + amount - k / y_new


def _bisect_sigma(sigma_fn, hint: float) -> float:
    """Find η ≥ 0 with Σp(η) = 1, given a strictly-monotone Σp(η) (GP5a).

    `sigma_fn(η)` returns Σp for a candidate arb size η. η = 0 is the
    un-arbitraged perturbation; we bracket [0, hi] (growing hi until the sign of
    Σp-1 flips) and bisect to machine precision. Returns 0.0 if no sign change is
    reachable (degenerate / no-arb case). Shared by all three parameterizations.
    """
    def f(eta: float) -> float:
        return sigma_fn(eta) - 1.0

    f0 = f(0.0)
    if abs(f0) < _SIGMA_TOL:
        return 0.0
    lo, hi = 0.0, max(1.0, abs(hint))
    f_hi = f(hi)
    grow = 0
    while f0 * f_hi > 0.0 and grow < 60:
        hi *= 2.0
        f_hi = f(hi)
        grow += 1
    if f0 * f_hi > 0.0:
        return 0.0  # no sign change — no arb

    eta = 0.0
    for _ in range(_MAX_BISECT):
        mid = 0.5 * (lo + hi)
        if mid in (lo, hi):
            break
        f_mid = f(mid)
        if abs(f_mid) < _SIGMA_TOL:
            return mid
        if (f_mid > 0.0) == (f0 > 0.0):
            lo = mid
        else:
            hi = mid
        eta = mid
    return eta


# --------------------------------------------------------------------------- #
# Dollar-centric Algorithm 1 (Strategy 1) — the direct equilibrium path.
# --------------------------------------------------------------------------- #
def _state_for_eta(
    pools: Pools, target: str, amount: float, eta: float, position: str
) -> dict:
    """Full market state for a candidate arbitrage size η (Strategy 1).

    Mirrors `execute_arbitrage_trades` but with inline closed forms. Returns
    target_shares, per-answer new pools, and Σp (so the bisection can read it).
    """
    arb_dir = "NO" if position == "YES" else "YES"
    other_ids = [a for a in pools if a != target]
    n = len(pools)

    new_pools: Pools = {}
    total_arb = 0.0
    for aid in other_ids:
        Y, N = pools[aid]["YES"], pools[aid]["NO"]
        if eta > 0.0:
            cost, pool = _cost_for_shares(Y, N, eta, arb_dir)
            new_pools[aid] = pool
            total_arb += cost
        else:
            new_pools[aid] = {"YES": Y, "NO": N}

    # Redemption identity: YES purchase frees η·(n-2) cash; NO purchase frees none.
    redeemed = eta * (n - 2) if position == "YES" else 0.0
    net_arb = total_arb - redeemed
    primary = amount - net_arb

    Yt, Nt = pools[target]["YES"], pools[target]["NO"]
    if primary > 0.0:
        primary_shares, target_pool = _buy_amount(Yt, Nt, primary, position)
    else:
        primary_shares, target_pool = 0.0, {"YES": Yt, "NO": Nt}
    new_pools[target] = target_pool

    target_shares = primary_shares + eta
    sigma = sum(_prob_yes(p["YES"], p["NO"]) for p in new_pools.values())
    return {"target_shares": target_shares, "pools": new_pools, "sigma": sigma}


def calculate_purchase_with_arbitrage_direct(
    pools: Pools, answer_id: str, amount: float, position: str = "YES"
) -> ArbitrageResult:
    """Direct closed-form auto-arb for a p=0.5 / no-limit multi-choice purchase.

    Independent reference for `calculate_purchase_with_arbitrage` on pure-CPMM
    cases (GP8). `pools` is a plain ``{answer_id: {"YES": float, "NO": float}}``
    dict — no limit orders, every answer p=0.5.

    Raises:
        ValueError: < 2 answers, unknown target, or a broken (Σp far from 1) book.
    """
    if position not in ("YES", "NO"):
        raise ValueError(f"position must be 'YES' or 'NO', got {position!r}")
    if answer_id not in pools:
        raise ValueError(f"answer_id {answer_id!r} not in pools")
    n = len(pools)
    if n < 2:
        raise ValueError("Multi-choice markets must have at least 2 answers")

    initial_sigma = sum(_prob_yes(p["YES"], p["NO"]) for p in pools.values())
    if abs(initial_sigma - 1.0) > 0.05:
        raise ValueError(
            f"Broken market: probabilities sum to {initial_sigma:.3f}, not 1.0."
        )

    def result_from(state: dict) -> ArbitrageResult:
        return _build_result(
            pools, answer_id, position, state["pools"],
            shares_bought=state["target_shares"], cost=amount,
            eta=state.get("eta", 0.0),
        )

    # Tiny amount → direct purchase, no arb (mirrors production's early return).
    if abs(amount) < _SIGMA_TOL:
        st = _state_for_eta(pools, answer_id, amount, 0.0, position)
        st["eta"] = 0.0
        return result_from(st)

    # η = 0 is the un-arbitraged perturbation. Buying YES pushes Σp > 1 (root at
    # η>0 lowers it); buying NO pushes Σp < 1 (root at η>0 raises it). Σp is
    # strictly monotone in η (GP5a) so a single bisection finds the root.
    eta = _bisect_sigma(
        lambda e: _state_for_eta(pools, answer_id, amount, e, position)["sigma"],
        hint=abs(amount),
    )
    st = _state_for_eta(pools, answer_id, amount, eta, position)
    st["eta"] = eta
    return result_from(st)


# --------------------------------------------------------------------------- #
# Shared result builder.
# --------------------------------------------------------------------------- #
def _build_result(
    pools_in: Pools,
    answer_id: str,
    position: str,
    new_pools: Pools,
    *,
    shares_bought: float,
    cost: float,
    eta: float,
) -> ArbitrageResult:
    """Assemble an ArbitrageResult. fills: target signed by position, others by
    the opposite (the arb leg). All three parameterizations reach the same final
    pools, so the dollar-style fills convention applies uniformly."""
    probs = {a: _prob_yes(p["YES"], p["NO"]) for a, p in new_pools.items()}
    fills: dict[str, float] = {}
    for a in pools_in:
        if a == answer_id:
            fills[a] = shares_bought if position == "YES" else -shares_bought
        else:
            fills[a] = (-eta if position == "YES" else eta) if eta else 0.0
    return ArbitrageResult(
        shares_bought=shares_bought, cost=cost, fills=fills,
        new_pools=new_pools, new_probabilities=probs,
    )


# --------------------------------------------------------------------------- #
# Share-centric Algorithm 2 — buy exact shares in target, arb NO in ALL n.
# --------------------------------------------------------------------------- #
def _state_for_eta_share(
    pools: Pools, target: str, target_shares: float, eta: float, position: str
) -> dict:
    """Paper Algorithm 2: buy `target_shares` in target first (outside the
    search), then η arb in ALL n answers. Redemption is the "in all" constant
    (GP5d): η(n-1) for a YES purchase (NO-in-all), η·1 for a NO purchase
    (YES-in-all, only the winner pays)."""
    arb_dir = "NO" if position == "YES" else "YES"
    n = len(pools)
    Yt, Nt = pools[target]["YES"], pools[target]["NO"]
    c_primary, target_pool = _cost_for_shares(Yt, Nt, target_shares, position)

    # Post-primary pools: target updated, others untouched. Arb hits all n.
    base = {a: (target_pool if a == target else pools[a]) for a in pools}
    new_pools: Pools = {}
    c_arb = 0.0
    for aid, pool in base.items():
        Y, N = pool["YES"], pool["NO"]
        if eta > 0.0:
            cost, p = _cost_for_shares(Y, N, eta, arb_dir)
            new_pools[aid] = p
            c_arb += cost
        else:
            new_pools[aid] = {"YES": Y, "NO": N}

    redemption = eta * (n - 1) if position == "YES" else eta
    cost = c_primary + c_arb - redemption
    sigma = sum(_prob_yes(p["YES"], p["NO"]) for p in new_pools.values())
    return {"cost": cost, "pools": new_pools, "sigma": sigma}


def calculate_shares_with_arbitrage_direct(
    pools: Pools, answer_id: str, target_shares: float, position: str = "YES"
) -> ArbitrageResult:
    """Share-centric direct closed-form auto-arb (paper Algorithm 2).

    Buy exactly `target_shares` of `position` in `answer_id`; the result is the
    balanced (Σp=1) market and the cost it took. Independent reference for
    `multi_choice.calculate_shares_exact` on p=0.5 / no-limit cases.
    """
    _validate(pools, answer_id, position)
    if abs(target_shares) < _SIGMA_TOL:
        st = _state_for_eta_share(pools, answer_id, target_shares, 0.0, position)
        return _build_result(pools, answer_id, position, st["pools"],
                             shares_bought=target_shares, cost=st["cost"], eta=0.0)

    eta = _bisect_sigma(
        lambda e: _state_for_eta_share(pools, answer_id, target_shares, e, position)["sigma"],
        hint=abs(target_shares),
    )
    st = _state_for_eta_share(pools, answer_id, target_shares, eta, position)
    return _build_result(pools, answer_id, position, st["pools"],
                         shares_bought=target_shares, cost=st["cost"], eta=eta)


# --------------------------------------------------------------------------- #
# Probability-centric Algorithm 3 — reach p_target in target, arb NO in others.
# --------------------------------------------------------------------------- #
def _state_for_eta_prob(
    pools: Pools, target: str, target_prob: float, eta: float, position: str
) -> dict:
    """Paper Algorithm 3: buy δ in target to reach `target_prob` first (outside
    the search), then η arb NO in OTHERS only. The target pool is untouched by
    the arb, so its probability stays exactly `target_prob` (Target Probability
    Invariance). Redemption mirrors the dollar-centric (in-others) convention:
    η(n-2) for YES, 0 for NO."""
    arb_dir = "NO" if position == "YES" else "YES"
    n = len(pools)
    other_ids = [a for a in pools if a != target]
    Yt, Nt = pools[target]["YES"], pools[target]["NO"]
    delta = _shares_for_prob(Yt, Nt, target_prob, position)
    c_primary, target_pool = _cost_for_shares(Yt, Nt, delta, position)

    new_pools: Pools = {target: target_pool}
    c_arb = 0.0
    for aid in other_ids:
        Y, N = pools[aid]["YES"], pools[aid]["NO"]
        if eta > 0.0:
            cost, p = _cost_for_shares(Y, N, eta, arb_dir)
            new_pools[aid] = p
            c_arb += cost
        else:
            new_pools[aid] = {"YES": Y, "NO": N}

    redemption = eta * (n - 2) if position == "YES" else 0.0
    cost = c_primary + c_arb - redemption
    sigma = sum(_prob_yes(p["YES"], p["NO"]) for p in new_pools.values())
    return {"delta": delta, "cost": cost, "pools": new_pools, "sigma": sigma}


def calculate_probability_with_arbitrage_direct(
    pools: Pools, answer_id: str, target_prob: float, position: str = "YES"
) -> ArbitrageResult:
    """Probability-centric direct closed-form auto-arb (paper Algorithm 3).

    Move `answer_id` to `target_prob`, then arbitrage the others back to Σp=1
    (target prob fixed). `shares_bought = δ + η`. Independent reference for
    `MarketSimulator.buy_to_probability` on p=0.5 / no-limit cases.
    """
    _validate(pools, answer_id, position)
    if not (0.0 < target_prob < 1.0):
        raise ValueError(f"target_prob must be in (0,1), got {target_prob}")

    eta = _bisect_sigma(
        lambda e: _state_for_eta_prob(pools, answer_id, target_prob, e, position)["sigma"],
        hint=1.0,
    )
    st = _state_for_eta_prob(pools, answer_id, target_prob, eta, position)
    shares_bought = st["delta"] + eta
    return _build_result(pools, answer_id, position, st["pools"],
                         shares_bought=shares_bought, cost=st["cost"], eta=eta)


def _validate(pools: Pools, answer_id: str, position: str) -> None:
    """Shared input guard for the share-/prob-centric entry points."""
    if position not in ("YES", "NO"):
        raise ValueError(f"position must be 'YES' or 'NO', got {position!r}")
    if answer_id not in pools:
        raise ValueError(f"answer_id {answer_id!r} not in pools")
    if len(pools) < 2:
        raise ValueError("Multi-choice markets must have at least 2 answers")
    initial_sigma = sum(_prob_yes(p["YES"], p["NO"]) for p in pools.values())
    if abs(initial_sigma - 1.0) > 0.05:
        raise ValueError(
            f"Broken market: probabilities sum to {initial_sigma:.3f}, not 1.0."
        )
