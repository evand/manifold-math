"""Multi-choice market arbitrage implementation following Manifold's algorithm.

This is the library version of the arbitrage logic, properly integrated
with the Market abstraction.

Uses amm_core for fundamental AMM calculations to maintain single source of truth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from manifold.amm_core import cost_for_shares, pool_after_trade, probability_from_pool
from manifold.constants import MANIFOLD_ARB_EPSILON

if TYPE_CHECKING:
    from collections.abc import Callable

    from manifold.liquidity import LiquidityContext


class AutoArbConvergenceError(RuntimeError):
    """Auto-arbitrage binary search failed to converge to Σp = 1.0.

    Raised when:
      - The bracket-expansion phase cannot find a sign flip for evaluate(η)
        within the maximum doublings (i.e., no root reachable).
      - The post-search residual |Σp - 1| exceeds the legitimate bound
        derived from the η-clamp threshold (`MANIFOLD_ARB_EPSILON * Σ p(1-p)/T`).

    Both conditions indicate a real numerical breakdown — either the cost
    function isn't monotonic (limit-order kink we don't model), the bracket
    grew unboundedly without crossing zero (pool exhaustion mid-search), or
    the implementation has a bug. Callers that want best-effort behavior may
    catch this and fall back; safety-critical callers should propagate.
    """

# Set up module logger
logger = logging.getLogger(__name__)


def _apply_arb_trades(
    ctx: LiquidityContext,
    answer_ids: list[str],
    eta: float,
    arb_direction: str,
    *,
    skip_id: str | None = None,
) -> float:
    """Apply η arb trades to answers. Returns total cost.

    Layer: 2, User-facing: NO

    This helper eliminates duplicated arb-loop code across auto-arb functions.

    Args:
        ctx: Context to MUTATE with trades
        answer_ids: All answer IDs to trade in
        eta: Shares to trade (positive = arb_direction, negative = opposite)
        arb_direction: "YES" or "NO" - direction for positive η
        skip_id: Optional answer ID to skip (for Strategy 1 algorithms)

    Returns:
        Total cost of all trades applied
    """
    total_cost = 0.0
    opposite = "YES" if arb_direction == "NO" else "NO"

    for aid in answer_ids:
        if skip_id and aid == skip_id:
            continue
        try:
            if eta > 0:
                cost = ctx.cost_for_shares(aid, eta, arb_direction)
                ctx.apply_trade(aid, cost, arb_direction)
                total_cost += cost
            elif eta < 0:
                cost = ctx.cost_for_shares(aid, -eta, opposite)
                ctx.apply_trade(aid, cost, opposite)
                total_cost += cost
        except (ValueError, ZeroDivisionError):
            continue

    return total_cost


def _solve_eta_for_zero(
    evaluate: Callable[[float], float],
    initial_step: float,
    *,
    clamp_threshold: float = MANIFOLD_ARB_EPSILON,
    max_expansions: int = 40,
    max_bisect: int = 60,
) -> float:
    """Find η such that evaluate(η) ≈ 0, assuming evaluate is monotonic in η.

    Direction-agnostic: detects the bracket by sign-flip on either side of zero.
    Replaces the position-baked-in bracket logic that previously failed for
    position="NO" cases (where evaluate_eta is increasing in η, opposite of
    the position="YES" assumption hardcoded in the prior expansion conditions).

    Args:
        evaluate: A function evaluate(η) -> float that should be approximately monotonic.
            evaluate(0) determines the side of zero we're starting from; the helper
            seeks η with evaluate(η) of the opposite sign and bisects.
        initial_step: Hint at the magnitude of |η| to begin probing. Bracket expands
            geometrically (doubling per step) up to max_expansions doublings on each side.
        clamp_threshold: |η| values smaller than this are returned as exactly 0.
            This matches Manifold's `floatingArbitrageEqual` epsilon
            (vendor/manifold/common/src/calculate-cpmm-arbitrage.ts) — when η is
            small enough, Manifold skips auto-arb entirely.
        max_expansions: Bracket-expansion iterations on each side.
        max_bisect: Bisection iterations once a bracket is found.

    Returns:
        η satisfying evaluate(η) ≈ 0, clamped to 0 if |η| < clamp_threshold.

    Raises:
        AutoArbConvergenceError: If no sign flip is reachable in either direction
            within max_expansions doublings of initial_step.
    """
    f0 = evaluate(0.0)
    if f0 == 0.0:
        return 0.0

    f0_sign = f0 > 0
    step = max(abs(initial_step), 1e-300)  # guard against zero step

    def _bisect(low: float, high: float, f_low: float, f_high: float) -> float:
        for _ in range(max_bisect):
            mid = (low + high) / 2
            if mid in (low, high):
                break
            f_mid = evaluate(mid)
            if f_mid == 0.0:
                low, high = mid, mid
                break
            if (f_mid > 0) == (f_low > 0):
                low, f_low = mid, f_mid
            else:
                high, f_high = mid, f_mid
        # Pick endpoint with smaller |f| for accuracy
        eta = low if abs(f_low) <= abs(f_high) else high
        return 0.0 if abs(eta) < clamp_threshold else eta

    # Try positive side
    pos = step
    pos_f = evaluate(pos)
    for _ in range(max_expansions):
        if (pos_f > 0) != f0_sign:
            return _bisect(0.0, pos, f0, pos_f)
        pos *= 2
        pos_f = evaluate(pos)
    if (pos_f > 0) != f0_sign:
        return _bisect(0.0, pos, f0, pos_f)

    # Try negative side
    neg = -step
    neg_f = evaluate(neg)
    for _ in range(max_expansions):
        if (neg_f > 0) != f0_sign:
            return _bisect(neg, 0.0, neg_f, f0)
        neg *= 2
        neg_f = evaluate(neg)
    if (neg_f > 0) != f0_sign:
        return _bisect(neg, 0.0, neg_f, f0)

    raise AutoArbConvergenceError(
        f"Could not bracket auto-arb root: evaluate(0) = {f0:.6e}; "
        f"after {max_expansions} doublings on each side starting from "
        f"|η| = {step:.6e}, no sign flip detected. "
        f"Last probes: evaluate({pos:.6e}) = {pos_f:.6e}, "
        f"evaluate({neg:.6e}) = {neg_f:.6e}. "
        f"This indicates a non-monotonic cost function (e.g., limit-order kink) "
        f"or pool exhaustion preventing convergence."
    )


def _max_legitimate_sigma_residual(
    ctx: LiquidityContext,
    answer_ids: list[str],
) -> float:
    """Upper bound on |Σp - 1| after a correctly-converged auto-arb.

    Theory (see docs/auto-arb-algorithms.tex and `MANIFOLD_ARB_EPSILON` rationale):
    when the binary search clamps |η| < MANIFOLD_ARB_EPSILON to zero (matching
    Manifold's `floatingArbitrageEqual` behavior), each answer's probability can
    drift by up to η·p(1-p)/T per answer, where T = pool YES + pool NO. The
    worst-case bound uses p(1-p) = 0.25 and sums across answers in this market.

    Used as the assert tolerance in calculate_*_shares_exact: a properly-converged
    search must finish within this bound. Anything larger is a real numerical
    failure (caller catches AutoArbConvergenceError).

    Args:
        ctx: Liquidity context (provides per-answer pool data).
        answer_ids: All answer IDs in the multi-choice market (the full set
            participating in the arb, not just the targets being purchased).

    Returns:
        Conservative bound, with a 4x safety multiplier to absorb second-order
        effects (e.g., probabilities moving during the primary purchase shifting
        the linearization point) and finite-precision noise from two clamp
        sites (binary search clamp and Manifold's own clamp at execution).
    """
    bound = 0.0
    for aid in answer_ids:
        pool = ctx.get_pool(aid)
        T = pool['YES'] + pool['NO']
        if T > 0:
            bound += MANIFOLD_ARB_EPSILON * 0.25 / T
    # Floor: numerical noise for cases where all answers have huge pools (bound -> 0)
    # 4x safety: see docstring.
    return max(4 * bound, 1e-9)


class PoolsContext:
    """Simple LiquidityContext adapter for raw pools dict (Layer 2).

    Implements LiquidityContext protocol using raw pool data.
    Does NOT support limit orders - uses pure AMM formulas.

    Used for:
    - Backwards compatibility with legacy callers
    - Tests that don't need a full MarketSimulator
    - Sequential calculations where you chain new_pools from results
    """

    def __init__(
        self,
        pools: dict[str, dict[str, float]],
        ps: dict[str, float] | None = None,
    ):
        self._pools = pools
        # Per-answer p weights (cpmm-multi-2). Absent -> 0.5, which reproduces
        # cpmm-multi-1's uniform answers exactly and keeps every existing caller
        # byte-identical (the lazy-migration default: p ?? 0.5).
        self._ps = ps if ps is not None else {aid: 0.5 for aid in pools}

    def _p(self, answer_id: str) -> float:
        """Per-answer p weight, defaulting to 0.5 when unspecified."""
        return self._ps.get(answer_id, 0.5)

    def get_answer_ids(self) -> list[str]:
        return list(self._pools.keys())

    def get_probability(self, answer_id: str) -> float:
        pool = self._pools[answer_id]
        return probability_from_pool(pool['YES'], pool['NO'], self._p(answer_id))

    def probability_for(self, answer_id: str, pool: dict[str, float]) -> float:
        return probability_from_pool(pool['YES'], pool['NO'], self._p(answer_id))

    def get_pool(self, answer_id: str) -> dict[str, float]:
        return self._pools[answer_id]

    def cost_for_shares(self, answer_id: str, shares: float, position: str) -> float:
        pool = self._pools[answer_id]
        return cost_for_shares(pool['YES'], pool['NO'], shares, position, p=self._p(answer_id))

    def shares_for_cost(self, answer_id: str, amount: float, position: str) -> float:
        from manifold.amm_core import shares_for_cost as amm_shares_for_cost
        pool = self._pools[answer_id]
        return amm_shares_for_cost(pool['YES'], pool['NO'], amount, position, p=self._p(answer_id))

    def pool_after_trade(self, answer_id: str, amount: float, position: str) -> dict[str, float]:
        pool = self._pools[answer_id]
        y_new, n_new = pool_after_trade(pool['YES'], pool['NO'], amount, position, p=self._p(answer_id))
        return {'YES': y_new, 'NO': n_new}

    def clone(self) -> PoolsContext:
        """Create an independent copy for hypothetical calculations.

        Returns:
            New PoolsContext with deep-copied pool + per-answer-p state
        """
        import copy
        return PoolsContext(copy.deepcopy(self._pools), copy.deepcopy(self._ps))

    def apply_trade(self, answer_id: str, cost: float, position: str) -> None:
        """Apply a trade to internal state. MUTATES SELF.

        Args:
            answer_id: The answer to trade in
            cost: Amount to spend (positive for buys)
            position: 'YES' or 'NO'
        """
        new_pool = self.pool_after_trade(answer_id, cost, position)
        self._pools[answer_id] = new_pool


@dataclass
class ArbitrageResult:
    """Result of a multi-choice arbitrage calculation."""
    shares_bought: float  # Total shares in target answer
    cost: float  # Amount spent by user
    fills: dict[str, float]  # answer_id -> shares bought (negative for NO)
    new_pools: dict[str, dict[str, float]]  # answer_id -> {YES: float, NO: float}
    new_probabilities: dict[str, float]  # answer_id -> probability


def calculate_yes_purchase_with_arbitrage(
    answer_id: str,
    amount: float,
    pools: dict[str, dict[str, float]],
    limit_prob: float | None = None
) -> ArbitrageResult:
    """Legacy wrapper for backwards compatibility.

    Creates a PoolsContext from the pools dict (no limit order support).
    For limit-aware calculations, use calculate_purchase_with_arbitrage with a MarketSimulator.
    """
    context = PoolsContext(pools)
    return calculate_purchase_with_arbitrage(context, answer_id, amount, "YES", limit_prob)


def calculate_purchase_with_arbitrage(
    context: LiquidityContext,
    answer_id: str,
    amount: float,
    position: str = "YES",
    limit_prob: float | None = None,
) -> ArbitrageResult:
    """Calculate the result of buying shares with auto-arbitrage (Dollar-centric).

    Internal: Not user-facing. Called by simulate_buy() and calculate_shares_for_amount().

    Algorithm: Dollar-centric auto-arbitrage (Algorithm 1 from auto_arb_algorithms.tex)
    ------------------------------------------------------------------------------------
    Given a fixed dollar amount C, find the number of shares S received.

    Redemption Strategy: Strategy 1 (arb in n-1 others, not target)
    - η NO in (n-1) others → η(n-2) cash + η YES in target

    Steps:
    1. Binary search for η (arb_shares) such that final Σp = 1.0
    2. For each η candidate:
       a. Calculate cost to buy η NO shares in each other answer (via context)
       b. Apply redemption: η NO in (n-1) others + η YES in target = η(n-2) cash
       c. Remaining budget goes to primary YES purchase in target
       d. Check if resulting Σp = 1.0

    Complexity: O(n log N) where n = number of answers, N = precision bits
    - ONE binary search for η (log N iterations)
    - Each iteration: O(n) calls to context.cost_for_shares (no nested searches)

    Manifold Reference:
    - calculateCpmmMultiArbitrageBetYes in calculate-cpmm-arbitrage.ts
    - Uses calculateAmountToBuySharesFixedP for inner cost calculations (O(1) formula)
    - Manifold also avoids nested searches via direct O(1) formulas

    Layer: 2 (Auto-Arbitrage)
    - Calls into Layer 3 (LiquidityContext) for cost_for_shares, pool_after_trade
    - Does NOT implement AMM math directly
    - Context handles limit order book when available (e.g., MarketSimulator)

    Limit Order Awareness: YES (via LiquidityContext protocol)
    - context.cost_for_shares() consumes limits in order
    - Caveat: Limit consumption is NOT reversible within binary search iterations
      (see auto_arb_algorithms.tex Section 6.1 for discussion)

    Mathematical Correctness: Correct modulo limit order reversibility caveat.

    Args:
        context: Liquidity context (e.g., MarketSimulator) for limit-aware calculations
        answer_id: Target answer to buy shares in
        amount: Amount to spend (the fixed input)
        position: 'YES' or 'NO' - which position to buy
        limit_prob: Optional limit probability (not currently used)

    Returns:
        ArbitrageResult with shares bought (variable output) and new market state
    """
    # Get answer IDs and validate
    all_answer_ids = context.get_answer_ids()
    n_answers = len(all_answer_ids)

    if n_answers < 2:
        raise ValueError("Multi-choice markets must have at least 2 answers")

    # FAIL FAST: Broken markets are out of scope
    initial_prob_sum = sum(context.get_probability(aid) for aid in all_answer_ids)
    if abs(initial_prob_sum - 1.0) > 0.05:  # Allow 5% deviation for rounding
        raise ValueError(
            f"Broken market detected: probabilities sum to {initial_prob_sum:.3f} "
            f"instead of 1.0. This violates AMM invariants. "
            f"Please report this market to Manifold."
        )

    # Handle very small amounts to avoid numerical issues
    if abs(amount) < MANIFOLD_ARB_EPSILON:
        # For tiny amounts, just return direct purchase without auto-arb
        shares = context.shares_for_cost(answer_id, amount, position)
        new_pool = context.pool_after_trade(answer_id, amount, position)

        # Build complete new_pools: target answer updated, others unchanged
        new_pools = {aid: context.get_pool(aid) for aid in all_answer_ids}
        new_pools[answer_id] = new_pool

        # Recalculate probabilities from final pools (per-answer p via context)
        new_probabilities = {
            aid: context.probability_for(aid, new_pools[aid])
            for aid in new_pools
        }

        return ArbitrageResult(
            shares_bought=shares,
            cost=amount,
            fills={answer_id: shares},
            new_pools=new_pools,
            new_probabilities=new_probabilities
        )

    # Determine arbitrage direction based on position
    arb_direction = "NO" if position == "YES" else "YES"

    # Calculate max arbitrage shares using context
    if arb_direction == "NO":
        arb_share_price_sum = sum(
            1 - context.get_probability(aid)  # NO price
            for aid in all_answer_ids
            if aid != answer_id
        )
    else:  # arb_direction == "YES"
        arb_share_price_sum = sum(
            context.get_probability(aid)  # YES price
            for aid in all_answer_ids
            if aid != answer_id
        )

    # Calculate denominator for max arb shares estimate
    if position == "YES":
        denominator = arb_share_price_sum - n_answers + 2
    else:  # position == "NO"
        denominator = arb_share_price_sum

    # Calculate max arbitrage shares
    max_arb_shares = 0.0 if abs(denominator) < 1e-10 else amount / denominator

    # Binary search for the right number of arbitrage shares
    def evaluate_arb_shares(arb_shares: float) -> float:
        """Returns the difference from 1.0 of the probability sum."""
        result = execute_arbitrage_trades(
            context, answer_id, amount, arb_shares, limit_prob, position
        )
        if result is None:
            return 1.0  # Too expensive, will search lower

        prob_sum = sum(result['probabilities'].values())
        return 1.0 - prob_sum

    # Binary search bounds
    low = 0.0
    high = max(0.0, max_arb_shares)
    best_arb_shares = 0.0

    for _ in range(50):
        mid = low + (high - low) / 2

        # Stop when we can't subdivide further
        if mid in (low, high):
            best_arb_shares = mid
            break

        diff = evaluate_arb_shares(mid)

        # Binary search logic depends on position
        if position == "YES":
            if diff > 0:
                high = mid  # Sum < 1, need fewer arb shares
            else:
                low = mid   # Sum > 1, need more arb shares
        else:  # position == "NO"
            if diff > 0:
                low = mid   # Sum < 1, need more arb shares
            else:
                high = mid  # Sum > 1, need fewer arb shares

        best_arb_shares = mid

    # Clamp near-zero arb shares to zero, matching Manifold's floatingArbitrageEqual
    # epsilon (vendor/.../calculate-cpmm-arbitrage.ts). When η < MANIFOLD_ARB_EPSILON,
    # Manifold skips auto-arb entirely. This leaves a small Σp residual:
    # ΔΣp ≤ η * p(1-p)/T per answer (see docs/auto-arb-algorithms.tex).
    if abs(best_arb_shares) < MANIFOLD_ARB_EPSILON:
        best_arb_shares = 0

    # Execute the final trade with the optimal arbitrage shares
    final_result = execute_arbitrage_trades(
        context, answer_id, amount, best_arb_shares, limit_prob, position
    )

    if final_result is None:
        raise ValueError(f"Failed to execute arbitrage trades: amount=${amount}, best_arb_shares={best_arb_shares}")

    # Validation: average price per share must be between $0 and $1
    target_shares_abs = abs(final_result['target_shares'])
    if target_shares_abs != 0:
        avg_price = amount / target_shares_abs
        if not (0 <= avg_price <= 1):
            raise ValueError(
                f"Invalid cost/shares ratio in final result: "
                f"{target_shares_abs:.6f} shares for ${amount:.2f} "
                f"gives avg price ${avg_price:.4f} (must be between $0 and $1)"
            )

    return ArbitrageResult(
        shares_bought=target_shares_abs,
        cost=amount,
        fills=final_result['fills'],
        new_pools=final_result['pools'],
        new_probabilities=final_result['probabilities']
    )


def execute_arbitrage_trades(
    context: LiquidityContext,
    answer_id: str,
    bet_amount: float,
    arb_shares: float,
    limit_prob: float | None,
    position: str,
) -> dict | None:
    """Execute auto-arbitrage and target purchase using context for limit-aware calculations.

    Layer: 2, User-facing: NO

    Args:
        context: Liquidity context (e.g., MarketSimulator) for limit-aware calculations
        answer_id: Target answer to buy shares in
        bet_amount: Total amount to spend
        arb_shares: Number of shares to buy for arbitrage in other answers
        limit_prob: Optional limit probability
        position: 'YES' or 'NO' - the position being taken in target answer

    Returns:
        Dictionary with target_shares, fills, pools, and probabilities
    """
    # Handle floating point precision issues
    # When binary search finds near-zero arbitrage shares, treat as zero (no auto-arb needed)
    # Match Manifold's floatingArbitrageEqual epsilon
    if abs(arb_shares) < MANIFOLD_ARB_EPSILON:
        arb_shares = 0

    # Determine arbitrage direction (opposite of target position)
    arb_direction = "NO" if position == "YES" else "YES"

    # Get all answer IDs from context
    all_answer_ids = context.get_answer_ids()
    n_answers = len(all_answer_ids)

    # Step 1: Calculate amounts needed to buy arbitrage shares in each other answer
    # Uses context for limit-aware cost calculation
    arb_amounts = {}
    total_arb_amount = 0.0

    for aid in all_answer_ids:
        if aid == answer_id:
            continue

        # Use context for limit-aware cost calculation
        amount_needed = context.cost_for_shares(aid, arb_shares, arb_direction)

        # Validation: price per share must be between $0 and $1
        if arb_shares != 0:
            avg_price = amount_needed / arb_shares
            if not (0 <= avg_price <= 1):
                raise ValueError(
                    f"cost_for_shares returned impossible price for {arb_direction}: "
                    f"{arb_shares:.2f} shares for ${amount_needed:.2f} "
                    f"(avg ${avg_price:.4f}/share, must be between $0 and $1) "
                    f"in answer {aid}"
                )

        arb_amounts[aid] = amount_needed
        total_arb_amount += amount_needed

    # Step 2: Apply redemption identity
    # For YES position: arb_shares NO in each of (n-1) others = arb_shares YES in target + arb_shares*(n-2) mana
    # For NO position: arb_shares YES in each of (n-1) others = arb_shares NO in target (no cash!)
    redeemed_amount = arb_shares * (n_answers - 2) if position == "YES" else 0
    net_arb_amount = total_arb_amount - redeemed_amount

    # Validate: In markets with Σp > 1 (YES) or Σp < 1 (NO), redemption should reduce net cost
    if net_arb_amount > total_arb_amount and arb_shares > 0.01:
        logger.warning(
            f"Redemption increased cost instead of reducing it! "
            f"total_arb_amount={total_arb_amount:.4f}, net={net_arb_amount:.4f}, "
            f"arb_shares={arb_shares:.4f}, n_answers={n_answers}"
        )

    # Step 3: Calculate primary purchase amount (in target answer)
    primary_bet_amount = bet_amount - net_arb_amount

    # Match Manifold's floatingArbitrageEqual behavior
    if abs(primary_bet_amount) < MANIFOLD_ARB_EPSILON:
        primary_bet_amount = 0.0

    # Step 4: Calculate final state using context (limit-aware, pure/no mutation)
    fills = {}
    new_pools = {}
    new_probabilities = {}

    # Calculate arb trades on other answers
    for aid in all_answer_ids:
        if aid == answer_id:
            continue

        if arb_shares != 0:
            # Use context for limit-aware pool calculation
            new_pool = context.pool_after_trade(aid, arb_amounts[aid], arb_direction)
            new_pools[aid] = new_pool

            if arb_direction == "NO":
                fills[aid] = -arb_shares  # Negative for NO
            else:
                fills[aid] = arb_shares  # Positive for YES
        else:
            fills[aid] = 0
            new_pools[aid] = context.get_pool(aid)

    # Calculate primary trade in target answer (limit-aware)
    if primary_bet_amount > 0:
        # Use context for limit-aware pool calculation
        target_new_pool = context.pool_after_trade(answer_id, primary_bet_amount, position)
        new_pools[answer_id] = target_new_pool

        # Calculate shares from cost using context
        primary_shares = context.shares_for_cost(answer_id, primary_bet_amount, position)
    else:
        new_pools[answer_id] = context.get_pool(answer_id)
        primary_shares = 0.0

    # Total shares in target = bought + redeemed from arbitrage
    total_target_shares = primary_shares + arb_shares

    # Set fills based on position
    if position == "YES":
        fills[answer_id] = total_target_shares  # Positive for YES
    else:  # position == "NO"
        fills[answer_id] = -total_target_shares  # Negative for NO

    # Recalculate ALL probabilities from final pools (per-answer p via context)
    for aid, pool in new_pools.items():
        new_probabilities[aid] = context.probability_for(aid, pool)

    return {
        'target_shares': total_target_shares,
        'fills': fills,
        'pools': new_pools,
        'probabilities': new_probabilities
    }


def calculate_shares_exact(
    context: LiquidityContext,
    answer_id: str,
    shares: float,
    position: str = "YES",
) -> ArbitrageResult:
    """Cost to buy exactly `shares` in target answer (share-centric query, Algo-1-faithful).

    Internal: not user-facing. Used by `calculate_buy_cost`, sales, and per-leg
    cost queries in the optimizer.

    Implementation: binary-search inversion of `calculate_purchase_with_arbitrage`
    (Algorithm 1, dollar-centric — what Manifold's API runs). Given target shares S,
    find cost C such that `simulate_buy(C).shares == S`. The (C, S, p) triple this
    returns is therefore the SAME equilibrium Manifold's API would settle at for a
    real $C trade — bug-for-bug match.

    --- Why we can't use Algorithm 2 (share-centric) standalone with limits ---

    The paper's Theorem EQ ("final state is mechanism-independent") relies on
    path-independence of the CPMM cost curve. Limit orders break path-independence:
    Algorithm 1 and Algorithm 2 reach DIFFERENT valid equilibria with limits, both
    satisfying Σp = 1 but with different per-answer probabilities and ~0.1-0.2%
    different costs. Manifold's API runs Algorithm 1 — Algorithm 2's standalone
    answer would mis-predict what real trades settle at.

    See `docs/auto-arb-algorithms.tex` §5.1 (the reversibility caveat — limits
    violate `C(δ) + C(-δ) = 0` so Theorem EQ doesn't extend), and
    `tasks/strategy_inverse_with_limits_2026_05_06/repro_synthetic.py` for the
    decisive 3-answer-1-limit measurement showing the split equilibria.

    --- Complexity trade-off ---

    O(n log² N): outer cost-bisect (log N) * inner Algorithm 1 (n log N). The
    paper's whole motivation was eliminating this nested search via Algorithm 2's
    O(n log N). We accept the regression on the limit-aware path because
    correctness > speed — the previous "fast Algorithm 2" produced costs that
    didn't match what Manifold actually charges. A fast-path that detects "no
    limits crossed in any leg" and falls back to Algorithm 2 is a reasonable
    follow-on optimization, deferred to keep this change scoped.

    Args:
        context: Liquidity context (e.g., MarketSimulator).
        answer_id: Target answer to buy shares in.
        shares: Exact number of shares to buy (positive).
        position: "YES" or "NO".

    Returns:
        ArbitrageResult: `shares_bought == shares` exactly (by contract);
        `cost`, `new_pools`, `new_probabilities` correspond to the equilibrium
        Algorithm 1 reaches at that cost.
    """
    position = position.upper()
    if position not in ("YES", "NO"):
        raise ValueError(f"Position must be 'YES' or 'NO', got '{position}'")

    answer_ids = context.get_answer_ids()
    if len(answer_ids) < 2:
        raise ValueError("Multi-choice markets must have at least 2 answers")

    if shares <= 0:
        raise ValueError(f"Shares must be positive, got {shares}")

    return _invert_dollar_centric(context, answer_id, shares, position)


def _invert_dollar_centric(
    context: LiquidityContext,
    answer_id: str,
    target_shares: float,
    position: str,
) -> ArbitrageResult:
    """Bisect over cost to find C such that Algorithm 1 yields `target_shares`.

    Numerical contract: returned `shares_bought == target_shares` exactly. Pools
    and probabilities come from Algorithm 1's output at the bisected cost; with
    60-80 bisection iterations on the smooth-with-kinks shares(cost) curve, the
    cost converges to within ULPs and the returned pools represent the same
    equilibrium Manifold's API would settle at.

    The shares(cost) curve produced by Algorithm 1 is monotonically non-decreasing
    in cost (more money buys more shares) and piecewise smooth (limits introduce
    kinks at limit-crossing prices). Bisection on a monotonic function converges
    at one bit per iteration regardless of kinks, so 80 iters → ~1e-24 relative
    cost precision.
    """
    # Initial bracket: cost ∈ [0, target_shares]. C=0 → 0 shares. C=target_shares is
    # an upper bound for any single-pool buy (avg price ≤ $1/share); also holds for
    # multi-pool auto-arb because the redemption identity (Strategy 1: η YES in
    # n-1 others ≡ η NO in target with no cash flow) means total cash spent equals
    # primary + arb costs in same-side-of-the-book quantities, all bounded by ≤ $1
    # per share. Expand defensively if pathological cases arise.
    #
    # Important: don't use a $1 floor here. In extreme-skew markets (e.g. p≈0.95
    # answer in a 2-answer market), probing Algorithm 1 with cost ≫ target_shares
    # can land outside its validated avg-price ∈ [0, 1] regime and raise. Sized
    # to target_shares the bracket stays in valid territory.
    lo = 0.0
    hi = max(float(target_shares), 1e-12)

    result_hi: ArbitrageResult | None = None
    bracket_found = False
    for _ in range(60):
        result_hi = calculate_purchase_with_arbitrage(
            context, answer_id, hi, position
        )
        if result_hi.shares_bought >= target_shares:
            bracket_found = True
            break
        hi *= 2

    if not bracket_found:
        assert result_hi is not None  # set in every loop iteration
        raise AutoArbConvergenceError(
            f"_invert_dollar_centric: could not bracket target_shares={target_shares} "
            f"within 60 doublings of initial $1 step. Last hi=${hi:.4f}, "
            f"shares_bought={result_hi.shares_bought:.6f}. "
            f"Answer {answer_id}, position {position}. "
            f"This indicates pool exhaustion or a non-monotonic shares(cost) curve."
        )

    # Bisect — converges to FP precision in ≤80 iterations on a monotonic function.
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if mid in (lo, hi):
            break  # no more subdivisible at FP precision
        result = calculate_purchase_with_arbitrage(context, answer_id, mid, position)
        if result.shares_bought < target_shares:
            lo = mid
        else:
            hi = mid

    # Return the result of one final Algorithm 1 call at the bisected cost. Override
    # `shares_bought` to `target_shares` exactly — bisection-noise bits in the share
    # count are a contract-level lie this layer should hide from callers.
    final_cost = (lo + hi) / 2.0
    final_result = calculate_purchase_with_arbitrage(
        context, answer_id, final_cost, position
    )
    fills = dict(final_result.fills)
    fills[answer_id] = target_shares if position == "YES" else -target_shares
    return ArbitrageResult(
        shares_bought=target_shares,
        cost=final_cost,
        fills=fills,
        new_pools=final_result.new_pools,
        new_probabilities=final_result.new_probabilities,
    )


def calculate_yes_shares_exact(
    context: LiquidityContext,
    answer_id: str,
    shares: float,
) -> ArbitrageResult:
    """Calculate cost to buy exactly `shares` YES shares. Wrapper for calculate_shares_exact."""
    return calculate_shares_exact(context, answer_id, shares, "YES")


def calculate_no_shares_exact(
    context: LiquidityContext,
    answer_id: str,
    shares: float,
) -> ArbitrageResult:
    """Calculate cost to buy exactly `shares` NO shares. Wrapper for calculate_shares_exact."""
    return calculate_shares_exact(context, answer_id, shares, "NO")


@dataclass
class MultiSharesResult:
    """Result of buying equal shares in multiple answers with auto-arb."""
    total_cost: float
    shares_per_outcome: float
    per_answer: dict[str, ArbitrageResult]  # answer_id -> individual result
    new_pools: dict[str, dict[str, float]] | None = None  # answer_id -> {YES, NO} final pools
    new_probabilities: dict[str, float] | None = None  # answer_id -> final probability
    price_ranges: dict[str, dict[str, float]] | None = None  # answer_id -> {initial, after_yes_purchase, final, min, max}


def calculate_multi_shares_exact(
    context: LiquidityContext,
    answer_ids: list[str],
    shares_per_outcome: float,
    position: str = "YES",
    *,
    track_price_ranges: bool = False,
) -> MultiSharesResult:
    """Calculate cost to buy exactly `shares_per_outcome` in each of multiple answers.

    Layer: 2, User-facing: NO

    This is the multi-answer variant of calculate_shares_exact(). It buys shares
    in each answer, then binary-searches for η across ALL answers to restore Σp=1.0.

    Algorithm:
    1. Clone context
    2. For each answer: buy shares_per_outcome via primary purchase (clone + apply)
    3. Binary search for η in ALL answers (including targets) to achieve Σp = 1.0
    4. Sum costs: Σ(primary_costs) + Σ(arb_costs) - redemption

    Complexity: O(k*n * log N) where k = len(answer_ids), n = total answers

    --- Caveat: limit-aware divergence from Manifold's API ---

    For YES position with multi-bet, Manifold's API runs vendor's iterative
    `calculateCpmmMultiArbitrageBetsYes`. With limits, the share-centric
    algorithm here reaches a DIFFERENT valid equilibrium (Σp = 1, exact shares,
    different per-answer probs and ~0.1-0.2% different total cost). The
    single-answer `calculate_shares_exact` was migrated to invert Algorithm 1
    for bug-for-bug Manifold match (see `tasks/strategy_inverse_with_limits_2026_05_06/`),
    but the multi-answer migration is deferred — see the same task directory.

    Args:
        context: Liquidity context with limit order support
        answer_ids: Target answer IDs to buy shares in
        shares_per_outcome: Exact shares to buy per answer
        position: "YES" or "NO"
        track_price_ranges: If True, include price_ranges in result with
            per-answer {initial, after_yes_purchase, final, min, max} probabilities

    Returns:
        MultiSharesResult with total cost, per-answer details, new_pools, and
        new_probabilities. If track_price_ranges=True, also includes price_ranges.
    """
    position = position.upper()
    if position not in ("YES", "NO"):
        raise ValueError(f"Position must be 'YES' or 'NO', got '{position}'")

    all_answer_ids = context.get_answer_ids()
    n_answers = len(all_answer_ids)

    if n_answers < 2:
        raise ValueError("Multi-choice markets must have at least 2 answers")

    if shares_per_outcome <= 0:
        raise ValueError(f"shares_per_outcome must be positive, got {shares_per_outcome}")

    # Single-target delegation: with one target answer, this reduces to the
    # single-answer case (no across-target redemption, no atomic-group semantics).
    # `calculate_shares_exact` inverts Algorithm 1 and is bug-for-bug
    # vendor-faithful (post `tasks/strategy_inverse_with_limits_2026_05_06/`).
    # Going through the share-centric `_solve_eta_for_zero` path below would
    # reach the equivalent A2 equilibrium, which diverges from Algorithm 1 by
    # ~0.04% with limits in the trajectory — the same split the single-answer
    # fix exists to avoid.
    if len(answer_ids) == 1:
        target_aid = answer_ids[0]
        single_result = calculate_shares_exact(
            context, target_aid, shares_per_outcome, position
        )
        per_answer = {target_aid: single_result}
        price_ranges = None
        if track_price_ranges:
            price_ranges = {}
            for aid in all_answer_ids:
                p_init = context.get_probability(aid)
                p_final = single_result.new_probabilities[aid]
                # No "after_yes_purchase" intermediate state in the single-answer
                # path — A1 is incremental, not a phase-1 / phase-2 split. Use
                # final to keep the field type-stable for downstream consumers
                # (limit-crossing detection compares against {min, max}).
                p_after = p_final
                price_ranges[aid] = {
                    'initial': p_init,
                    'after_yes_purchase': p_after,
                    'final': p_final,
                    'min': min(p_init, p_after, p_final),
                    'max': max(p_init, p_after, p_final),
                }
        return MultiSharesResult(
            total_cost=single_result.cost,
            shares_per_outcome=shares_per_outcome,
            per_answer=per_answer,
            new_pools=single_result.new_pools,
            new_probabilities=single_result.new_probabilities,
            price_ranges=price_ranges,
        )

    arb_direction = "NO" if position == "YES" else "YES"

    # Snapshot initial probabilities (breakpoint 1)
    initial_probs = {aid: context.get_probability(aid) for aid in all_answer_ids}

    # Step 1: Clone and apply primary purchases for all target answers
    ctx_after_primary = context.clone()
    primary_costs: dict[str, float] = {}

    for aid in answer_ids:
        cost = ctx_after_primary.cost_for_shares(aid, shares_per_outcome, position)
        ctx_after_primary.apply_trade(aid, cost, position)
        primary_costs[aid] = cost

    total_primary_cost = sum(primary_costs.values())

    # Snapshot after-primary probabilities (breakpoint 2)
    after_primary_probs = {aid: ctx_after_primary.get_probability(aid) for aid in all_answer_ids}

    # Step 2: Binary search for η to restore Σp = 1.0
    prob_sum_after = sum(after_primary_probs.values())
    logger.debug(
        f"After buying {shares_per_outcome} {position} in {len(answer_ids)} answers: "
        f"Σp = {prob_sum_after:.10f}"
    )

    # Early return if Σp is already 1.0
    if abs(prob_sum_after - 1.0) < 1e-8:
        logger.debug("Multi-shares: Σp already 1.0 after primary purchases, no arb needed")
        final_pools = {a: ctx_after_primary.get_pool(a) for a in all_answer_ids}
        final_probs = {a: ctx_after_primary.get_probability(a) for a in all_answer_ids}
        per_answer = {}
        for aid in answer_ids:
            fills = {aid: shares_per_outcome if position == "YES" else -shares_per_outcome}
            per_answer[aid] = ArbitrageResult(
                shares_bought=shares_per_outcome,
                cost=primary_costs[aid],
                fills=fills,
                new_pools=final_pools,
                new_probabilities=final_probs,
            )
        price_ranges = None
        if track_price_ranges:
            price_ranges = {}
            for aid in all_answer_ids:
                p_init = initial_probs[aid]
                p_final = final_probs[aid]
                price_ranges[aid] = {
                    'initial': p_init,
                    'after_yes_purchase': after_primary_probs[aid],
                    'final': p_final,
                    'min': min(p_init, after_primary_probs[aid], p_final),
                    'max': max(p_init, after_primary_probs[aid], p_final),
                }
        return MultiSharesResult(
            total_cost=total_primary_cost,
            shares_per_outcome=shares_per_outcome,
            per_answer=per_answer,
            new_pools=final_pools,
            new_probabilities=final_probs,
            price_ranges=price_ranges,
        )

    def evaluate_eta(eta: float) -> float:
        """Returns Σp - 1.0 after applying η arb trades across all answers.

        Same monotonicity story as calculate_shares_exact's evaluate_eta:
        increasing in η for position="NO", decreasing for position="YES". The
        helper handles both directions without baking in the assumption.
        """
        test_ctx = ctx_after_primary.clone()
        _apply_arb_trades(test_ctx, all_answer_ids, eta, arb_direction)
        return sum(test_ctx.get_probability(aid) for aid in all_answer_ids) - 1.0

    # `shares_per_outcome * len(answer_ids) * 2` is a hint, not a hard bound.
    initial_step = shares_per_outcome * len(answer_ids) * 2
    best_eta = _solve_eta_for_zero(evaluate_eta, initial_step=initial_step)
    logger.debug(f"Multi-shares binary search found η = {best_eta:.10f}")

    # Step 3: Apply best_eta and calculate costs
    final_ctx = ctx_after_primary.clone()
    total_arb_cost = _apply_arb_trades(final_ctx, all_answer_ids, best_eta, arb_direction)

    # Redemption
    if best_eta > 0:
        redemption_value = best_eta * (n_answers - 1) if position == "YES" else best_eta
    else:
        redemption_value = 0

    net_cost = total_primary_cost + total_arb_cost - redemption_value

    # Build final state (compute once, share across per-answer results)
    final_pools = {a: final_ctx.get_pool(a) for a in all_answer_ids}
    final_probs = {a: final_ctx.get_probability(a) for a in all_answer_ids}

    # Build per-answer results
    per_answer = {}
    for aid in answer_ids:
        fills = {aid: shares_per_outcome if position == "YES" else -shares_per_outcome}

        per_answer[aid] = ArbitrageResult(
            shares_bought=shares_per_outcome,
            cost=primary_costs[aid],
            fills=fills,
            new_pools=final_pools,
            new_probabilities=final_probs,
        )

    # Σp invariant: a properly-converged auto-arb must leave Σp = 1.0 within the
    # legitimate residual bound. See calculate_shares_exact for full rationale.
    final_prob_sum = sum(final_probs.values())
    sigma_bound = _max_legitimate_sigma_residual(ctx_after_primary, all_answer_ids)
    if abs(final_prob_sum - 1.0) > sigma_bound:
        raise AutoArbConvergenceError(
            f"calculate_multi_shares_exact: Σp = {final_prob_sum:.10f} after auto-arb "
            f"(residual {final_prob_sum - 1.0:+.3e}, bound {sigma_bound:.3e}). "
            f"Inputs: {answer_ids} {position} {shares_per_outcome} shares each; "
            f"η = {best_eta:.6e}; per-answer probs: {final_probs}"
        )

    # Build price_ranges if requested
    price_ranges = None
    if track_price_ranges:
        price_ranges = {}
        for aid in all_answer_ids:
            p_init = initial_probs[aid]
            p_after = after_primary_probs[aid]
            p_final = final_probs[aid]
            price_ranges[aid] = {
                'initial': p_init,
                'after_yes_purchase': p_after,
                'final': p_final,
                'min': min(p_init, p_after, p_final),
                'max': max(p_init, p_after, p_final),
            }

    return MultiSharesResult(
        total_cost=net_cost,
        shares_per_outcome=shares_per_outcome,
        per_answer=per_answer,
        new_pools=final_pools,
        new_probabilities=final_probs,
        price_ranges=price_ranges,
    )


# Manifold's multi-bet loop stops when the recycled redemption drops below this many
# mana (vendor: `while (amountToBet > 0.01)`). Replicated for bug-for-bug share counts.
_VENDOR_MULTIBET_STOP_MANA = 0.01


def calculate_multi_shares_for_amount(
    context: LiquidityContext,
    answer_ids: list[str],
    amount: float,
    position: str = "YES",
) -> MultiSharesResult:
    """Vendor-faithful AMOUNT → equal-shares multibet (inverse of `calculate_multi_shares_exact`).

    Ports Manifold's `calculateCpmmMultiArbitrageBetsYes`
    (`vendor/manifold/common/src/calculate-cpmm-arbitrage.ts:203-253`), which is a TRUNCATED
    iterative loop, NOT the converged Option-2 solve. Each round:
      1. spends `amount_to_bet` on equal YES (primary) shares across the target answers,
      2. auto-arb buys NO in all answers to restore Σp=1 and redeems complete sets,
      3. recycles the redemption proceeds (`extraMana = redemption - arb_cost`) as the next
         round's budget, looping `while (amount_to_bet > 0.01)`.
    It stops when the leftover redemption is below 0.01 mana, leaving that sliver unconverted,
    so it lands a hair short of the exact equilibrium `calculate_multi_shares_exact` computes
    (which is why a forecast built on the latter over-predicts the fill and trips the per-leg
    safety check). This reproduces what the multi-bet API actually delivers. See
    `tasks/multibet_sum_to_one_shortfall/onboarding.md`.

    Scope: pure AMM (no limit orders / fees) — covers the patterns that hit this in loop mode;
    the limit-aware extension is the deferred Strategy-1↔2 work.

    Returns a `MultiSharesResult` whose `shares_per_outcome` is the (equal) shares Manifold
    delivers per target leg for `amount`, and `total_cost` is the net mana spent (≈ `amount`).
    """
    if not answer_ids:
        raise ValueError("answer_ids must be non-empty")
    if amount <= 0:
        raise ValueError(f"amount must be positive, got {amount}")

    all_answer_ids = context.get_answer_ids()
    n_answers = len(all_answer_ids)
    arb_direction = "NO" if position == "YES" else "YES"

    ctx = context.clone()
    total_shares = 0.0
    total_primary = 0.0
    total_arb = 0.0
    total_redemption = 0.0
    amount_to_bet = amount
    # Geometric convergence (each round's redemption is a fraction of the last); the
    # `> 0.01` stop normally ends it in a handful of rounds. Guard against pathology.
    for _round in range(200):
        if amount_to_bet <= _VENDOR_MULTIBET_STOP_MANA:
            break

        # 1. Find equal shares `s` whose primary cost across the targets == amount_to_bet.
        #    cost_for_shares is monotone increasing in s; slippage makes the cost of
        #    `amount_to_bet / Σprob` shares exceed amount_to_bet, so that is a valid upper
        #    bound (matches the vendor's `maxYesShares = amountToBet / yesSharePriceSum`).
        def _primary_cost(s, _ctx=ctx, _aids=answer_ids, _pos=position):
            return sum(_ctx.cost_for_shares(aid, s, _pos) for aid in _aids)

        price_sum = sum(ctx.get_probability(aid) for aid in answer_ids)
        hi = amount_to_bet / price_sum if price_sum > 0 else amount_to_bet
        lo = 0.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if _primary_cost(mid) < amount_to_bet:
                lo = mid
            else:
                hi = mid
        s = 0.5 * (lo + hi)

        # 2. Apply the primary purchase (s YES in each target answer).
        round_primary = 0.0
        for aid in answer_ids:
            c = ctx.cost_for_shares(aid, s, position)
            ctx.apply_trade(aid, c, position)
            round_primary += c

        # 3. Restore Σp = 1 via the auto-arb (η trades in arb_direction across all answers).
        prob_sum = sum(ctx.get_probability(aid) for aid in all_answer_ids)
        if abs(prob_sum - 1.0) < 1e-9:
            eta = 0.0
            round_arb = 0.0
        else:
            def _evaluate_eta(eta, _ctx=ctx, _aids=all_answer_ids, _dir=arb_direction):
                test_ctx = _ctx.clone()
                _apply_arb_trades(test_ctx, _aids, eta, _dir)
                return sum(test_ctx.get_probability(aid) for aid in _aids) - 1.0

            eta = _solve_eta_for_zero(
                _evaluate_eta, initial_step=s * len(answer_ids) * 2
            )
            round_arb = _apply_arb_trades(ctx, all_answer_ids, eta, arb_direction)

        # 4. Redemption + recycled extraMana (the vendor's `noBuyResults.extraMana`).
        round_redemption = eta * (n_answers - 1) if position == "YES" else eta
        total_shares += s
        total_primary += round_primary
        total_arb += round_arb
        total_redemption += round_redemption
        amount_to_bet = round_redemption - round_arb

    final_pools = {aid: ctx.get_pool(aid) for aid in all_answer_ids}
    final_probs = {aid: ctx.get_probability(aid) for aid in all_answer_ids}
    net_cost = total_primary + total_arb - total_redemption
    per_answer = {
        aid: ArbitrageResult(
            shares_bought=total_shares,
            cost=net_cost / len(answer_ids),
            fills={aid: total_shares if position == "YES" else -total_shares},
            new_pools=final_pools,
            new_probabilities=final_probs,
        )
        for aid in answer_ids
    }
    return MultiSharesResult(
        total_cost=net_cost,
        shares_per_outcome=total_shares,
        per_answer=per_answer,
        new_pools=final_pools,
        new_probabilities=final_probs,
        price_ranges=None,
    )
