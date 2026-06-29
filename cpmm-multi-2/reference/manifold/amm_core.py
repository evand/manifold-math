"""Core AMM (Automated Market Maker) formulas for Manifold Markets.

Layer 4: Raw AMM math. Pure functions, no state, no limit awareness.
Called by: Layer 3 only. NEVER call from Layer 2 or Layer 1.

This module provides the fundamental CPMM calculations as pure functions.
All other AMM code should use these primitives rather than reimplementing
the formulas.

The key insight: for p=0.5 markets (all multi-choice, many binary), there are
two fundamental operations:

1. SHARES → COST: Given a number of shares to buy, what does it cost?
2. COST → SHARES: Given an amount to spend, how many shares do we get?

Both directions have closed-form solutions for p=0.5.

For general p values, we also provide the formulas but they're more complex.

Mathematical basis:
- Invariant: k = Y^p * N^(1-p) must be preserved
- For p=0.5: k = sqrt(Y * N)
- Probability: P(YES) = (p * N) / ((1-p) * Y + p * N)
- For p=0.5: P(YES) = N / (Y + N)

See docs/amm-invariants.md for proofs.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from manifold.constants import BUDGET_EPSILON, PRICE_TOLERANCE

if TYPE_CHECKING:
    from manifold.order_book import LimitOrder, OrderBook


def cost_for_shares(
    y: float,
    n: float,
    shares: float,
    position: str,
    order_book: OrderBook | None = None,
    p: float = 0.5,
) -> float:
    """Calculate the cost to buy a given number of shares.

    Layer: 4, User-facing: NO

    This is the SHARES → COST direction. Given pool state and desired shares,
    returns the mana cost.

    Args:
        y: YES shares in pool
        n: NO shares in pool
        shares: Number of shares to buy. Positive = buy, negative = sell.
        position: "YES" or "NO" - which position to buy/sell
        order_book: Optional OrderBook with limit orders. When provided,
                   uses piecewise calculation (AMM + limit fills).
        p: Probability parameter (default 0.5)

    Returns:
        Cost in mana. Positive = user pays, negative = user receives.
        Returns 0 if shares is 0.

    Note:
        For p=0.5, there's a closed-form solution.
        For general p, uses binary search (inverts shares_for_cost).
    """
    # Edge case: 0 shares costs nothing
    if shares == 0:
        return 0.0

    position = position.upper()
    if position not in ("YES", "NO"):
        raise ValueError(f"position must be 'YES' or 'NO', got {position}")

    # If order_book provided, use piecewise calculation (AMM + limits)
    if order_book is not None:
        cost, _, _, _ = buy_with_limits(
            y, n, position, order_book,
            target_shares=shares,
            p=p,
        )
        return cost

    # For p=0.5, use closed-form formula
    if p == 0.5:
        # The formula differs only in which pool variable is used
        # YES: use N (the other pool)
        # NO: use Y (the other pool)
        other_pool = n if position == "YES" else y

        discriminant = (y + n - shares) ** 2 + 4 * shares * other_pool
        if discriminant < 0:
            raise ValueError(
                f"Cannot buy {shares} {position} shares - insufficient liquidity. "
                f"Pool: YES={y:.2f}, NO={n:.2f}"
            )

        cost = (shares - y - n + math.sqrt(discriminant)) / 2
        return cost

    # For general p, invert shares_for_cost by bisection. This is the pure-CPMM
    # cost (convention A, GP4) and works for BOTH signs now that shares_for_cost /
    # pool_after_trade handle a reverse correctly. Bracket is sign-aware:
    #   buy  (shares > 0): cost in [0, shares*10]   (each share costs < $1)
    #   sell (shares < 0): cost in (-other_pool, 0] (a reverse can't drain the
    #                      traded pool side; cost is the negative proceeds)
    # shares_for_cost is monotone increasing in cost in both regimes.
    if shares > 0:
        low, high = 0.0, shares * 10  # Upper bound estimate
    else:
        other_pool = n if position == "YES" else y
        low, high = -other_pool * (1 - 1e-9), 0.0
    for _ in range(50):  # Sufficient iterations for double precision
        mid = (low + high) / 2

        # Break once we've reached max precision
        if mid in (low, high):
            break

        shares_for_mid = shares_for_cost(y, n, mid, position, p=p)
        if shares_for_mid < shares:
            low = mid
        else:
            high = mid

    return (low + high) / 2


def shares_for_cost(
    y: float,
    n: float,
    cost: float,
    position: str,
    order_book: OrderBook | None = None,
    p: float = 0.5,
) -> float:
    """Calculate shares received for a given cost.

    Layer: 4, User-facing: NO

    This is the COST → SHARES direction. Given pool state and amount to spend,
    returns the shares received.

    Args:
        y: YES shares in pool
        n: NO shares in pool
        cost: Amount of mana to spend. Positive = buying, negative = selling.
        position: "YES" or "NO" - which position to trade
        order_book: Optional OrderBook with limit orders. When provided,
                   uses piecewise calculation (AMM + limit fills).
        p: Probability parameter (default 0.5)

    Returns:
        Number of shares received (positive) or sold (negative)

    Raises:
        ValueError: If cost == 0 or position invalid

    Formula:
        When spending cost c to buy YES:
        - c creates c YES + c NO pairs
        - New pools: N' = N + c, Y' from k-invariant
        - k = Y^p * N^(1-p)
        - Y' = (k / N'^(1-p))^(1/p)
        - shares = c + (Y - Y')
    """
    if cost == 0:
        return 0.0

    position = position.upper()
    if position not in ("YES", "NO"):
        raise ValueError(f"position must be 'YES' or 'NO', got {position}")

    # If order_book provided AND buying (positive cost), use piecewise calculation.
    # Selling (cost < 0) is the pure-CPMM REVERSE (convention A, GP4) and falls
    # through to the pure-AMM formula below -- it doesn't walk the limit book (a
    # real economic sell is buy-opposite+redeem, Theorem 9, handled elsewhere).
    if order_book is not None and cost > 0:
        _, shares, _, _ = buy_with_limits(
            y, n, position, order_book,
            target_cost=cost,
            p=p,
        )
        return shares

    # Pure AMM calculation - use pool_after_trade for the math. With convention-A
    # pool_after_trade, `cost + (pool - pool_new)` is correct for BOTH signs: for a
    # sell, pool_new grows on the traded side so (pool - pool_new) < 0 and shares
    # come out negative (the old buy-only derivation collapsed to 2*cost -- fixed
    # by the pool_after_trade convention, not here).
    y_new, n_new = pool_after_trade(y, n, cost, position, p)

    # Shares = cost (from pair creation) + pool change
    shares = cost + (y - y_new) if position == "YES" else cost + (n - n_new)

    return shares


def pool_after_trade(
    y: float, n: float, cost: float, position: str, p: float = 0.5
) -> tuple[float, float]:
    """Calculate new pool state after a trade.

    Layer: 4, User-facing: NO

    Args:
        y: YES shares in pool before trade
        n: NO shares in pool before trade
        cost: Amount spent (positive = buying, negative = selling)
        position: "YES" or "NO" - which position was traded
        p: Probability parameter (default 0.5)

    Returns:
        Tuple of (y_new, n_new) - new pool state

    Note:
        Negative cost represents selling (or equivalently, buying the opposite).
        The k-invariant is preserved: y^p * n^(1-p) = constant
    """
    position = position.upper()
    if position not in ("YES", "NO"):
        raise ValueError(f"position must be 'YES' or 'NO', got {position}")

    # Convention A (GP4): a SELL is the pure-CPMM REVERSE of a buy, so it acts on
    # the SAME pool side as the buy regardless of sign -- a YES trade moves the NO
    # pool (n_new = n + cost), a NO trade moves the YES pool (y_new = y + cost).
    # cost < 0 simply shrinks that side. This makes buy-then-reverse return EXACTLY
    # to the start (proofs/general_p_cost.py GP4, line 118: N_back = N_buy - R).
    # The old sign-switched form (sell YES grew the YES pool) was NOT reversible --
    # a buy/sell round-trip drifted off the start, even at p=0.5.
    #
    # NB: this pure-CPMM reverse is DISTINCT from the economic sell Manifold runs
    # (buy-opposite + redeem complete sets, Theorem 9), which yields different cash
    # and pool when selling from an arbitrary state. Production selling goes through
    # the Theorem-9 path (positive args, opposite position) and never reaches this
    # negative-cost branch. See tasks/cpmm_multi_2/findings-sell-accounting.md.

    # For p=0.5, use simple formula to preserve numerical precision
    if p == 0.5:
        k_squared = y * n

        if position == "YES":
            n_new = n + cost
            if n_new <= 0:
                raise ValueError(
                    f"Cannot reverse {cost} YES: drains NO pool (N={n})"
                )
            y_new = k_squared / n_new
        else:  # NO
            y_new = y + cost
            if y_new <= 0:
                raise ValueError(
                    f"Cannot reverse {cost} NO: drains YES pool (Y={y})"
                )
            n_new = k_squared / y_new

        return (y_new, n_new)

    # For general p, use the power formula (same convention-A pool sides).
    k = (y ** p) * (n ** (1 - p))

    if position == "YES":
        n_new = n + cost
        if n_new <= 0:
            raise ValueError(
                f"Cannot reverse {cost} YES: drains NO pool (N={n})"
            )
        y_new = (k / (n_new ** (1 - p))) ** (1 / p)
    else:  # NO
        y_new = y + cost
        if y_new <= 0:
            raise ValueError(
                f"Cannot reverse {cost} NO: drains YES pool (Y={y})"
            )
        n_new = (k / (y_new ** p)) ** (1 / (1 - p))

    return (y_new, n_new)


def probability_from_pool(y: float, n: float, p: float = 0.5) -> float:
    """Calculate probability from pool state.

    Layer: 4, User-facing: NO

    Args:
        y: YES shares in pool
        n: NO shares in pool
        p: Probability parameter (default 0.5 for multi-choice)

    Returns:
        P(YES) probability between 0 and 1

    Formula:
        P(YES) = (p * N) / ((1-p) * Y + p * N)
        For p=0.5: P(YES) = N / (Y + N)
    """
    if y + n == 0:
        return 0.5

    if p == 0.5:
        return n / (y + n)
    else:
        return (p * n) / ((1 - p) * y + p * n)


def probability_after_trade(
    y: float, n: float, cost: float, position: str, p: float = 0.5
) -> float:
    """Calculate probability after a trade.

    Layer: 4, User-facing: NO

    Convenience function combining pool_after_trade and probability_from_pool.

    Args:
        y: YES shares in pool before trade
        n: NO shares in pool before trade
        cost: Amount spent
        position: "YES" or "NO"
        p: Probability parameter (default 0.5)

    Returns:
        P(YES) probability after the trade
    """
    y_new, n_new = pool_after_trade(y, n, cost, position, p)
    return probability_from_pool(y_new, n_new, p)


def cost_to_probability(
    y: float, n: float, target_prob: float, position: str, p: float = 0.5
) -> float:
    """Calculate cost to move market to target probability.

    Layer: 4, User-facing: NO

    Args:
        y: YES shares in pool
        n: NO shares in pool
        target_prob: Target P(YES) probability
        position: "YES" or "NO" - which position to buy
        p: Probability parameter (default 0.5)

    Returns:
        Cost in mana to reach target probability

    Raises:
        ValueError: If target is unreachable with given position

    Formula (for p=0.5, position=YES):
        Target: N' / (Y' + N') = t
        Constraint: Y' * N' = Y * N = k²
        Solving: N' = sqrt(k² * t / (1-t))
        Cost = N' - N
    """
    if target_prob <= 0 or target_prob >= 1:
        raise ValueError(f"target_prob must be in (0, 1), got {target_prob}")

    position = position.upper()
    current_prob = probability_from_pool(y, n, p)

    if position == "YES":
        if target_prob <= current_prob:
            raise ValueError(
                f"Cannot reach {target_prob:.4f} by buying YES from {current_prob:.4f}"
            )
    else:
        if target_prob >= current_prob:
            raise ValueError(
                f"Cannot reach {target_prob:.4f} by buying NO from {current_prob:.4f}"
            )

    if p == 0.5:
        k_squared = y * n
        t = target_prob

        # Final pool state from target probability
        # P(YES) = N' / (Y' + N') = t, and Y' * N' = k²
        # Solving: N' = sqrt(k² * t / (1-t)), Y' = sqrt(k² * (1-t) / t)
        n_final = math.sqrt(k_squared * t / (1 - t))
        y_final = math.sqrt(k_squared * (1 - t) / t)

        # Cost = change in appropriate pool (N for YES, Y for NO)
        return n_final - n if position == "YES" else y_final - y
    else:
        # For general p, use binary search
        low, high = 0.0, 10000.0
        for _ in range(50):
            mid = (low + high) / 2
            prob = probability_after_trade(y, n, mid, position, p)

            if position == "YES":
                if prob < target_prob:
                    low = mid
                else:
                    high = mid
            else:
                if prob > target_prob:
                    low = mid
                else:
                    high = mid

            if abs(prob - target_prob) < 1e-10:
                break

        return (low + high) / 2


def shares_to_probability(
    y: float, n: float, target_prob: float, position: str, p: float = 0.5
) -> float:
    """Calculate shares needed to move market to target probability.

    Layer: 4, User-facing: NO

    Args:
        y: YES shares in pool
        n: NO shares in pool
        target_prob: Target P(YES) probability
        position: "YES" or "NO" - which position to buy
        p: Probability parameter (default 0.5)

    Returns:
        Number of shares to buy to reach target probability
    """
    cost = cost_to_probability(y, n, target_prob, position, p)
    return shares_for_cost(y, n, cost, position)


# =============================================================================
# Limit-aware calculations
# =============================================================================


def _is_limit_behind(limit_price: float, current_prob: float, direction: str) -> bool:
    """Check if a limit is strictly behind current position.

    A limit is "behind" if we've already passed it - it should have been
    filled earlier and is no longer relevant.

    Args:
        limit_price: The limit order's price
        current_prob: Current pool probability
        direction: "UP" (buying YES) or "DOWN" (buying NO)

    Returns:
        True if limit is behind (should be skipped), False if at/ahead.

    Note:
        Uses PRICE_TOLERANCE to handle floating-point comparison between
        exact limit prices (from API) and computed pool probabilities.
        A limit within tolerance of current price is NOT behind.
    """
    if direction == "UP":
        # Moving up: limit is behind if it's below current (outside tolerance)
        return limit_price < current_prob - PRICE_TOLERANCE
    else:
        # Moving down: limit is behind if it's above current (outside tolerance)
        return limit_price > current_prob + PRICE_TOLERANCE


def _is_limit_at_price(limit_price: float, current_prob: float) -> bool:
    """Check if a limit is at the current price (within tolerance).

    Args:
        limit_price: The limit order's price
        current_prob: Current pool probability

    Returns:
        True if limit is at current price (should be filled now).
    """
    return abs(limit_price - current_prob) <= PRICE_TOLERANCE


def _skip_passed_limits(
    limits: list[LimitOrder],
    idx: int,
    current_prob: float,
    direction: str,
) -> int:
    """Advance index past limits we've already passed.

    This is the cursor advancement logic. After AMM trading, the pool
    probability may have moved past some limits - this skips them.

    Args:
        limits: Sorted list of limit orders
        idx: Current cursor position
        current_prob: Current pool probability
        direction: "UP" or "DOWN"

    Returns:
        New index pointing to first limit not behind current position.
    """
    while idx < len(limits) and _is_limit_behind(limits[idx].price, current_prob, direction):
        idx += 1
    return idx


def _skip_limits_at_or_behind(
    limits: list[LimitOrder],
    idx: int,
    current_prob: float,
    direction: str,
) -> int:
    """Advance index past limits at or behind current position.

    Used for INITIAL cursor positioning only. At the start of a trade,
    limits exactly at current price haven't been crossed yet.

    Args:
        limits: Sorted list of limit orders
        idx: Current cursor position
        current_prob: Current pool probability
        direction: "UP" or "DOWN"

    Returns:
        New index pointing to first limit strictly ahead of current position.
    """
    while idx < len(limits):
        limit_price = limits[idx].price
        # Skip if behind OR at current price
        if _is_limit_behind(limit_price, current_prob, direction) or _is_limit_at_price(
            limit_price, current_prob
        ):
            idx += 1
        else:
            break
    return idx


def _fill_single_limit(
    limit: LimitOrder,
    position: str,
    max_shares: float,
    max_cost: float,
    maker_balance: float | None = None,
) -> tuple[float, float, float]:
    """Fill a single limit order up to constraints.

    Args:
        limit: The limit order to fill
        position: "YES" or "NO" - what WE are buying
        max_shares: Maximum shares we can buy
        max_cost: Maximum cost we can spend
        maker_balance: Maker's available account balance in mana, or None to
            disable the matcher's `min(orderRemaining, makerBalance)` clamp
            (legacy behavior). Mirrors `matchedBetUserBalance` in
            `vendor/manifold/common/src/calculate-cpmm.ts:388-396`. A balance
            of `<= 0` means "treat as no balance"; the limit produces an
            empty fill and the caller should advance past it (matches
            vendor's `matchableUserBalance < 0 ? 0` clamp combined with the
            cancel-orders check).

    Returns:
        Tuple of (cost, shares, maker_amount) for this fill, where
        maker_amount is the mana the maker contributes — needed by callers
        that track per-maker balance decrements across sequential fills.

    Note:
        Cost per share depends on what WE buy:
        - Buying YES at price P costs P per share
        - Buying NO at price P costs (1-P) per share
        Maker's cost per share is the complement (sums to 1, CPMM
        conservation).
    """
    # Cost per share from OUR perspective
    cost_per_share = limit.price if position == "YES" else (1 - limit.price)
    # Cost per share from the MAKER's perspective (what their balance funds).
    # Sums to 1 with our cost: a YES+NO pair always costs $1 in CPMM.
    maker_cost_per_share = limit.price if limit.outcome == "YES" else (1 - limit.price)

    # Maker-balance clamp. None = no clamping (legacy / live API where balance
    # is unknown). Zero or negative = treat as no remaining balance, no fill.
    if maker_balance is None:
        max_by_maker_balance = float("inf")
    elif maker_balance <= 0:
        return (0.0, 0.0, 0.0)
    else:
        max_by_maker_balance = (
            maker_balance / maker_cost_per_share
            if maker_cost_per_share > 0
            else float("inf")
        )

    # How many shares can we fill?
    max_by_limit = limit.size
    max_by_shares = max_shares
    max_by_cost = max_cost / cost_per_share if cost_per_share > 0 else float("inf")

    shares_filled = min(max_by_limit, max_by_shares, max_by_cost, max_by_maker_balance)
    cost = shares_filled * cost_per_share
    maker_amount = shares_filled * maker_cost_per_share

    return cost, shares_filled, maker_amount


def buy_until(
    y: float,
    n: float,
    position: str,
    max_cost: float | None = None,
    max_shares: float | None = None,
    max_prob: float | None = None,
    p: float = 0.5,
) -> tuple[float, float, float, float, str]:
    """Buy from AMM until the first constraint is hit.

    This is the core primitive for limit-aware calculations. It buys from the AMM
    until one of the constraints (cost budget, shares target, or probability target)
    is reached.

    Args:
        y: YES shares in pool
        n: NO shares in pool
        position: "YES" or "NO" - which position to buy
        max_cost: Maximum cost to spend (optional)
        max_shares: Maximum shares to buy (optional)
        max_prob: Maximum probability to reach (optional, for YES buys this is
                  an upper bound, for NO buys this is a lower bound)
        p: Probability parameter (default 0.5)

    Returns:
        Tuple of (cost, shares, y_new, n_new, constraint_hit) where constraint_hit
        is one of "cost", "shares", "prob", or "none" (if no constraints given).

    Example:
        >>> y, n = 100.0, 100.0  # 50% prob
        >>> cost, shares, y_new, n_new, hit = buy_until(
        ...     y, n, "YES", max_prob=0.60
        ... )
        >>> hit
        'prob'
    """
    position = position.upper()
    if position not in ("YES", "NO"):
        raise ValueError(f"position must be 'YES' or 'NO', got {position}")

    # Handle zero-shares request immediately - can't buy if max_shares is 0
    if max_shares is not None and max_shares <= 0:
        return (0.0, 0.0, y, n, "shares")

    current_prob = probability_from_pool(y, n, p)

    # "Go up to" semantics: if we're already at/past max_prob, return 0 immediately.
    # This is crucial for the cursor-based limit order algorithm - it tells the caller
    # "nothing to do in AMM phase, proceed to fill the limit".
    if max_prob is not None:
        if position == "YES" and current_prob >= max_prob:
            return (0.0, 0.0, y, n, "prob")
        if position == "NO" and current_prob <= max_prob:
            return (0.0, 0.0, y, n, "prob")

    # Calculate the cost implied by each constraint
    # Only add finite constraints - inf means "no constraint"
    costs = {}

    if max_cost is not None and max_cost > 0 and not math.isinf(max_cost):
        costs["cost"] = max_cost

    if max_shares is not None and max_shares > 0 and not math.isinf(max_shares):
        costs["shares"] = cost_for_shares(y, n, max_shares, position, p=p)

    # max_prob is reachable (we checked above), so add it to costs
    if max_prob is not None:
        costs["prob"] = cost_to_probability(y, n, max_prob, position, p)

    if not costs:
        # No constraints - return zero trade
        return (0.0, 0.0, y, n, "none")

    # Find the binding constraint (minimum cost)
    constraint_hit = min(costs, key=costs.get)
    actual_cost = costs[constraint_hit]

    # Execute the trade
    if actual_cost <= 0:
        return (0.0, 0.0, y, n, constraint_hit)

    actual_shares = shares_for_cost(y, n, actual_cost, position, p=p)
    y_new, n_new = pool_after_trade(y, n, actual_cost, position, p)

    return (actual_cost, actual_shares, y_new, n_new, constraint_hit)


def buy_with_limits(
    y: float,
    n: float,
    position: str,
    order_book: OrderBook,
    target_cost: float | None = None,
    target_shares: float | None = None,
    target_prob: float | None = None,
    p: float = 0.5,
    consumption_report: dict[str, float] | None = None,
    maker_balances: dict[str, float] | None = None,
    maker_amount_report: dict[str, float] | None = None,
) -> tuple[float, float, float, float]:
    """Buy with limit order support, alternating between AMM and limit fills.

    Layer: 4, User-facing: NO

    This is the main entry point for limit-aware calculations. It buys from the AMM
    and fills limit orders as they are crossed, until the target is reached.

    Args:
        y: YES shares in pool
        n: NO shares in pool
        position: "YES" or "NO" - which position to buy
        order_book: OrderBook with limit orders
        target_cost: Target cost to spend (optional)
        target_shares: Target shares to buy (optional)
        target_prob: Target probability to reach (optional)
        p: Probability parameter (default 0.5)
        consumption_report: Optional dict to populate with limit consumption.
            When provided, records {limit.order_id: mana_consumed} for each
            limit order filled during this trade. Backward-compatible: existing
            callers pass nothing and see no change.
        maker_balances: Optional `{userId: balance_mana}`. When provided,
            each limit's fill is clamped to its maker's available balance
            (vendor's `min(orderRemaining, matchableUserBalance)` at
            calculate-cpmm.ts:393), and balances decrement *within this
            call* so multiple fills against the same maker chain correctly.
            The caller's dict is NOT mutated — a local copy is used. None
            (default) = no clamping (legacy behavior preserved).
        maker_amount_report: Optional dict to populate with per-maker mana
            actually consumed by this call. Used by the simulator's
            `apply_trade` to merge decrements into persistent maker-balance
            state. Mirrors the consumption_report pattern.

    Returns:
        Tuple of (total_cost, total_shares, y_new, n_new) where y_new and n_new
        are the pool state after AMM trades (limit fills don't affect the pool).

    Note:
        At least one target must be specified. If multiple targets are given,
        execution stops when ANY target is reached.

    Example:
        >>> from manifold.order_book import LimitOrder, OrderBook
        >>> y, n = 100.0, 100.0  # 50% prob
        >>> limit = LimitOrder(price=0.60, size=10.0, outcome="YES")
        >>> order_book = OrderBook(limits_above=[limit])
        >>> cost, shares, y_new, n_new = buy_with_limits(
        ...     y, n, "YES", order_book, target_prob=0.70
        ... )
    """
    position = position.upper()
    if position not in ("YES", "NO"):
        raise ValueError(f"position must be 'YES' or 'NO', got {position}")

    if target_cost is None and target_shares is None and target_prob is None:
        raise ValueError("At least one target must be specified")

    # Setup: direction and sorted limits
    direction = "UP" if position == "YES" else "DOWN"
    limits = order_book.limits_above if direction == "UP" else order_book.limits_below

    # Initialize cursor and budget tracking
    cursor = 0
    remaining_cost = target_cost if target_cost is not None else float("inf")
    remaining_shares = target_shares if target_shares is not None else float("inf")
    total_cost = 0.0
    total_shares = 0.0
    current_y, current_n = y, n

    # Local mutable copy of maker balances. Within ONE call we walk multiple
    # limits, possibly hitting the same maker more than once (different price
    # tiers, or stage-4 cross-leg). Each fill decrements the local copy so the
    # next fill sees the reduced balance, mirroring vendor's
    # `currentBalanceByUserId` at calculate-cpmm.ts:252,279-289. Caller's dict
    # is never mutated.
    local_balances = dict(maker_balances) if maker_balances is not None else None

    # Position cursor past any limits we've already passed
    # Note: Limits AT current price (within tolerance) are NOT skipped -
    # they should be filled (matches Manifold API behavior)
    current_prob = probability_from_pool(current_y, current_n, p)
    cursor = _skip_passed_limits(limits, cursor, current_prob, direction)

    # Main loop: alternate between AMM segments and limit fills
    #
    # Algorithm (cursor-as-epsilon approach from Stockfighter):
    #   1. AMM to next limit price (may return 0 if already there - that's fine!)
    #   2. If budget exhausted or target reached, stop
    #   3. Fill the limit at cursor, advance cursor
    #   4. Repeat
    #
    # Key insight: The integer cursor tells us which limits we've processed,
    # avoiding floating-point comparison issues. See patio11's Stockfighter article.
    #
    # Loop invariant: cursor points to first unprocessed limit
    while True:
        next_limit = limits[cursor] if cursor < len(limits) else None

        # Determine segment target: next limit, target_prob, or boundary
        # If target_prob is BEFORE next limit, go to target and stop (don't fill limit at target)
        if next_limit is not None:
            limit_is_past_target = (
                target_prob is not None
                and (
                    (direction == "UP" and target_prob <= next_limit.price)
                    or (direction == "DOWN" and target_prob >= next_limit.price)
                )
            )
            segment_prob = target_prob if limit_is_past_target else next_limit.price
        else:
            # No more limits - go to target (or no prob constraint if target not set)
            # When there's no target_prob, we rely purely on cost/shares constraints
            segment_prob = target_prob  # None is fine - buy_until handles it

        # Phase 1: AMM trade to segment target
        # buy_until returns 0 if already at/past max_prob (clean "go up to" semantics)
        cost, shares, current_y, current_n, hit = buy_until(
            current_y, current_n, position,
            max_cost=remaining_cost,
            max_shares=remaining_shares,
            max_prob=segment_prob,
            p=p,
        )

        total_cost += cost
        total_shares += shares
        remaining_cost -= cost
        remaining_shares -= shares

        # Check termination: budget exhausted?
        if remaining_cost <= BUDGET_EPSILON or remaining_shares <= BUDGET_EPSILON:
            break

        # Check termination: hit cost/shares constraint?
        if hit in ("cost", "shares"):
            break

        # Check termination: reached target_prob (which is before/at next limit)?
        # Note: We DON'T fill a limit exactly at target_prob (user doesn't need that liquidity)
        if next_limit is None or (target_prob is not None and segment_prob == target_prob):
            break

        # Phase 2: Fill the limit at cursor position
        # We reached limit price (hit == "prob") - fill ALL limits at this exact price
        limit_price = next_limit.price
        while cursor < len(limits) and limits[cursor].price == limit_price:
            cur_limit = limits[cursor]

            # Look up maker's *current* balance (post-decrement from prior
            # fills in this call). `None` means "not in dict" — vendor treats
            # that as no clamping (matchableUserBalance ?? amountRemaining).
            uid = cur_limit.user_id
            maker_balance = (
                local_balances.get(uid)
                if local_balances is not None and uid is not None
                else None
            )

            fill_cost, fill_shares, fill_maker_amount = _fill_single_limit(
                cur_limit, position, remaining_shares, remaining_cost,
                maker_balance=maker_balance,
            )

            # Decrement local maker balance per vendor's
            # `currentBalanceByUserId[userId] = makerBalance - maker.amount`
            # (calculate-cpmm.ts:282). No floor: balances can go negative
            # via FP / exotic ledger states; the next iteration's
            # `<= 0` check is the gate.
            if local_balances is not None and uid is not None and uid in local_balances:
                local_balances[uid] = local_balances[uid] - fill_maker_amount

            # Record consumption for clone-based limit tracking
            if consumption_report is not None and fill_cost > 0:
                oid = cur_limit.order_id
                if oid is not None:
                    consumption_report[oid] = consumption_report.get(oid, 0.0) + fill_cost

            # Record per-maker mana consumed for simulator-side balance tracking
            if maker_amount_report is not None and uid is not None and fill_maker_amount > 0:
                maker_amount_report[uid] = maker_amount_report.get(uid, 0.0) + fill_maker_amount

            total_cost += fill_cost
            total_shares += fill_shares
            remaining_cost -= fill_cost
            remaining_shares -= fill_shares
            cursor += 1  # Cursor advancement IS the epsilon

            if remaining_cost <= BUDGET_EPSILON or remaining_shares <= BUDGET_EPSILON:
                break

        # Check if budget exhausted after limit fills
        if remaining_cost <= BUDGET_EPSILON or remaining_shares <= BUDGET_EPSILON:
            break

    return (total_cost, total_shares, current_y, current_n)


# =============================================================================
# Convenience wrappers for limit-aware calculations
# =============================================================================


def cost_to_probability_with_limits(
    y: float,
    n: float,
    target_prob: float,
    position: str,
    order_book: OrderBook,
    ignore_limits: bool = False,
    p: float = 0.5,
    maker_balances: dict[str, float] | None = None,
) -> tuple[float, float]:
    """Calculate cost and shares to reach target probability, considering limits.

    Layer: 4, User-facing: NO

    Args:
        y: YES shares in pool
        n: NO shares in pool
        target_prob: Target P(YES) probability
        position: "YES" or "NO" - which position to buy
        order_book: Order book with limit orders
        ignore_limits: If True, use pure AMM (ignore order book)
        p: Probability parameter (default 0.5)
        maker_balances: Optional `{userId: balance}` for matcher-faithful
            limit clamping. See `buy_with_limits`.

    Returns:
        Tuple of (cost, shares)
    """
    if ignore_limits:
        cost = cost_to_probability(y, n, target_prob, position, p=p)
        shares = shares_to_probability(y, n, target_prob, position, p=p)
        return (cost, shares)

    cost, shares, _, _ = buy_with_limits(
        y, n, position, order_book,
        target_prob=target_prob,
        p=p,
        maker_balances=maker_balances,
    )
    return (cost, shares)


def cost_for_shares_with_limits(
    y: float,
    n: float,
    shares: float,
    position: str,
    order_book: OrderBook,
    ignore_limits: bool = False,
    p: float = 0.5,
    maker_balances: dict[str, float] | None = None,
) -> float:
    """Calculate cost to buy a given number of shares, considering limits.

    Layer: 4, User-facing: NO

    Args:
        y: YES shares in pool
        n: NO shares in pool
        shares: Number of shares to buy
        position: "YES" or "NO" - which position to buy
        order_book: Order book with limit orders
        ignore_limits: If True, use pure AMM (ignore order book)
        p: Probability parameter (default 0.5)
        maker_balances: Optional `{userId: balance}` for matcher-faithful
            limit clamping. See `buy_with_limits`.

    Returns:
        Cost in mana
    """
    if ignore_limits:
        return cost_for_shares(y, n, shares, position, p=p)

    cost, _, _, _ = buy_with_limits(
        y, n, position, order_book,
        target_shares=shares,
        p=p,
        maker_balances=maker_balances,
    )
    return cost


def shares_for_cost_with_limits(
    y: float,
    n: float,
    cost: float,
    position: str,
    order_book: OrderBook,
    ignore_limits: bool = False,
    p: float = 0.5,
    maker_balances: dict[str, float] | None = None,
) -> float:
    """Calculate shares received for a given cost, considering limits.

    Layer: 4, User-facing: NO

    Args:
        y: YES shares in pool
        n: NO shares in pool
        cost: Amount of mana to spend
        position: "YES" or "NO" - which position to buy
        order_book: Order book with limit orders
        ignore_limits: If True, use pure AMM (ignore order book)
        p: Probability parameter (default 0.5)
        maker_balances: Optional `{userId: balance}` for matcher-faithful
            limit clamping. See `buy_with_limits`.

    Returns:
        Number of shares received
    """
    if ignore_limits:
        return shares_for_cost(y, n, cost, position, p=p)

    _, shares, _, _ = buy_with_limits(
        y, n, position, order_book,
        target_cost=cost,
        p=p,
        maker_balances=maker_balances,
    )
    return shares


# =============================================================================
# Validation helpers
# =============================================================================


def validate_trade_result(
    y: float, n: float, cost: float, shares: float, position: str
) -> bool:
    """Validate that cost and shares are consistent for a trade.

    Useful for debugging and testing.

    Args:
        y: YES shares in pool before trade
        n: NO shares in pool before trade
        cost: Cost paid
        shares: Shares received
        position: "YES" or "NO"

    Returns:
        True if the values are consistent (within floating point tolerance)
    """
    # Check forward: cost_for_shares matches
    expected_cost = cost_for_shares(y, n, shares, position)
    cost_match = abs(expected_cost - cost) < 1e-8

    # Check backward: shares_for_cost matches
    expected_shares = shares_for_cost(y, n, cost, position)
    shares_match = abs(expected_shares - shares) < 1e-8

    return cost_match and shares_match
