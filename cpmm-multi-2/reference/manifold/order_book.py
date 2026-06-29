"""Order book data structures for limit order simulation.

This module provides data structures for representing limit orders and order books.
The simulation logic that uses these structures lives in amm_core.py.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class CrossedLimitsError(ValueError):
    """Raised when order book contains crossed limits that should have matched.

    This indicates a bug in Manifold's limit order matching - opposing limits
    at the same price should match against each other but didn't.
    """

    def __init__(self, crossed_prices: list[float], details: str = ""):
        self.crossed_prices = crossed_prices
        self.details = details
        prices_str = ", ".join(f"{p:.2%}" for p in crossed_prices)
        super().__init__(
            f"Order book has crossed limits at prices: {prices_str}. "
            f"BUY YES and BUY NO limits at same price should have matched. "
            f"This is a Manifold bug. {details}"
        )


class LimitOrder:
    """A limit order in the order book.

    Attributes:
        price: The probability at which this limit is set (0-1)
        size: Number of shares available at this price
        outcome: "YES" or "NO" - which side this limit is on
        order_id: Optional API limit order ID, for tracking consumption across clones
        user_id: Optional API maker (Manifold user) ID. When provided, lets the
            simulator clamp this limit's fill against the maker's account
            balance, mirroring Manifold's matcher behavior at
            `vendor/manifold/common/src/calculate-cpmm.ts:388-396`. Optional
            for back-compat with snapshots / hand-built test fixtures from
            before the maker-balance feature.
    """

    def __init__(
        self,
        price: float,
        size: float,
        outcome: str = "YES",
        order_id: str | None = None,
        user_id: str | None = None,
    ):
        if not 0 < price < 1:
            raise ValueError(f"price must be in (0, 1), got {price}")
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        outcome = outcome.upper()
        if outcome not in ("YES", "NO"):
            raise ValueError(f"outcome must be YES or NO, got {outcome}")

        self.price = price
        self.size = size
        self.outcome = outcome
        self.order_id = order_id
        self.user_id = user_id

    def __repr__(self):
        return f"LimitOrder({self.outcome} @ {self.price:.2%}, size={self.size})"


class OrderBook:
    """Order book with sorted limit orders.

    For YES purchases (price moving UP):
        - limits_above: NO limits above current price, sorted ascending
        - BUY NO limits get filled as we push price up past them

    For NO purchases (price moving DOWN):
        - limits_below: YES limits below current price, sorted descending
        - BUY YES limits get filled as we push price down past them
    """

    def __init__(
        self,
        limits_above: list[LimitOrder] | None = None,
        limits_below: list[LimitOrder] | None = None,
    ):
        self.limits_above = sorted(limits_above or [], key=lambda x: x.price)
        self.limits_below = sorted(
            limits_below or [], key=lambda x: x.price, reverse=True
        )

    @classmethod
    def empty(cls) -> OrderBook:
        """Create an empty order book."""
        return cls([], [])

    @classmethod
    def from_api_bets(
        cls,
        limit_orders: list[dict],
        answer_id: str | None = None,
        validate_crossed: bool = False,
        now_ms: int | None = None,
    ) -> OrderBook:
        """Create an OrderBook from API limit order data.

        Args:
            limit_orders: List of bet dicts from get_bets(kinds="open-limit")
                Each has: outcome, limitProb, orderAmount, amount, answerId (optional)
            answer_id: For multi-choice markets, filter to only this answer.
                       For binary markets, pass None.
            validate_crossed: If True, raise CrossedLimitsError when
                            BUY YES and BUY NO limits exist at the same price.
                            This indicates a Manifold bug. Default False to avoid
                            breaking existing code that uses data with stale limits.
            now_ms: Current time in milliseconds since epoch. Used to filter
                    expired limit orders (matching Manifold's computeFills filter:
                    calculate-cpmm.ts:234). Defaults to wall clock time.

        Returns:
            OrderBook with limit orders converted to LimitOrder objects.

        Raises:
            CrossedLimitsError: If validate_crossed=True and crossed limits detected.

        Note:
            Remaining shares are calculated as: (orderAmount - amount) / price
            where price is limitProb for YES or (1 - limitProb) for NO.
        """
        if now_ms is None:
            now_ms = int(time.time() * 1000)

        limits_above = []  # YES limits (above current price)
        limits_below = []  # NO limits (below current price)

        for order in limit_orders:
            # Skip if filtering by answer_id and this doesn't match
            order_answer_id = order.get("answerId")
            if answer_id is not None and order_answer_id != answer_id:
                continue

            # Skip expired limit orders (matches Manifold computeFills filter:
            # calculate-cpmm.ts:234 — bet.expiresAt ? bet.expiresAt > now : true)
            expires_at = order.get("expiresAt")
            if expires_at is not None and expires_at <= now_ms:
                continue

            outcome = order.get("outcome", "").upper()
            limit_prob = order.get("limitProb")
            order_amount = order.get("orderAmount", 0)
            filled_amount = order.get("amount", 0)
            oid = order.get("id")
            uid = order.get("userId")

            # Skip malformed or fully-filled orders
            if not outcome or limit_prob is None:
                continue
            if limit_prob <= 0 or limit_prob >= 1:
                continue
            if order_amount <= filled_amount:
                continue

            remaining_mana = order_amount - filled_amount

            if outcome == "YES":
                # BUY YES limit at X%: filled when price DROPS to X%
                # Crossed when we sell YES (push price DOWN) → limits_below
                price = limit_prob
                remaining_shares = remaining_mana / price
                try:
                    limit = LimitOrder(price=price, size=remaining_shares, outcome="YES", order_id=oid, user_id=uid)
                    limits_below.append(limit)
                except ValueError:
                    # Skip invalid orders (e.g., zero or negative size)
                    continue
            elif outcome == "NO":
                # BUY NO limit at X%: filled when YES price RISES to X%
                # Crossed when we buy YES (push price UP) → limits_above
                price = limit_prob  # The YES probability threshold
                no_price = 1 - limit_prob  # Cost per NO share
                remaining_shares = remaining_mana / no_price
                try:
                    limit = LimitOrder(price=price, size=remaining_shares, outcome="NO", order_id=oid, user_id=uid)
                    limits_above.append(limit)
                except ValueError:
                    continue

        # Validate for crossed limits (Manifold bug)
        if validate_crossed:
            # Get prices where YES limits exist (in limits_below)
            yes_prices = {lim.price for lim in limits_below}
            # Get prices where NO limits exist (in limits_above)
            no_prices = {lim.price for lim in limits_above}
            # Find prices that appear in both - these are crossed
            crossed = yes_prices & no_prices

            if crossed:
                # Build diagnostic details
                crossed_sorted = sorted(crossed)
                details_parts = []
                for price in crossed_sorted:
                    yes_size = sum(
                        lim.size for lim in limits_below if lim.price == price
                    )
                    no_size = sum(
                        lim.size for lim in limits_above if lim.price == price
                    )
                    details_parts.append(
                        f"  @ {price:.2%}: YES={yes_size:.1f} shares, NO={no_size:.1f} shares"
                    )
                details = "\n".join(details_parts)
                if answer_id:
                    details = f"Answer ID: {answer_id}\n{details}"
                raise CrossedLimitsError(crossed_sorted, details)

        return cls(limits_above=limits_above, limits_below=limits_below)

    def limits_in_range(
        self, start_prob: float, end_prob: float, direction: str,
        tolerance: float = 1e-9
    ) -> list[LimitOrder]:
        """Get limits that would be crossed moving from start to end probability.

        Args:
            start_prob: Starting probability (computed from pool, may have FP error)
            end_prob: Target probability
            direction: "UP" (buying YES) or "DOWN" (buying NO)
            tolerance: Floating-point tolerance for start boundary comparison.
                       Limit prices are exact from API, but start_prob is computed
                       from pool state and may have floating-point error.

        Returns:
            List of limits between start and end, in order of crossing.
            Limits AT start ARE included (we're at the trigger price).
            Limits AT end are NOT included (we stop there, don't cross).

        Note:
            The tolerance is applied to the START boundary only, not the end.
            This handles the case where pool probability computes to slightly
            less/more than a limit price due to floating-point arithmetic.
            Example: pool gives 0.18999999999999995, limit is at 0.19 (exact).
        """
        if direction == "UP":
            # Moving price up, cross limits_above between start and end
            # Include start (with tolerance), exclude end (strict)
            # Use (start_prob - tolerance) to include limits AT the start
            return [
                lim
                for lim in self.limits_above
                if (start_prob - tolerance) <= lim.price < end_prob
            ]
        else:
            # Moving price down, cross limits_below between start and end
            # Include start (with tolerance), exclude end (strict)
            # Use (start_prob + tolerance) to include limits AT the start
            return [
                lim
                for lim in self.limits_below
                if end_prob < lim.price <= (start_prob + tolerance)
            ]

    def limits_at_price(
        self, price: float, direction: str, tolerance: float = 1e-9
    ) -> list[LimitOrder]:
        """Get limits at exactly the specified price.

        Args:
            price: Target probability (P(YES))
            direction: "UP" (buying YES, check limits_above) or "DOWN" (buying NO, check limits_below)
            tolerance: Floating point tolerance for price comparison

        Returns:
            List of limits at exactly this price.
        """
        if direction == "UP":
            return [
                lim
                for lim in self.limits_above
                if abs(lim.price - price) < tolerance
            ]
        else:
            return [
                lim
                for lim in self.limits_below
                if abs(lim.price - price) < tolerance
            ]

    def total_liquidity_at_price(
        self, price: float, direction: str, tolerance: float = 1e-9
    ) -> float:
        """Get total limit order liquidity (in shares) at a specific price.

        Args:
            price: Target probability (P(YES))
            direction: "UP" (buying YES) or "DOWN" (buying NO)
            tolerance: Floating point tolerance for price comparison

        Returns:
            Total shares available from limit orders at this price.
        """
        limits = self.limits_at_price(price, direction, tolerance)
        return sum(lim.size for lim in limits)

    def get_crossed_prices(self, tolerance: float = 1e-9) -> list[float]:
        """Find prices where both YES and NO limits exist (crossed/stale limits).

        Crossed limits indicate a Manifold bug - opposing limits at the same
        price should have matched against each other but didn't.

        Args:
            tolerance: Floating point tolerance for price comparison

        Returns:
            List of prices where both YES and NO limits exist, sorted ascending.
            Empty list if no crossed limits.
        """
        # Get prices from each side
        yes_prices = {lim.price for lim in self.limits_below}
        no_prices = {lim.price for lim in self.limits_above}

        # Find crossed prices (within tolerance)
        crossed = set()
        for yes_price in yes_prices:
            for no_price in no_prices:
                if abs(yes_price - no_price) < tolerance:
                    crossed.add(yes_price)

        return sorted(crossed)

    def has_crossed_limits(self, tolerance: float = 1e-9) -> bool:
        """Check if this order book has crossed limits.

        Returns:
            True if BUY YES and BUY NO limits exist at the same price.
        """
        return len(self.get_crossed_prices(tolerance)) > 0
