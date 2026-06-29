"""MarketSimulator: Local market state for calculations and simulations.

A simulator holds a local copy of market data that can be manipulated without
affecting the original. This is where all the AMM calculations happen.

Uses amm_core for fundamental CPMM formulas to maintain single source of truth.
"""

import copy
from typing import Any, Dict, List, Optional

from manifold.amm_core import (
    buy_with_limits,
    cost_for_shares_with_limits,
    cost_to_probability,
    cost_to_probability_with_limits,
    probability_from_pool,
    shares_for_cost,
    shares_for_cost_with_limits,
    shares_to_probability,
)
from manifold.amm_core import (
    cost_for_shares as amm_cost_for_shares,
)
from manifold.amm_core import (
    pool_after_trade as amm_pool_after_trade,
)
from manifold.multi_choice import MultiSharesResult, _apply_arb_trades
from manifold.types import MarketDict

# Also keep the non-aliased import for backwards compatibility in the module
cost_for_shares = amm_cost_for_shares
pool_after_trade = amm_pool_after_trade

# Numerical precision constants
EPSILON = 1e-10  # For floating point comparisons

# Manifold CPMM probability limits
# These limits are enforced by Manifold's API for market buys
# Source: vendor/manifold/common/src/contract.ts:492-493
MAX_CPMM_PROB = 0.99
MIN_CPMM_PROB = 0.01

# Fail-fast tolerances for the "modelable at p=0.5" ingestion guard. A real v1
# (cpmm-multi-1) answer reports prob == N/(Y+N) exactly (computed from the same
# pool); a cpmm-multi-2 answer (per-answer p != 0.5) is off by the full p-skew
# (e.g. 0.7 vs 0.5), orders of magnitude above this floor. The tolerance only
# absorbs snapshot staleness / float noise.
_MODELABLE_PROB_TOL = 1e-3
_MODELABLE_P_TOL = 1e-9

# Multi-choice mechanisms our p=0.5 model handles. Both MULTIPLE_CHOICE and
# MULTI_NUMERIC report `cpmm-multi-1` (see _is_multi_choice_type). A new
# mechanism string (e.g. `cpmm-multi-2`) is refused by name at ingestion.
_SUPPORTED_MULTI_MECHANISMS = frozenset({"cpmm-multi-1"})

# Outcome types that flow through the multi-choice (per-answer pool) pricing.
_MULTI_CHOICE_OUTCOME_TYPES = ("MULTIPLE_CHOICE", "MULTI_NUMERIC")


class UnmodeledMarketError(ValueError):
    """A market cannot be safely priced by our p=0.5 multi-choice model.

    Raised at ingestion when the market declares a mechanism we don't model, or
    an answer's reported probability can't be reproduced by the p=0.5 pool
    formula N/(Y+N) (or carries an explicit per-answer ``p`` != 0.5). This is the
    cpmm-multi-2 fail-fast: rather than silently forecast a wrong number on a
    per-answer-p market, we halt. Subclass of ``ValueError`` so existing broad
    handlers still catch it, but distinct so callers (e.g. the arb bot) can skip
    an unmodeled leg deliberately.
    """


def _extract_answer_pools(answer: dict) -> tuple:
    """(YES, NO) pool for an answer. MULTIPLE_CHOICE uses poolYes/poolNo;
    MULTI_NUMERIC uses pool.YES/pool.NO. Raises ValueError if absent.

    Module-level so both `MarketSimulator._get_answer_pools` and the standalone
    `assert_market_modelable` share one extraction (no duplicated format logic).
    """
    if 'poolYes' in answer or 'poolYES' in answer:
        pool_yes = answer.get('poolYes', answer.get('poolYES'))
        pool_no = answer.get('poolNo', answer.get('poolNO'))
    else:
        pool = answer.get('pool', {})
        pool_yes = pool.get('YES')
        pool_no = pool.get('NO')
    if pool_yes is None or pool_no is None:
        name = answer.get('text', answer.get('id', 'unknown'))
        raise ValueError(f"Answer '{name}' missing pool data")
    return pool_yes, pool_no


def assert_market_modelable(market_data: dict) -> None:
    """Raise `UnmodeledMarketError` if this multi-choice market can't be priced
    by our p=0.5 model (the cpmm-multi-2 fail-fast).

    Single source of truth for the guard, called from `MarketSimulator.__init__`
    (ingestion) AND reusable by the arb bot to skip a pattern whose leg sits on an
    unmodeled market. Two layers (see `MarketSimulator._assert_modelable_at_p_half`
    for the full rationale):
      1. **Mechanism allowlist (direct):** we model only `cpmm-multi-1`; a new
         string like `cpmm-multi-2` is refused by name. Absent mechanism falls
         through to (2) rather than rejecting.
      2. **Un-foolable pool check (backup):** the p=0.5 formula N/(Y+N) must
         reproduce each live answer's reported prob; no explicit per-answer
         p != 0.5. Catches a v2 reusing the cpmm-multi-1 string with `p` fields,
         or a weight field-name surprise.

    Non-multi-choice markets (binary, etc.) are a no-op here.
    """
    if market_data.get('outcomeType') not in _MULTI_CHOICE_OUTCOME_TYPES:
        return
    market_id = market_data.get('id')

    # (1) Direct signal: refuse an unsupported mechanism by name.
    mechanism = market_data.get('mechanism')
    if mechanism is not None and mechanism not in _SUPPORTED_MULTI_MECHANISMS:
        raise UnmodeledMarketError(
            f"Market {market_id!r} has mechanism {mechanism!r}, which we don't "
            f"model (supported: {sorted(_SUPPORTED_MULTI_MECHANISMS)}). This is "
            f"the cpmm-multi-2 fail-fast — refusing to price it rather than "
            f"forecast a wrong number."
        )

    # (2) Un-foolable backup: pool/prob consistency + explicit-p check.
    for ans in market_data.get('answers') or []:
        # Skip resolved answers (settled prob is 0/1 by fiat, not a pool reading).
        if ans.get('resolution') is not None or ans.get('resolutionTime'):
            continue
        p = ans.get('p')
        if p is not None and abs(float(p) - 0.5) > _MODELABLE_P_TOL:
            raise UnmodeledMarketError(
                f"Market {market_id!r} answer {ans.get('id')!r} carries explicit "
                f"per-answer p={p} != 0.5 (cpmm-multi-2). Our model prices "
                f"multi-choice at p=0.5 only; refusing to misprice it."
            )
        reported = ans.get('prob', ans.get('probability'))
        if reported is None:
            continue
        try:
            y, n = _extract_answer_pools(ans)
        except (ValueError, TypeError):
            continue  # no pool to check against (e.g. resolved/degenerate)
        if y is None or n is None or (y + n) <= 0:
            continue
        implied = probability_from_pool(y, n, 0.5)  # == N/(Y+N)
        if abs(implied - float(reported)) > _MODELABLE_PROB_TOL:
            raise UnmodeledMarketError(
                f"Market {market_id!r} answer {ans.get('id')!r}: reported prob "
                f"{float(reported):.6f} != pool-implied {implied:.6f} at p=0.5 "
                f"(|Δ|={abs(implied - float(reported)):.2e} > {_MODELABLE_PROB_TOL}). "
                f"Looks like a cpmm-multi-2 (per-answer p) market we don't yet "
                f"model — refusing to price it at p=0.5 rather than forecast a "
                f"wrong number."
            )


def is_market_modelable(market_data: dict) -> bool:
    """Bool wrapper around `assert_market_modelable` (no raise)."""
    try:
        assert_market_modelable(market_data)
        return True
    except UnmodeledMarketError:
        return False


class MarketSimulator:
    """Local simulation of market state.

    The simulator holds an isolated copy of market data and provides methods for:
    - Calculating trade costs and outcomes
    - Simulating trades (modifying local state)
    - Querying current probabilities

    Each simulator is independent - changes to one don't affect others.

    Example:
        >>> handle = MarketHandle(client, "market-slug")
        >>> sim = handle.create_simulator()
        >>>
        >>> # Calculations (don't change state)
        >>> cost = sim.calculate_buy_cost("YES", 100)
        >>>
        >>> # Simulations (DO change state)
        >>> trade = sim.simulate_buy("YES", 100)
        >>> new_prob = sim.get_probability("YES")
    """

    def __init__(self, market_data: MarketDict, _internal: bool = False, _skip_initial: bool = False, _validate: bool = True):
        """Initialize simulator with market data.

        PRIVATE: Do not construct directly. Use:
        - MarketHandle.create_simulator() for production code
        - MarketHandle.get_simulator() in ExecutionContext
        - MarketBuilder (coming soon) for tests

        Args:
            market_data: Market data dictionary (will be copied to ensure isolation)
            _internal: Must be True. Enforces use of proper factory methods.
            _skip_initial: If True, skip copying initial_data (optimization for clones)

        Raises:
            TypeError: If _internal is not True
        """
        if not _internal:
            raise TypeError(
                "MarketSimulator cannot be constructed directly.\n"
                "Use one of these instead:\n"
                "  - handle.create_simulator() for one-off simulations\n"
                "  - handle.get_simulator() within ExecutionContext.simulation_mode()\n"
                "  - MarketBuilder for tests (coming soon)\n"
                "\nTo temporarily suppress this error in tests, pass _internal=True"
            )

        self.data = copy.deepcopy(market_data)  # Each simulator gets its own copy
        # Skip initial_data for clones (performance optimization - not needed for temp calculations)
        self.initial_data = None if _skip_initial else copy.deepcopy(market_data)

        # Set up based on market type
        self.market_type = market_data['outcomeType']
        self.market_id = market_data['id']
        self.slug = market_data.get('slug', '')
        self.question = market_data.get('question', '')

        # Limit orders for simulation (set via set_order_book())
        # When set, calculation methods will include limit order fills
        self._order_book = None  # Optional[OrderBook] - for binary markets
        self._limit_orders_raw = None  # Raw API data for multi-choice (per-answer filtering)
        self._limit_orders_now_ms = None  # Timestamp for expiry filtering
        self._order_book_cache = {}  # Per-answer OrderBook cache for multi-choice

        # Limit consumption tracking: maps limit order API ID → additional mana
        # consumed beyond the baseline in _limit_orders_raw. This enables clone()
        # to create independent consumption state, fixing the bug where binary
        # search iterations in calculate_shares_exact all saw the same unconsumed
        # limits. Typically 0-5 entries per simulator.
        self._limit_consumption: dict[str, float] = {}
        # Set of answer IDs with consumption entries, for selective cache invalidation
        self._answers_with_consumption: set[str] = set()

        # Maker balance tracking. Mirrors the matcher's `min(orderRemaining,
        # makerCurrentBalance)` clamp at calculate-cpmm.ts:388-396.
        # `_maker_balances` is mutable and is decremented by `apply_trade` as
        # fills land. Read-only `simulate_buy`-style calls pass it into
        # `buy_with_limits` which uses a local copy. Empty (or absent
        # user_id on each LimitOrder) → no clamping (legacy behavior).
        self._maker_balances: dict[str, float] = {}

        # Answer lookup index for O(1) lookups instead of O(n) linear search
        # Maps both text -> index and id -> index for the answers list
        # Rebuilt on demand, shared on clone (answer list structure is preserved)
        self._answer_index: dict[str, int] | None = None

        # Get p value for binary markets
        if self.market_type == 'BINARY':
            self.p = market_data.get('p')
            if self.p is None:
                raise ValueError(f"Binary market {self.market_id} missing 'p' value")
            # Validate p value
            if not (0 < self.p < 1):
                raise ValueError(f"Binary market {self.market_id} has invalid p value: {self.p}. Must be between 0 and 1.")
        else:
            # Multi-choice: `self.p` is the fallback default only. Per-answer p
            # (cpmm-multi-2) is read per-site via `_answer_p`/`_p_of`
            # (`answer.get('p', 0.5)`); v1 answers have no `p` and price at 0.5.
            self.p = 0.5
            # Fail-fast: refuse a market we can't model at p=0.5 (cpmm-multi-2).
            # Skipped for clones (they bypass __init__) and re-validation is
            # cheap; tests building synthetic per-answer-p markets pass
            # _validate=False.
            if _validate:
                self._assert_modelable_at_p_half()

    def _assert_modelable_at_p_half(self) -> None:
        """Halt if this multi-choice market isn't priceable at p=0.5 (the v1 model).

        Layer: 3, User-facing: NO

        Thin instance wrapper over the module-level `assert_market_modelable`
        (the single source of truth, also reused by the arb bot). A
        cpmm-multi-2 market (per-answer p != 0.5) reports
        ``outcomeType == 'MULTIPLE_CHOICE'`` and would otherwise flow through the
        identical p=0.5 paths and silently misprice. Two layers of defense live in
        the module function: (1) a mechanism allowlist (refuse e.g.
        ``cpmm-multi-2`` by name), and (2) an un-foolable pool/prob consistency
        check (catches a v2 reusing the ``cpmm-multi-1`` string with per-answer
        ``p``, or a weight field-name surprise).

        Raises:
            UnmodeledMarketError: unsupported mechanism, explicit per-answer
                p != 0.5, or reported prob not reproducible at p=0.5.
        """
        assert_market_modelable(self.data)

    def _is_multi_choice_type(self) -> bool:
        """Check if market is a multi-choice type (MULTIPLE_CHOICE or MULTI_NUMERIC).

        Both types use the same CPMM mechanism (cpmm-multi-1) and pool structure.
        MULTI_NUMERIC is just a variant with numeric thresholds and different UI.
        """
        return self.market_type in _MULTI_CHOICE_OUTCOME_TYPES

    def _get_answer_index(self) -> dict[str, int]:
        """Get or build the answer lookup index.

        Returns a dict mapping both text and id to the answer's index in answers list.
        This enables O(1) lookups instead of O(n) linear search.
        """
        if self._answer_index is None:
            self._answer_index = {}
            for i, answer in enumerate(self.data.get('answers', [])):
                text = answer.get('text')
                aid = answer.get('id')
                if text:
                    self._answer_index[text] = i
                if aid:
                    self._answer_index[aid] = i
        return self._answer_index

    def _get_answer(self, outcome: str) -> dict | None:
        """Get answer dict by text or id using O(1) index lookup.

        Args:
            outcome: Answer text or ID

        Returns:
            Answer dict or None if not found
        """
        index = self._get_answer_index()
        idx = index.get(outcome)
        if idx is None:
            return None
        answers = self.data.get('answers', [])
        if idx < len(answers):
            return answers[idx]
        return None

    def _get_answer_pools(self, answer: dict) -> tuple:
        """Extract YES and NO pool values from an answer.

        Handles both MULTIPLE_CHOICE (poolYes/poolNo) and MULTI_NUMERIC
        (pool.YES/pool.NO) formats. Delegates to the module-level
        `_extract_answer_pools` (shared with `assert_market_modelable`).

        Raises:
            ValueError: If pool data is missing or invalid
        """
        return _extract_answer_pools(answer)

    def set_limit_orders(self, limit_orders: list, now_ms: int | None = None) -> None:
        """Set limit orders for this simulator.

        For binary markets, builds an OrderBook immediately.
        For multi-choice markets, stores raw data and builds per-answer OrderBooks lazily.

        Args:
            limit_orders: List of limit order bet dicts from API
                         (from context.get_limit_orders() or similar)
            now_ms: Current time in milliseconds for expiry filtering.
                    Defaults to wall clock time. Pass snapshot timestamp
                    for snapshot-based testing.
        """
        from manifold.order_book import OrderBook

        self._limit_orders_raw = limit_orders
        self._limit_orders_now_ms = now_ms  # Stored for lazy order book builds
        self._order_book_cache = {}  # Clear cache when limit orders change

        if self.market_type == 'BINARY':
            # Binary: one order book for the whole market
            self._order_book = OrderBook.from_api_bets(limit_orders, now_ms=now_ms)
        else:
            # Multi-choice: will build per-answer on demand (cached)
            self._order_book = None

    def set_maker_balances(self, balances: dict[str, float]) -> None:
        """Provide public-API maker balances for matcher-faithful clamping.

        Each fill against a limit owned by `userId` will be clamped at
        `min(orderRemaining, balance)` and the balance decremented within
        the trade — mirroring vendor's `currentBalanceByUserId` mechanism
        at `calculate-cpmm.ts:252,279-289`.

        If never called (or called with an empty dict), legacy non-clamping
        behavior is preserved. Limits whose `user_id` isn't in the dict
        are also not clamped (vendor-equivalent: `matchableUserBalance ??
        amountRemaining`). The intended source is `user_balances.json`
        from the snapshot or fresh `get_user_by_id` reads.

        Args:
            balances: `{userId: balance_mana}`. Make a copy on the way in
                so caller mutations don't bleed into the simulator's state.
        """
        self._maker_balances = dict(balances)

    # ============= Layer-3 helpers: balance-aware Layer-4 wrappers =============
    # `amm_core.*_with_limits` functions are nominally Layer 4 (state passed in
    # as args). When called from Layer 1 helpers in this file, those helpers
    # must apply this simulator's balance state. Rather than threading
    # `maker_balances=self._maker_balances if self._maker_balances else None`
    # at every call site (responsibility-spreading), wrap the Layer 4 calls
    # here and have callers go through these wrappers.
    #
    # Per docs/layer-architecture.md: Layer 3 owns the simulator's state;
    # Layer 4 is pure(ish) pool-math. These wrappers are the Layer 3 ↔ 4
    # bridge: pool/orderbook/balance state in, AMM result out.

    def _maker_balances_arg(self) -> Optional[dict[str, float]]:
        """Return `self._maker_balances` if non-empty, else None.

        Layer-4 `*_with_limits` helpers expect None to mean "no clamping"
        (legacy back-compat); we don't want to send empty dicts which would
        be misread as "no makers known, treat all as undefined-balance".
        Same convention everywhere in this module.
        """
        return self._maker_balances if self._maker_balances else None

    def _pool_cost_for_shares(
        self, Y: float, N: float, shares: float, position: str,
        order_book, *, p: Optional[float] = None,
    ) -> float:
        """Layer-3 wrapper: cost for shares from a given pool, balance-aware.

        Layer: 3, User-facing: NO
        """
        return cost_for_shares_with_limits(
            Y, N, shares, position, order_book,
            p=p if p is not None else self.p,
            maker_balances=self._maker_balances_arg(),
        )

    def _pool_shares_for_cost(
        self, Y: float, N: float, amount: float, position: str,
        order_book, *, p: Optional[float] = None,
    ) -> float:
        """Layer-3 wrapper: shares for cost from a given pool, balance-aware.

        Layer: 3, User-facing: NO
        """
        return shares_for_cost_with_limits(
            Y, N, amount, position, order_book,
            p=p if p is not None else self.p,
            maker_balances=self._maker_balances_arg(),
        )

    def _pool_cost_to_probability(
        self, Y: float, N: float, target_prob: float, position: str,
        order_book, *, p: Optional[float] = None,
    ) -> tuple[float, float]:
        """Layer-3 wrapper: (cost, shares) to reach `target_prob`, balance-aware.

        Layer: 3, User-facing: NO
        """
        return cost_to_probability_with_limits(
            Y, N, target_prob, position, order_book,
            p=p if p is not None else self.p,
            maker_balances=self._maker_balances_arg(),
        )

    def _balance_aware_limit_shares_at(
        self, order_book, target_prob: float, direction: str,
    ) -> float:
        """Sum the actually-fillable limit shares at `target_prob`.

        Layer: 3, User-facing: NO

        `OrderBook.total_liquidity_at_price` returns nominal capacity. For
        balance-aware reasoning (e.g., `buy_to_probability_interval`), we
        instead sum `min(lim.size, balance / maker_cost_per_share)` per limit.
        Limits without `user_id`, or whose maker isn't in
        `self._maker_balances`, fall through to nominal capacity (legacy
        behavior — same convention as Layer 4 `*_with_limits` helpers).
        """
        balances = self._maker_balances_arg()
        if balances is None:
            return order_book.total_liquidity_at_price(target_prob, direction)

        total = 0.0
        for lim in order_book.limits_at_price(target_prob, direction):
            if lim.user_id is None or lim.user_id not in balances:
                total += lim.size
                continue
            balance = balances[lim.user_id]
            if balance <= 0:
                continue
            # YES limit: maker pays `lim.price` per YES share. NO limit:
            # maker pays `1 - lim.price` per NO share. Mirrors
            # `_fill_single_limit`'s `maker_cost_per_share`.
            maker_cost_per_share = (
                lim.price if lim.outcome == "YES" else (1.0 - lim.price)
            )
            if maker_cost_per_share <= 0:
                total += lim.size
                continue
            total += min(lim.size, balance / maker_cost_per_share)
        return total

    def get_order_book(self, answer_id: str | None = None):
        """Get the OrderBook for calculations.

        Layer: 3, User-facing: NO

        If limit consumption has been tracked (via apply_trade), the returned
        OrderBook reflects reduced limit sizes based on consumption state.

        Args:
            answer_id: For multi-choice markets, the answer ID to filter by.
                       For binary markets, ignored.

        Returns:
            OrderBook instance, or None if no limit orders are set.
        """
        from manifold.order_book import OrderBook

        if self._limit_orders_raw is None:
            return None

        if self.market_type == 'BINARY':
            # For binary markets with consumption, rebuild with adjusted limits
            if self._limit_consumption and self._order_book is not None:
                return self._build_adjusted_order_book(self._order_book)
            return self._order_book
        else:
            # Multi-choice: build order book filtered to this answer (cached)
            if answer_id is None:
                return OrderBook.empty()

            # If this answer has consumption entries, we need a fresh build
            # with adjusted limits (can't use shared cache)
            if answer_id in self._answers_with_consumption:
                # Build fresh each time since consumption changes per clone
                base_book = OrderBook.from_api_bets(
                    self._limit_orders_raw, answer_id=answer_id,
                    now_ms=self._limit_orders_now_ms,
                )
                return self._build_adjusted_order_book(base_book)

            # Fast path: no consumption for this answer, use shared cache
            if answer_id not in self._order_book_cache:
                self._order_book_cache[answer_id] = OrderBook.from_api_bets(
                    self._limit_orders_raw, answer_id=answer_id,
                    now_ms=self._limit_orders_now_ms,
                )
            return self._order_book_cache[answer_id]

    def _build_adjusted_order_book(self, base_book):
        """Build an OrderBook with limit sizes reduced by tracked consumption.

        Args:
            base_book: Original OrderBook to adjust

        Returns:
            New OrderBook with consumed amounts subtracted from limit sizes
        """
        from manifold.order_book import LimitOrder, OrderBook

        def adjust_limits(limits):
            adjusted = []
            for lim in limits:
                if lim.order_id and lim.order_id in self._limit_consumption:
                    consumed_mana = self._limit_consumption[lim.order_id]
                    # Convert consumed mana to shares
                    cost_per_share = lim.price if lim.outcome == "YES" else (1 - lim.price)
                    consumed_shares = consumed_mana / cost_per_share if cost_per_share > 0 else 0
                    remaining = lim.size - consumed_shares
                    if remaining > 1e-10:
                        adjusted.append(LimitOrder(
                            price=lim.price, size=remaining,
                            outcome=lim.outcome, order_id=lim.order_id,
                            user_id=lim.user_id,
                        ))
                    # else: fully consumed, skip
                else:
                    adjusted.append(lim)
            return adjusted

        return OrderBook(
            limits_above=adjust_limits(base_book.limits_above),
            limits_below=adjust_limits(base_book.limits_below),
        )

    def has_limit_orders(self) -> bool:
        """Check if this simulator has limit orders configured."""
        return self._limit_orders_raw is not None and len(self._limit_orders_raw) > 0

    def reset(self):
        """Reset market to initial state."""
        self.data = copy.deepcopy(self.initial_data)

    def copy(self) -> 'MarketSimulator':
        """Create an independent copy of this simulator."""
        return MarketSimulator(copy.deepcopy(self.data), _internal=True)

    # ============= Probability Queries =============

    def get_probability(self, outcome: Optional[str] = None, position: str = "YES") -> float:
        """Get current probability for an outcome.

        Layer: 3, User-facing: NO

        Args:
            outcome: Answer text or ID for multi-choice markets.
                    For binary markets, can be omitted (uses position parameter instead).
                    Legacy: "YES"/"NO" still works for binary (treated as position).
            position: "YES" or "NO" - which position's probability to return (default: "YES")

        Returns:
            Probability as decimal (0.0 to 1.0)
        """
        if self.market_type == 'BINARY':
            # Always calculate from pool (single source of truth)
            prob = self._calculate_probability_from_pool()

            if prob is None:
                raise ValueError(f"Cannot determine probability for market {self.market_id}: missing pool data")

            # Determine which position to return
            # Support legacy: outcome="YES"/"NO" treated as position (deprecated)
            if outcome is not None and outcome.upper() in ['YES', 'NO']:
                actual_position = outcome.upper()
            else:
                if outcome is not None and outcome.upper() not in ['YES', 'NO']:
                    raise ValueError(f"Binary markets don't accept outcome '{outcome}'. Use position parameter instead.")
                actual_position = position.upper()

            if actual_position == 'YES':
                return prob
            elif actual_position == 'NO':
                return 1 - prob
            else:
                raise ValueError(f"Invalid position for binary market: {position}")

        elif self._is_multi_choice_type():
            if outcome is None:
                raise ValueError("Multi-choice markets require outcome parameter")

            if 'answers' not in self.data:
                raise ValueError("Multi-choice market missing answers")

            # Find answer by text or ID using O(1) index lookup
            answer = self._get_answer(outcome)
            if answer is None:
                raise ValueError(f"Answer not found: {outcome}")

            # Always calculate from pool values (single source of truth).
            # Per-answer p (cpmm-multi-2 Slice 1b); v1 answers (no `p`) -> 0.5,
            # where probability_from_pool reduces to poolNo / (poolYes + poolNo).
            pool_yes, pool_no = self._get_answer_pools(answer)

            if pool_yes + pool_no == 0:
                raise ValueError(f"Answer '{outcome}' has empty pools")

            # YES probability for this answer.
            yes_prob = probability_from_pool(pool_yes, pool_no, self._p_of(answer))

            # Return YES or NO probability based on position parameter
            if position.upper() == 'YES':
                return yes_prob
            elif position.upper() == 'NO':
                return 1 - yes_prob
            else:
                raise ValueError(f"Invalid position: {position}")

        else:
            raise NotImplementedError(f"Market type {self.market_type} not supported")

    def get_all_probabilities(self) -> Dict[str, float]:
        """Get all outcome probabilities.

        Returns:
            Dictionary mapping outcome to probability
        """
        if self.market_type == 'BINARY':
            # Always calculate from pool
            prob = self._calculate_probability_from_pool()
            if prob is None:
                raise ValueError(f"Market {self.market_id} missing pool data")
            return {'YES': prob, 'NO': 1 - prob}

        elif self._is_multi_choice_type():
            if 'answers' not in self.data:
                return {}

            result = {}
            for answer in self.data['answers']:
                text = answer.get('text', answer.get('id', 'unknown'))

                # Always calculate from pool values
                pool_yes, pool_no = self._get_answer_pools(answer)

                if pool_yes + pool_no == 0:
                    raise ValueError(f"Answer '{text}' has empty pools")

                # Per-answer p (cpmm-multi-2 Slice 1b); v1 answers -> 0.5.
                prob = probability_from_pool(pool_yes, pool_no, self._p_of(answer))
                result[text] = prob
            return result

        else:
            raise NotImplementedError(f"Market type {self.market_type} not supported")

    def get_answer_ids(self) -> list[str]:
        """Get all answer IDs in a multi-choice market.

        Returns:
            List of answer IDs

        Raises:
            ValueError: If not a multi-choice market
        """
        if not self._is_multi_choice_type():
            raise ValueError("get_answer_ids() only valid for multi-choice markets")

        return [a.get('id') for a in self.data.get('answers', []) if a.get('id')]

    def _resolve_answer_id(self, outcome: str) -> str:
        """Resolve outcome (text or ID) to answer ID.

        Args:
            outcome: Answer text or answer ID

        Returns:
            The answer ID

        Raises:
            ValueError: If answer not found
        """
        for answer in self.data.get('answers', []):
            if answer.get('id') == outcome or answer.get('text') == outcome:
                return answer['id']
        raise ValueError(f"Answer '{outcome}' not found in market {self.market_id}")

    @staticmethod
    def _p_of(answer: dict) -> float:
        """Per-answer CPMM weight `p` for a multi-choice answer (cpmm-multi-2).

        Layer: 3, User-facing: NO

        Absent `p` -> 0.5, i.e. a `cpmm-multi-1` (v1) answer. This is the single
        place the v1-default lives; `probability_from_pool` / `cost_for_shares` /
        `pool_after_trade` all special-case p == 0.5, so a v1 answer is priced
        byte-identically to the old hardcoded path. Returns a concrete `float`
        so per-answer p threads through the amm primitives cleanly (unlike the
        legacy `self.p`, which is typed `object`).
        """
        return float(answer.get('p', 0.5))

    def _answer_p(self, answer_id: str) -> float:
        """Per-answer `p` for `answer_id` (cpmm-multi-2 Slice 1b).

        Layer: 3, User-facing: NO

        Looks the answer up and returns its `p` (default 0.5). Returns 0.5 for an
        unknown id so a missing answer prices as v1 rather than crashing here
        (callers that need the answer raise their own clearer error).
        """
        answer = self._get_answer(answer_id)
        return self._p_of(answer) if answer is not None else 0.5

    def probability_for(self, answer_id: str, pool: dict[str, float]) -> float:
        """Protocol method: probability of a hypothetical `pool` for `answer_id`.

        Layer: 3, User-facing: NO

        Prices a caller-computed pool using the answer's per-answer `p` weight
        (`answer.get('p', 0.5)`), without mutating state. cpmm-multi-2 Slice 1b:
        threads per-answer `p`; byte-identical to the old NO/(YES+NO) at p=0.5.
        """
        return probability_from_pool(pool['YES'], pool['NO'], self._answer_p(answer_id))

    def get_pool(self, answer_id: str) -> dict[str, float]:
        """Get current pool state for an answer.

        Layer: 3, User-facing: NO

        Args:
            answer_id: Answer ID (or text for convenience)

        Returns:
            {'YES': float, 'NO': float}
        """
        if not self._is_multi_choice_type():
            # Binary market - return the single pool
            pool = self.data.get('pool', {})
            return {
                'YES': pool.get('YES', self.data.get('poolYes', 0)),
                'NO': pool.get('NO', self.data.get('poolNo', 0))
            }

        # Multi-choice - find the answer using O(1) index lookup
        answer = self._get_answer(answer_id)
        if answer is None:
            raise ValueError(f"Answer not found: {answer_id}")
        pool_yes, pool_no = self._get_answer_pools(answer)
        return {'YES': pool_yes, 'NO': pool_no}

    def pool_after_trade(
        self, answer_id: str, amount: float, position: str
    ) -> dict[str, float]:
        """Calculate pool state after spending `amount` on `position`.

        Layer: 3, User-facing: NO

        This is PURE (does not mutate state) - safe for hypothetical calculations.
        Limit-aware: uses limit orders if available.

        Args:
            answer_id: The answer to trade in
            amount: Amount to spend (positive)
            position: 'YES' or 'NO'

        Returns:
            New pool state {'YES': float, 'NO': float}
        """
        pool = self.get_pool(answer_id)
        y, n = pool['YES'], pool['NO']
        p = self._answer_p(answer_id)  # per-answer p (cpmm-multi-2); v1 -> 0.5

        # Get order book for this answer (if available)
        order_book = self.get_order_book(answer_id)

        if order_book is not None and amount > 0:
            # Use limit-aware calculation. Pass current maker balances for
            # matcher-faithful clamping (read-only — buy_with_limits uses
            # a local copy and discards mutations on this read path).
            _, _, y_new, n_new = buy_with_limits(
                y, n, position, order_book, target_cost=amount, p=p,
                maker_balances=self._maker_balances if self._maker_balances else None,
            )
            return {'YES': y_new, 'NO': n_new}
        else:
            # Pure AMM calculation
            y_new, n_new = amm_pool_after_trade(y, n, amount, position, p=p)
            return {'YES': y_new, 'NO': n_new}

    # ============= LiquidityContext Protocol (Layer 3 - single-pool ops) =============
    # These are PRIMITIVE operations. They must NOT call Layer 2 (auto-arb).
    # See manifold/liquidity.py for layer architecture documentation.
    def cost_for_shares(
        self, answer_id: str, shares: float, position: str
    ) -> float:
        """Protocol method: cost to buy `shares` of `position`. Limit-aware.

        Layer: 3, User-facing: NO

        For multi-choice markets, this returns DIRECT cost without auto-arb.
        This is intentional - when used within auto-arb calculations, we need
        the direct cost of the arb trade, not nested auto-arb.
        """
        if self._is_multi_choice_type():
            # Direct cost without auto-arb (but with limits!)
            return self._calculate_multi_choice_cost(
                answer_id, shares, position, include_auto_arbitrage=False
            )
        else:
            return self.calculate_buy_cost(answer_id, shares, position)

    def shares_for_cost(
        self, answer_id: str, amount: float, position: str
    ) -> float:
        """Protocol method: shares received for `amount` in a single pool.

        Layer: 3, User-facing: NO

        This is the RAW AMM calculation (with limits), NOT including auto-arb.
        Auto-arb is handled by calculate_purchase_with_arbitrage which calls this.

        For user-facing "shares for amount including auto-arb", use simulate_buy.
        """
        pool = self.get_pool(answer_id)
        Y = pool['YES']
        N = pool['NO']
        p = self._answer_p(answer_id)  # per-answer p (cpmm-multi-2); v1 -> 0.5

        # Check for limit orders
        order_book = self.get_order_book(answer_id)
        has_limits = order_book is not None and (order_book.limits_above or order_book.limits_below)

        if has_limits:
            # Layer-3 protocol: route through self-aware wrapper so balance
            # state is honored. (Layer 4 helper would skip clamping.)
            return self._pool_shares_for_cost(Y, N, amount, position, order_book, p=p)
        else:
            return shares_for_cost(Y, N, amount, position, p=p)

    def clone(self) -> "MarketSimulator":
        """Create an independent copy for hypothetical calculations.

        Layer: 3, User-facing: NO

        Protocol method: Returns a copy that can be mutated without affecting
        the original. Only mutable state (pools) is copied; immutable data is shared.

        Used by Layer 2 (auto-arb) to try different rebalancing scenarios.
        Optimized for the η search which creates ~12,000 clones.

        Returns:
            New MarketSimulator with independent mutable state
        """
        # PERFORMANCE: Use shallow copy + targeted deep copy instead of full deepcopy.
        # Only pools and probabilities are mutable; strings/IDs/etc are shared.
        # This is ~18x faster than copy.deepcopy(self.data).
        #
        # What gets mutated by apply_trade():
        # - answer['poolYes'], answer['poolNo'], answer['poolYES'], answer['poolNO']
        # - answer['pool']['YES'], answer['pool']['NO']
        # - answer['prob'], answer['probability']

        # Bypass __init__ to avoid deepcopy
        cloned = MarketSimulator.__new__(MarketSimulator)

        # Shallow copy top-level dict (shares strings like question, slug, id)
        cloned.data = dict(self.data)

        # Shallow copy answers list, then shallow copy each answer dict
        if 'answers' in self.data:
            cloned.data['answers'] = []
            for ans in self.data['answers']:
                ans_copy = dict(ans)
                # Deep copy only the nested mutable dicts
                if 'pool' in ans_copy and isinstance(ans_copy['pool'], dict):
                    ans_copy['pool'] = dict(ans_copy['pool'])
                if 'probChanges' in ans_copy and isinstance(ans_copy['probChanges'], dict):
                    ans_copy['probChanges'] = dict(ans_copy['probChanges'])
                cloned.data['answers'].append(ans_copy)

        # Copy scalar attributes (these reference immutable values, so sharing is safe)
        cloned.initial_data = None  # Clones don't need reset capability
        cloned.market_type = self.market_type
        cloned.market_id = self.market_id
        cloned.slug = self.slug
        cloned.question = self.question
        cloned.p = self.p

        # Share limit order data (read-only during clone lifetime)
        cloned._order_book = self._order_book
        cloned._limit_orders_raw = self._limit_orders_raw
        cloned._limit_orders_now_ms = self._limit_orders_now_ms
        # Share order book cache - it's keyed by answer_id and limit data is read-only
        # This avoids rebuilding OrderBook objects 290K times during binary search
        cloned._order_book_cache = self._order_book_cache
        # Share answer index - structure is preserved, only pool values change
        cloned._answer_index = self._answer_index

        # Copy consumption state (lightweight: typically 0-5 entries)
        # This is the critical fix: each clone gets independent limit consumption
        # state, so binary search iterations in calculate_shares_exact don't
        # share/corrupt each other's limit fills.
        cloned._limit_consumption = dict(self._limit_consumption)
        cloned._answers_with_consumption = set(self._answers_with_consumption)

        # Copy maker balance state. Same lifecycle as _limit_consumption:
        # carries forward parent's apply_trade decrements; binary-search clones
        # mutate their own copy locally inside buy_with_limits. Without this,
        # clones would share a dict reference and a hypothetical fill in one
        # iteration would leak balance state into the next.
        cloned._maker_balances = dict(self._maker_balances)

        return cloned

    def without_limits(self) -> "MarketSimulator":
        """Return an independent copy with all limit orders stripped.

        Layer: 3, User-facing: NO

        Yields the *limit-free counterfactual*: the same market state, but trades
        move only against the AMM (and auto-arb), as if no resting limit orders
        existed. Two uses:

        - **Intended-path detection**: a large resting limit at the current price
          can *pin* a price, so the realized post-trade prob doesn't move and
          movement-based limit detection misses it. Simulating the same trade
          here gives the price the trade *intended* to reach; a limit lying
          between the pre-trade prob and this intended prob was crossed.
        - **Oracle for tests**: divergence between this sim's result and the
          limit-aware sim's result is, by definition, "a limit affected the
          trade" — independent of any detection heuristic.

        See: tasks/limit_pin_detection/onboarding.md, docs/amm-invariants.md §8.

        Returns:
            New MarketSimulator with independent mutable state and no limits.
        """
        bare = self.clone()
        bare._order_book = None
        bare._limit_orders_raw = None
        bare._limit_orders_now_ms = None
        bare._order_book_cache = {}
        bare._limit_consumption = {}
        bare._answers_with_consumption = set()
        bare._maker_balances = {}
        return bare

    def apply_trade(self, answer_id: str, cost: float, position: str) -> None:
        """Apply a trade to internal state. MUTATES SELF.

        Layer: 3, User-facing: NO

        Protocol method: Updates pool state after buying shares. Use on
        cloned contexts during binary search, never on the original.

        Also tracks limit order consumption so that clones see the correct
        remaining limit capacity.

        Args:
            answer_id: The answer to trade in
            cost: Amount to spend (positive for buys)
            position: 'YES' or 'NO'
        """
        # Get order book and track limit consumption if available
        p = self._answer_p(answer_id)  # per-answer p (cpmm-multi-2); v1 -> 0.5
        order_book = self.get_order_book(answer_id)
        if order_book is not None and cost > 0 and (order_book.limits_above or order_book.limits_below):
            pool = self.get_pool(answer_id)
            y, n = pool['YES'], pool['NO']
            consumption_report: dict[str, float] = {}
            maker_amount_report: dict[str, float] = {}
            _, _, y_new, n_new = buy_with_limits(
                y, n, position, order_book, target_cost=cost, p=p,
                consumption_report=consumption_report,
                maker_balances=self._maker_balances if self._maker_balances else None,
                maker_amount_report=maker_amount_report,
            )
            new_pool = {'YES': y_new, 'NO': n_new}

            # Merge consumption into our tracking state
            for oid, mana in consumption_report.items():
                self._limit_consumption[oid] = self._limit_consumption.get(oid, 0.0) + mana
            if consumption_report:
                self._answers_with_consumption.add(answer_id)
                # Invalidate cached order book for this answer since consumption changed
                self._order_book_cache.pop(answer_id, None)

            # Merge per-maker mana into persistent balance state. No floor —
            # vendor's `currentBalanceByUserId` can go slightly negative;
            # the next fill's `<= 0` clamp gates the actual cancel.
            for uid, mana in maker_amount_report.items():
                if uid in self._maker_balances:
                    self._maker_balances[uid] = self._maker_balances[uid] - mana
        else:
            # No limits or non-positive cost — use standard path
            new_pool = self.pool_after_trade(answer_id, cost, position)

        # Update the answer's pool in self.data using O(1) index lookup
        answer = self._get_answer(answer_id)
        if answer is not None:
            # Update both formats
            answer['poolYes'] = new_pool['YES']
            answer['poolNo'] = new_pool['NO']
            answer['poolYES'] = new_pool['YES']
            answer['poolNO'] = new_pool['NO']
            if 'pool' in answer:
                answer['pool']['YES'] = new_pool['YES']
                answer['pool']['NO'] = new_pool['NO']
            # Update probability (per-answer p; v1 -> 0.5 == N / (Y + N))
            Y, N = new_pool['YES'], new_pool['NO']
            answer['prob'] = probability_from_pool(Y, N, p)
            answer['probability'] = answer['prob']

    # ============= Calculations (don't change state) =============

    def calculate_shares_for_amount(self, outcome: str, amount: float, position: str = "YES") -> float:
        """Calculate shares obtainable for given amount.

        Layer: 1, User-facing: YES

        Args:
            outcome: Outcome to buy
            amount: Amount to spend in mana
            position: "YES" or "NO" - position type (default: "YES")

        Returns:
            Number of shares obtainable (always positive)
        """
        if self.market_type == 'BINARY':
            return self._calculate_binary_shares_for_amount(position, amount)
        elif self._is_multi_choice_type():
            # Pass position through to handle both YES and NO uniformly
            return self._calculate_multi_choice_shares_for_amount(outcome, amount, position)
        else:
            raise NotImplementedError(f"Market type {self.market_type} not supported")

    def buy_to_probability(self, outcome: str, target_prob: float, position: str = "YES") -> Dict[str, float]:
        """Calculate shares and cost needed to reach target probability.

        Layer: 1, User-facing: YES

        This function answers: "How much do I need to spend to move the market to probability p?"

        Supports both binary and multi-choice markets. For multi-choice, uses Algorithm 3
        (Probability-Centric) from auto_arb_algorithms.tex for O(n log N) complexity.

        Args:
            outcome: Outcome to buy (answer text or ID for multi-choice)
            target_prob: Target probability in POSITION FRAME (0 < prob < 1)
                        - For position="YES": target P(YES)
                        - For position="NO": target P(NO)
            position: "YES" or "NO" - position to take (default: "YES")

        Returns:
            Dictionary with:
                - 'shares': Number of shares obtained (in target outcome)
                - 'cost': Amount spent (mana)
                - 'final_prob': Final probability in POSITION FRAME (should equal target_prob)
                               Returns P(YES) for YES position, P(NO) for NO position

        Raises:
            ValueError: If target_prob is outside valid range (0, 1) or violates
                       Manifold's CPMM probability limits [0.01, 0.99]

        Examples:
            >>> sim = handle.create_simulator()
            >>> # Buy YES to move P(YES) to 75%
            >>> result = sim.buy_to_probability("YES", 0.75, "YES")
            >>> print(f"Need ${result['cost']:.2f} to get {result['shares']:.2f} YES shares")
            >>> print(f"Final P(YES): {result['final_prob']:.4f}")

            >>> # Buy NO to move P(NO) to 40% (which means P(YES) becomes 60%)
            >>> result = sim.buy_to_probability("YES", 0.40, "NO")
            >>> print(f"Need ${result['cost']:.2f} to get {result['shares']:.2f} NO shares")
            >>> print(f"Final P(NO): {result['final_prob']:.4f}")  # Returns 0.40
        """
        if target_prob <= 0 or target_prob >= 1:
            raise ValueError(f"target_prob must be in range (0, 1), got {target_prob}")

        # Check Manifold's CPMM probability limits
        # Manifold enforces these limits on buy operations via the API
        # Attempting to buy beyond these limits results in partial fills
        if target_prob > MAX_CPMM_PROB or target_prob < MIN_CPMM_PROB:
            raise ValueError(
                f"target_prob must be in range [{MIN_CPMM_PROB}, {MAX_CPMM_PROB}] "
                f"due to Manifold's CPMM probability limits, got {target_prob:.6f}"
            )

        if self.market_type == 'BINARY':
            return self._buy_to_probability_binary(outcome, target_prob, position)
        elif self._is_multi_choice_type():
            return self._buy_to_probability_multi_choice(outcome, target_prob, position)
        else:
            raise NotImplementedError(f"Market type {self.market_type} not supported")

    def _buy_to_probability_binary(self, outcome: str, target_prob: float, position: str) -> Dict[str, float]:
        """Binary market implementation of buy_to_probability.

        Args:
            outcome: Outcome to buy (for binary markets, typically "YES")
            target_prob: Target probability IN POSITION FRAME
                        - For YES position: target P(YES)
                        - For NO position: target P(NO)
            position: "YES" or "NO" - which position to buy

        Returns:
            Dict with shares, cost, and final_prob (in position frame)

        Example:
            buy_to_probability("YES", 0.6, "YES") → Buy YES until P(YES)=60%
            buy_to_probability("YES", 0.6, "NO") → Buy NO until P(NO)=60% (P(YES)=40%)
        """
        from manifold.order_book import OrderBook

        pool = self.data.get('pool', {})
        if 'YES' not in pool or 'NO' not in pool:
            raise ValueError(f"Binary market {self.market_id} missing pool data (YES/NO)")

        Y = pool['YES']
        N = pool['NO']
        p = self.p

        # Get order book (may be None if no limits set)
        order_book = self.get_order_book()
        if order_book is None:
            order_book = OrderBook.empty()

        # Convert target_prob from POSITION frame to EVENT frame (P(YES))
        # - For YES position: target_prob is P(YES) - no conversion needed
        # - For NO position: target_prob is P(NO), so target P(YES) = 1 - target_prob
        if position.upper() == 'YES':
            target_yes_prob = target_prob
        else:
            target_yes_prob = 1.0 - target_prob

        # Use limit-aware calculation via Layer-3 wrapper (balance-aware).
        cost, shares = self._pool_cost_to_probability(
            Y, N, target_yes_prob, position, order_book, p=p
        )

        # Calculate final probability (for verification)
        # Create temporary copy to avoid modifying state
        temp_sim = self.copy()
        temp_sim.simulate_buy(outcome, cost, position)
        # For binary markets, pass outcome=None to avoid legacy behavior that treats
        # outcome="YES"/"NO" as position, which would ignore the position parameter
        final_prob = temp_sim.get_probability(None, position)

        # Verify final probability matches target (in position frame)
        if abs(final_prob - target_prob) > EPSILON:
            # This might fail due to numerical precision, so just warn
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(
                f"Probability target miss: target={target_prob:.10f}, "
                f"achieved={final_prob:.10f}, diff={abs(final_prob - target_prob):.2e}"
            )

        return {
            'shares': shares,
            'cost': cost,
            'final_prob': final_prob
        }

    def buy_to_probability_interval(
        self, outcome: str, target_prob: float, position: str = "YES"
    ) -> Dict[str, float]:
        """Calculate the INTERVAL of achievable shares at target probability.

        At non-limit prices, there's exactly one answer (min == max).
        At limit prices, there's an interval: [AMM only, AMM + full limit consumption].

        This is essential for equivalent set optimization where we need to know
        the range of possible shares at each candidate price.

        Args:
            outcome: Outcome to buy (answer text or ID for multi-choice)
            target_prob: Target probability in POSITION FRAME (0 < prob < 1)
            position: "YES" or "NO" - position to take (default: "YES")

        Returns:
            Dictionary with:
                - 'min_shares': Shares with AMM only (no limit consumption at target)
                - 'max_shares': Shares with AMM + full limit consumption at target
                - 'min_cost': Cost for min_shares
                - 'max_cost': Cost for max_shares
                - 'limit_shares': Additional shares available from limits at target price
                - 'is_limit_price': True if target_prob matches a limit order price
        """
        from manifold.order_book import OrderBook

        position = position.upper()

        # Convert target_prob from POSITION frame to EVENT frame (P(YES))
        if position == 'YES':
            target_event_prob = target_prob
            direction = "UP"
        else:
            target_event_prob = 1.0 - target_prob
            direction = "DOWN"

        if self.market_type == 'BINARY':
            # For BINARY: use cost_to_probability_with_limits directly
            # This doesn't enforce Manifold probability limits (0.01-0.99), which
            # is needed for optimizer binary search that may probe beyond limits
            pool = self.data.get('pool', {})
            Y = pool.get('YES', 0)
            N = pool.get('NO', 0)
            order_book = self.get_order_book()
            if order_book is None:
                order_book = OrderBook.empty()

            min_cost, min_shares = self._pool_cost_to_probability(
                Y, N, target_event_prob, position, order_book, p=self.p
            )
        else:
            # For MULTIPLE_CHOICE: use buy_to_probability which handles auto-arb
            # Note: this enforces Manifold probability limits, so optimizer will
            # catch ValueError for targets beyond 0.01-0.99
            base_result = self.buy_to_probability(outcome, target_prob, position)
            min_shares = base_result['shares']
            min_cost = base_result['cost']
            answer_id = self._resolve_answer_id(outcome)
            order_book = self.get_order_book(answer_id)
            if order_book is None:
                order_book = OrderBook.empty()

        # Check for limits at target price.
        # Use a balance-aware sum rather than `total_liquidity_at_price`'s
        # nominal capacity — phantom limits (cash-poor maker, see
        # `tasks/maker_balance_simulation/`) shouldn't widen the interval the
        # optimizer searches over since they won't actually fill.
        limit_shares = self._balance_aware_limit_shares_at(
            order_book, target_event_prob, direction
        )
        is_limit_price = limit_shares > 0

        if is_limit_price:
            # Max: min_shares plus the full limit AT target
            # Cost per share at limit price
            if position == 'YES':
                cost_per_limit_share = target_event_prob
            else:
                cost_per_limit_share = 1.0 - target_event_prob
            max_shares = min_shares + limit_shares
            max_cost = min_cost + limit_shares * cost_per_limit_share
        else:
            # Not a limit price: single-valued
            max_shares = min_shares
            max_cost = min_cost

        return {
            'min_shares': min_shares,
            'max_shares': max_shares,
            'min_cost': min_cost,
            'max_cost': max_cost,
            'limit_shares': limit_shares,
            'is_limit_price': is_limit_price,
        }

    def get_limits_in_range(
        self, p_min: float, p_max: float, position: str
    ) -> list:
        """Get limit orders in a probability range for a given position.

        Args:
            p_min: Minimum probability (in POSITION FRAME)
            p_max: Maximum probability (in POSITION FRAME)
            position: "YES" or "NO" - which position we're considering

        Returns:
            List of LimitOrder objects in the range, in order of crossing.

        Note:
            For YES position, moving from p_min to p_max means price going UP.
            For NO position, moving from p_min to p_max means P(NO) increasing,
            which means P(YES) going DOWN.
        """
        if self.market_type != 'BINARY':
            raise NotImplementedError(
                "get_limits_in_range only supports BINARY markets currently"
            )

        order_book = self.get_order_book()
        if order_book is None:
            return []

        position = position.upper()
        if position == 'YES':
            # Moving P(YES) from p_min to p_max (price going UP)
            return order_book.limits_in_range(p_min, p_max, "UP")
        else:
            # Moving P(NO) from p_min to p_max means P(YES) going from (1-p_min) to (1-p_max)
            # i.e., P(YES) going DOWN from (1-p_min) to (1-p_max)
            yes_start = 1.0 - p_min
            yes_end = 1.0 - p_max
            return order_book.limits_in_range(yes_start, yes_end, "DOWN")

    def _buy_to_probability_multi_choice(self, outcome: str, target_prob: float, position: str) -> Dict[str, float]:
        """Multi-choice market implementation of buy_to_probability (Probability-centric).

        Internal: Not user-facing. Called by buy_to_probability() for multi-choice markets.

        Algorithm: Probability-centric auto-arbitrage (Algorithm 3 from auto_arb_algorithms.tex)
        ---------------------------------------------------------------------------------------
        Given a target probability p_target, find cost C and shares S.

        Redemption Strategy: Strategy 1 (arb in n-1 others, not target)
        - YES position: η NO in (n-1) others → η(n-2) cash + η YES in target
        - NO position: η YES in (n-1) others → no redemption (can't form complete sets)

        Implementation:
        1. Target purchase OUTSIDE search: Move answer from p_current to p_target
        2. Binary search for η (rebalancing shares in OTHER answers) to restore Σp = 1.0
        3. Calculate total cost: c_primary + c_arb - redemption

        Complexity: O(n log N) where n = number of answers, N = precision bits
        - ONE binary search for η (log N iterations)
        - Each iteration: O(n) clone + apply operations (no nested searches)

        Manifold Reference:
        - No direct equivalent in Manifold API (Manifold uses dollar-centric)
        - Our implementation for optimizer convergence checking

        Layer: 2 (Auto-Arbitrage)
        - Calls into Layer 3 (LiquidityContext/self) for cost calculations
        - Does NOT implement AMM math directly

        Limit Order Awareness: YES (via clone + apply pattern)
        - Target purchase (Step 1): Uses cost_to_probability_with_limits
        - Rebalancing search (Step 2): Uses clone + ctx.cost_for_shares
        - Final calculation (Step 3): Uses clone + ctx.cost_for_shares
        - Caveat: Limit order reversibility issue applies (Section 6.1 of paper)

        Mathematical Correctness: Correct modulo limit order reversibility caveat.
        For non-auto-arb markets, uses limit-aware amm_core functions directly.

        Args:
            outcome: Answer text or ID to trade
            target_prob: Target probability IN POSITION FRAME
                        - For YES position: target P(answer)
                        - For NO position: target P(NOT answer)
            position: "YES" or "NO" - which position to buy

        Returns:
            Dict with shares, cost, and final_prob (in position frame)
        """
        # Convert target_prob from position frame to event frame for internal calculations
        target_prob_event = target_prob if position.upper() == 'YES' else 1 - target_prob

        # Find target answer using O(1) index lookup
        answers = self.data.get('answers', [])
        target_answer = self._get_answer(outcome)

        if target_answer is None:
            raise ValueError(f"Answer '{outcome}' not found in market {self.market_id}")

        answer_id = target_answer['id']

        # Check if this market has auto-arbitrage
        should_arb = self.data.get('shouldAnswersSumToOne', True)

        if not should_arb:
            # Non-auto-arb markets: each answer acts like an independent binary market
            # Use limit-aware amm_core function directly
            pool = self.get_pool(answer_id)
            Y, N = pool['YES'], pool['NO']
            order_book = self.get_order_book(answer_id)

            # Get cost and shares using limit-aware calculation
            if order_book is not None:
                cost, shares = self._pool_cost_to_probability(
                    Y, N, target_prob_event, position, order_book
                )
            else:
                cost = cost_to_probability(Y, N, target_prob_event, position)
                shares = shares_to_probability(Y, N, target_prob_event, position)

            # Get new pool state (limit-aware)
            new_pool = self.pool_after_trade(answer_id, cost, position)

            # Update answer pools in market data
            target_answer['poolYes'] = new_pool['YES']
            target_answer['poolNo'] = new_pool['NO']
            target_answer['poolYES'] = new_pool['YES']
            target_answer['poolNO'] = new_pool['NO']

            # Calculate final probability in position frame
            final_prob_event = new_pool['NO'] / (new_pool['YES'] + new_pool['NO'])
            final_prob = final_prob_event if position.upper() == 'YES' else 1 - final_prob_event

            return {
                'shares': shares,
                'cost': cost,
                'final_prob': final_prob
            }

        # ====================================================================
        # PROBABILITY-CENTRIC AUTO-ARBITRAGE (Algorithm 3 from the paper)
        # Complexity: O(n log N) with ONE binary search layer
        #
        # Key insight: Target purchase happens OUTSIDE the search loop.
        # Search is ONLY for redemption amount η.
        #
        # Uses clone+apply pattern for FULL limit-awareness.
        # ====================================================================
        current_prob = self.get_probability(answer_id)
        n_answers = len(answers)

        # Check if target is reachable
        if position.upper() == 'YES':
            if target_prob_event <= current_prob:
                raise ValueError(
                    f"Cannot reach {target_prob_event:.4f} by buying YES from {current_prob:.4f}"
                )
        else:
            if target_prob_event >= current_prob:
                raise ValueError(
                    f"Cannot reach {target_prob_event:.4f} by buying NO from {current_prob:.4f}"
                )

        # Get all answer IDs for iteration
        all_answer_ids = [a['id'] for a in answers]

        # -------------------- STEP 1: Target Purchase (OUTSIDE search) --------------------
        # Move target answer to p_target using limit-aware calculation
        target_pool = self.get_pool(answer_id)
        Y_k, N_k = target_pool['YES'], target_pool['NO']

        # For limit orders on target: use limit-aware calculation
        target_order_book = self.get_order_book(answer_id)
        if target_order_book is not None:
            c_primary, delta = self._pool_cost_to_probability(
                Y_k, N_k, target_prob_event, position, target_order_book
            )
        else:
            c_primary = cost_to_probability(Y_k, N_k, target_prob_event, position)
            delta = shares_to_probability(Y_k, N_k, target_prob_event, position)

        # Clone self and apply target purchase (LIMIT-AWARE)
        ctx_after_primary = self.clone()
        ctx_after_primary.apply_trade(answer_id, c_primary, position)

        # -------------------- STEP 2: Binary Search for η (ONE search) --------------------
        # Search for η arb shares in OTHER answers (not target) - Redemption Strategy 1
        # Uses clone+apply pattern for limit-awareness
        arb_direction = "NO" if position.upper() == "YES" else "YES"

        # Upper bound for η - conservative estimate
        max_eta = 10000.0

        def check_balance(eta: float) -> float:
            """Check if η yields Σp = 1. Returns (Σp - 1) for binary search.

            Uses clone+apply pattern: clones post-primary context, applies
            η arb trades to OTHER answers (not target), checks probability sum.
            """
            test_ctx = ctx_after_primary.clone()
            _apply_arb_trades(test_ctx, all_answer_ids, eta, arb_direction, skip_id=answer_id)
            return sum(test_ctx.get_probability(aid) for aid in all_answer_ids) - 1.0

        # Binary search for η
        low, high = 0.0, max_eta
        best_eta = 0.0

        for _ in range(50):  # Converges in ~20 iterations typically
            mid = (low + high) / 2

            if mid in (low, high):
                best_eta = mid
                break

            diff = check_balance(mid)

            # Binary search direction depends on position
            if position.upper() == "YES":
                # Buying YES in target → buying NO in others
                # More NO in others → lower Σp (others' probs decrease)
                if diff > 0:  # Σp > 1, need more arb to reduce others
                    low = mid
                else:  # Σp < 1, need less arb
                    high = mid
            else:
                # Buying NO in target → buying YES in others
                # More YES in others → higher Σp
                if diff > 0:  # Σp > 1, need less arb
                    high = mid
                else:  # Σp < 1, need more arb
                    low = mid

            best_eta = mid

        # Clamp near-zero to exactly zero
        if abs(best_eta) < 0.001:
            best_eta = 0.0

        # -------------------- STEP 3: Compute Final Result --------------------
        # Apply best_eta to get final costs (LIMIT-AWARE)
        final_ctx = ctx_after_primary.clone()
        c_arb = _apply_arb_trades(final_ctx, all_answer_ids, best_eta, arb_direction, skip_id=answer_id)

        # Redemption depends on position:
        # YES position: η NO in (n-1) others = η YES in target + η*(n-2) cash
        # NO position: η YES in (n-1) others = no redemption (can't form complete sets)
        redemption = best_eta * (n_answers - 2) if position.upper() == "YES" else 0

        # Total cost
        total_cost = c_primary + c_arb - redemption

        # Total shares = target shares + arb shares
        total_shares = delta + best_eta

        # Return final_prob in position frame
        final_prob = target_prob_event if position.upper() == 'YES' else 1 - target_prob_event

        return {
            'shares': total_shares,
            'cost': total_cost,
            'final_prob': final_prob
        }

    # ============= Simulations (DO change state) =============

    def simulate_buy(self, outcome: str, amount: float, position: str = "YES") -> Dict[str, Any]:
        """Simulate buying with given amount.

        Layer: 1, User-facing: YES. Use this to simulate a bet and update state.

        This MODIFIES the simulator's state!

        Args:
            outcome: Outcome to buy (answer text for multi-choice)
            amount: Amount to spend
            position: "YES" or "NO" - position to take (default: "YES")

        Returns:
            Trade result dictionary with shares bought and new probability
        """
        if self.market_type == 'BINARY':
            # Binary markets: use old flow
            shares = self.calculate_shares_for_amount(outcome, amount, position=position)
            self._update_binary_state(outcome, shares, position=position)

            return {
                'outcome': outcome,
                'amount': amount,
                'shares': shares,
                'newProb': self.get_probability(position=position),  # Return prob for position bought
                'fees': 0  # Simplified - no fees in basic simulator
            }

        elif self._is_multi_choice_type():
            # Multi-choice markets: use calculate_purchase_with_arbitrage directly
            # This gives us the correct final state including auto-arb
            should_arb = self.data.get('shouldAnswersSumToOne', True)

            # Find answer ID for limit check
            answer_id = None
            for a in self.data.get('answers', []):
                if a.get('text') == outcome or a.get('id') == outcome:
                    answer_id = a.get('id')
                    break

            if not should_arb:
                # No auto-arb (SET market) - use limit-aware simulation
                # buy_with_limits returns correct final pool state after limit fills
                answer = self._get_answer(outcome)
                Y = answer.get('poolYes') or answer.get('poolYES') or answer.get('pool', {}).get('YES')
                N = answer.get('poolNo') or answer.get('poolNO') or answer.get('pool', {}).get('NO')

                order_book = self.get_order_book(answer.get('id'))
                if order_book and (order_book.limits_above or order_book.limits_below):
                    # Use limit-aware calculation that returns correct final state
                    from .amm_core import buy_with_limits
                    p = self._p_of(answer)  # per-answer p (cpmm-multi-2); v1 -> 0.5
                    _cost, shares, final_Y, final_N = buy_with_limits(
                        Y, N, position, order_book, target_cost=amount, p=p,
                        maker_balances=self._maker_balances if self._maker_balances else None,
                    )
                    # Update pool state directly from limit-aware result
                    answer['poolYes'] = final_Y
                    answer['poolNo'] = final_N
                    answer['poolYES'] = final_Y
                    answer['poolNO'] = final_N
                    new_prob = probability_from_pool(final_Y, final_N, p)
                    answer['prob'] = new_prob
                    answer['probability'] = new_prob
                else:
                    # No limits - use original path
                    shares = self.calculate_shares_for_amount(outcome, amount, position=position)
                    self._update_multi_choice_state(outcome, shares, position=position)

                return {
                    'outcome': outcome,
                    'amount': amount,
                    'shares': shares,
                    'newProb': self.get_probability(outcome, position=position),
                    'fees': 0
                }

            # With auto-arb: use calculate_purchase_with_arbitrage
            # Context pattern handles limit orders automatically
            from .multi_choice import calculate_purchase_with_arbitrage

            # Find answer ID
            answer_id = None
            for a in self.data.get('answers', []):
                if a.get('text') == outcome or a.get('id') == outcome:
                    answer_id = a.get('id')
                    break

            if not answer_id:
                raise ValueError(f"Answer '{outcome}' not found")

            # Calculate trade with auto-arb using context (limit-aware)
            result = calculate_purchase_with_arbitrage(
                self, answer_id, amount, position
            )

            self._apply_arb_result_to_answers(result.new_pools, result.new_probabilities)

            # Get the YES probability from the result
            yes_prob = result.new_probabilities[answer_id]

            # Return the probability for the position bought
            final_prob = (1 - yes_prob) if position.upper() == 'NO' else yes_prob

            return {
                'outcome': outcome,
                'amount': amount,
                'shares': result.shares_bought,
                'newProb': final_prob,
                'fees': 0
            }
        else:
            raise NotImplementedError(f"Market type {self.market_type} not supported")

    def calculate_sell_proceeds(self, outcome: str, shares: float) -> float:
        """Calculate proceeds from selling YES shares in a multi-choice market.

        Sell = buy NO + redeem pairs at $1 per set. When applied equally to all
        outcomes, this destroys complete sets: proceeds = shares, pools unchanged
        (Theorem 9, docs/amm-invariants.md).

        Pure calculation (no state change). Limit-aware via calculate_buy_cost.

        Args:
            outcome: Answer text or ID to sell
            shares: Number of YES shares to sell (positive)

        Returns:
            Proceeds in mana (positive value)
        """
        cost_to_buy_no = self.calculate_buy_cost(outcome, shares, position="NO")
        return shares - cost_to_buy_no

    def simulate_sell(self, outcome: str, shares: float) -> Dict[str, Any]:
        """Simulate selling shares. MODIFIES state.

        Sell = buy NO + redeem pairs (Theorem 9, docs/amm-invariants.md).
        Uses calculate_shares_exact for both proceeds and state from a single
        computation, ensuring they come from the same code path.

        Args:
            outcome: Outcome to sell
            shares: Number of shares to sell

        Returns:
            Trade result dictionary
        """
        # Selling shares is the inverse of buying
        # For multi-choice, we only support selling YES shares (the rebalance use case)

        if self.market_type == 'BINARY':
            # Selling YES is like buying NO
            opposite = 'NO' if outcome.upper() == 'YES' else 'YES'
            # Note: This is simplified - real calculation is different
            amount_received = self.calculate_buy_cost(opposite, shares)
            self._update_binary_state(opposite, shares)
        else:
            # Multi-choice: selling YES = buy NO + redeem pairs
            # Use calculate_shares_exact to get cost AND final state from one computation
            from .multi_choice import calculate_shares_exact

            answer_id = self._resolve_answer_id(outcome)
            result = calculate_shares_exact(self, answer_id, shares, "NO")

            # Proceeds: buying NO costs result.cost, redeeming pairs returns shares * $1
            amount_received = shares - result.cost

            self._apply_arb_result_to_answers(result.new_pools, result.new_probabilities)

        return {
            'outcome': outcome,
            'shares': shares,
            'amount': amount_received,
            'newProb': self.get_probability(outcome)
        }

    # ============= Private Helper Methods =============

    def _calculate_probability_from_pool(self) -> Optional[float]:
        """Calculate YES probability from pool state.

        This is the SINGLE SOURCE OF TRUTH for probability calculations.

        Returns:
            YES probability as decimal, or None if insufficient data
        """
        if 'pool' not in self.data:
            return None

        pool = self.data['pool']

        if self.market_type == 'BINARY':
            yes_pool = pool.get('YES')
            no_pool = pool.get('NO')

            if yes_pool is None or no_pool is None:
                return None

            # Use the p value from the simulator (set in __init__)
            # General formula: prob = (p * NO) / ((1-p) * YES + p * NO)
            denominator = (1 - self.p) * yes_pool + self.p * no_pool
            if denominator == 0:
                return None

            return (self.p * no_pool) / denominator

        else:
            # Multi-choice doesn't use this method - each answer has its own probability
            return None

    def _calculate_binary_cost(
        self, position: str, shares: float, ignore_limits: bool = False
    ) -> float:
        """Calculate cost for binary market using CPMM formula.

        Layer: 1 (User-facing helper)
        Calls: Layer 3 (cost_for_shares_with_limits) or Layer 4 fallback
        Invariant: Must be inverse of _calculate_binary_shares_for_amount
            i.e., cost_for_shares(shares_for_cost(c)) == c

        Args:
            position: 'YES' or 'NO' - which position to buy
            shares: Number of shares to buy
            ignore_limits: If True, ignore limit orders even if set
        """
        pool = self.data.get('pool', {})

        if 'YES' in pool and 'NO' in pool:
            Y = pool['YES']
            N = pool['NO']
        else:
            raise ValueError(f"Binary market {self.market_id} missing pool data (YES/NO)")

        # Check if we should use limit-aware calculation
        order_book = None if ignore_limits else self.get_order_book()

        if order_book is not None:
            # Use limit-aware calculation via Layer-3 wrapper (balance-aware).
            return self._pool_cost_for_shares(Y, N, shares, position, order_book)
        else:
            # Use pure AMM calculation (now supports general p)
            return cost_for_shares(Y, N, shares, position, p=self.p)

    def _calculate_binary_shares_for_amount(
        self, position: str, amount: float, ignore_limits: bool = False
    ) -> float:
        """Calculate shares obtainable for amount in binary market.

        Layer: 1 (User-facing helper)
        Calls: Layer 3 (shares_for_cost_with_limits) or Layer 4 fallback
        Invariant: Must be inverse of _calculate_binary_cost
            i.e., shares_for_amount(cost_for_shares(s)) == s

        Args:
            position: 'YES' or 'NO' - which position to buy
            amount: Dollar amount to spend
            ignore_limits: If True, ignore limit orders even if set
        """
        pool = self.data.get('pool', {})

        if 'YES' in pool and 'NO' in pool:
            Y = pool['YES']
            N = pool['NO']
        else:
            raise ValueError(f"Binary market {self.market_id} missing pool data (YES/NO)")

        # Check if we should use limit-aware calculation
        # CRITICAL: Must match _calculate_binary_cost's limit-awareness to maintain
        # the round-trip invariant: shares_for_cost(cost_for_shares(s)) == s
        order_book = None if ignore_limits else self.get_order_book()

        if order_book is not None:
            # Use limit-aware calculation via Layer-3 wrapper (balance-aware).
            return self._pool_shares_for_cost(Y, N, amount, position, order_book)
        else:
            # Use pure AMM calculation
            return shares_for_cost(Y, N, amount, position, p=self.p)

    def _apply_binary_trade(
        self, position: str, shares: float
    ) -> tuple[float, float, float, float]:
        """Execute a binary trade and return cost plus new pool state.

        Layer: 2 (Pool operations)
        Calls: Layer 3 (buy_with_limits)

        This is the source of truth for how trades affect pool state. Limit order
        fills do NOT affect the AMM pool - only the AMM portion of the trade does.

        Args:
            position: 'YES' or 'NO' - which position to buy
            shares: Number of shares to buy

        Returns:
            Tuple of (total_cost, total_shares, y_new, n_new) where y_new/n_new
            reflect only the AMM portion of the trade (limit fills don't move pool).
        """
        pool = self.data.get('pool', {})
        if 'YES' not in pool or 'NO' not in pool:
            raise ValueError(f"Binary market {self.market_id} missing pool data")

        Y = pool['YES']
        N = pool['NO']
        order_book = self.get_order_book()

        if order_book is not None:
            # Use limit-aware calculation - returns correct pool state
            cost, actual_shares, y_new, n_new = buy_with_limits(
                Y, N, position, order_book, target_shares=shares, p=self.p,
                maker_balances=self._maker_balances if self._maker_balances else None,
            )
            return cost, actual_shares, y_new, n_new
        else:
            # No limits - use pure AMM formula for pool update
            cost = cost_for_shares(Y, N, shares, position, p=self.p)
            if position.upper() == 'YES':
                y_new = Y - shares + cost
                n_new = N + cost
            else:
                y_new = Y + cost
                n_new = N - shares + cost
            return cost, shares, y_new, n_new

    def _calculate_multi_choice_cost(
        self,
        outcome: str,
        shares: float,
        position: str = "YES",
        include_auto_arbitrage: Optional[bool] = None,
        ignore_limits: bool = False,
    ) -> float:
        """Calculate cost for multi-choice market.

        For markets with shouldAnswersSumToOne=true, uses the same approach as Market class:
        binary search to find the dollar amount that yields exactly the requested shares
        after auto-arbitrage.

        Args:
            outcome: Answer to buy shares in
            shares: Number of shares to buy (always positive)
            position: 'YES' or 'NO' - the type of shares to buy
            include_auto_arbitrage: Whether to include auto-arb cost
                                  (default: True if shouldAnswersSumToOne, else False)
            ignore_limits: If True, ignore limit orders even if set

        Returns:
            Total cost including auto-arbitrage if applicable (always positive)
        """
        # Determine if we should include auto-arbitrage
        if include_auto_arbitrage is None:
            # Check if this market should auto-arbitrage
            should_arb = self.data.get('shouldAnswersSumToOne', True)
            include_auto_arbitrage = should_arb

        if not include_auto_arbitrage:
            # Simple case - just return direct cost
            # Use O(1) index lookup instead of O(n) linear search
            answer = self._get_answer(outcome)
            if not answer:
                raise ValueError(f"Answer not found: {outcome}")
            answer_id = answer.get('id')

            # Get pool state for this answer
            # Handle both poolYes/poolNo and pool.YES/pool.NO formats
            if 'poolYes' in answer or 'poolYES' in answer:
                Y = answer.get('poolYes', answer.get('poolYES'))
                N = answer.get('poolNo', answer.get('poolNO'))
            else:
                pool = answer.get('pool', {})
                Y = pool.get('YES', 0)
                N = pool.get('NO', 0)

            if Y is None or N is None or Y == 0 or N == 0:
                raise ValueError(f"Answer '{outcome}' missing pool data (poolYes/poolNo or pool.YES/NO)")

            # Check if we should use limit-aware calculation
            order_book = None if ignore_limits else self.get_order_book(answer_id)

            p = self._p_of(answer)  # per-answer p (cpmm-multi-2); v1 -> 0.5
            if order_book is not None:
                # Use limit-aware calculation via Layer-3 wrapper (balance-aware).
                direct_cost = self._pool_cost_for_shares(Y, N, shares, position, order_book, p=p)
            else:
                # Use pure AMM calculation
                direct_cost = cost_for_shares(Y, N, shares, position, p=p)

            return direct_cost  # Negative for selling, positive for buying

        # With auto-arbitrage: use calculate_shares_exact from multi_choice.py
        # which goes through the LiquidityContext protocol. clone() now properly
        # tracks limit consumption (Phase 3 fix), so the earlier bug where
        # limit orders were shared between clones is resolved.
        from manifold.multi_choice import calculate_shares_exact

        # Resolve outcome to answer_id
        answer = self._get_answer(outcome)
        if not answer:
            raise ValueError(f"Answer not found: {outcome}")
        answer_id = answer.get('id', outcome)

        result = calculate_shares_exact(self, answer_id, shares, position)
        return result.cost

    def _calculate_multi_choice_shares_for_amount(self, outcome: str, amount: float, position: str = "YES") -> float:
        """Calculate shares for amount in multi-choice market.

        For markets with auto-arbitrage, use the same function as Market class.
        Otherwise, use binary search to invert the cost function.

        When limit orders are present for this answer, uses limit-aware calculation
        which provides bonus shares at the limit price.

        Args:
            outcome: Answer to buy shares in
            amount: Amount to spend
            position: "YES" or "NO" - position type (default: "YES")
        """
        # Find answer and its ID
        answer = None
        answer_id = None
        for a in self.data.get('answers', []):
            if a.get('text') == outcome or a.get('id') == outcome:
                answer = a
                answer_id = a.get('id')
                break

        if not answer:
            raise ValueError(f"Answer '{outcome}' not found")

        # Check if this market should auto-arbitrage
        should_arb = self.data.get('shouldAnswersSumToOne', True)

        if should_arb:
            # With auto-arb: use calculate_purchase_with_arbitrage
            # Context pattern handles limit orders automatically
            from .multi_choice import calculate_purchase_with_arbitrage

            result = calculate_purchase_with_arbitrage(
                self, answer_id, amount, position
            )
            return result.shares_bought
        else:
            # No auto-arbitrage - use limit-aware calculation if limits exist
            order_book = self.get_order_book(answer_id)
            has_limits = order_book is not None and (order_book.limits_above or order_book.limits_below)

            if has_limits:
                # Use limit-aware calculation
                # Support all pool field formats (poolYes/poolNo, poolYES/poolNO, pool.YES/pool.NO)
                Y = answer.get('poolYes') or answer.get('poolYES') or answer.get('pool', {}).get('YES')
                N = answer.get('poolNo') or answer.get('poolNO') or answer.get('pool', {}).get('NO')
                if Y is None or N is None:
                    raise ValueError(f"Answer '{outcome}' missing pool data")

                return self._pool_shares_for_cost(
                    Y, N, amount, position, order_book, p=self._p_of(answer)
                )

            # No limits - use binary search
            C = amount
            low, high = 0, 1000 + C * 10  # Upper bound
            for _ in range(50):  # Match Manifold's iteration limit
                mid = (low + high) / 2

                # Break once we've reached max precision (matching Manifold's implementation)
                if mid in (low, high):
                    break

                # Don't include auto-arb since it's disabled
                cost = self._calculate_multi_choice_cost(outcome, mid, position, include_auto_arbitrage=False)
                if cost < C:
                    low = mid
                else:
                    high = mid

            return (low + high) / 2

    def _update_binary_state(self, outcome: str, shares: float, position: str = "YES"):
        """Update binary market state after trade.

        Layer: 1 (User-facing)
        Calls: Layer 2 (_apply_binary_trade)

        Uses _apply_binary_trade to get the correct post-trade pool state.
        This is critical for limit order handling: limit fills do NOT affect
        the AMM pool, only the AMM portion of the trade does.

        Args:
            outcome: Which outcome (for binary, usually 'YES')
            shares: Number of shares
            position: 'YES' or 'NO' - what position to take
        """
        pool = self.data.get('pool', {})

        # Initialize pool if missing
        if 'YES' not in pool or 'NO' not in pool:
            total_liquidity = self.data.get('totalLiquidity', 100)
            pool['YES'] = total_liquidity / 2
            pool['NO'] = total_liquidity / 2
            self.data['pool'] = pool

        # Use Layer 2 method to get correct pool state after trade
        # This properly handles limit orders (limit fills don't move the pool)
        _cost, _actual_shares, y_new, n_new = self._apply_binary_trade(position, shares)

        # Set pool directly from the Layer 2 result
        pool['YES'] = y_new
        pool['NO'] = n_new

        # Update probability using single source of truth
        new_prob = self._calculate_probability_from_pool()
        if new_prob is None:
            raise ValueError("Failed to calculate new probability after trade")
        self.data['probability'] = new_prob
        self.data['prob'] = new_prob
        self.data['pool'] = pool

    def _update_multi_choice_state(self, outcome: str, shares: float, position: str = "YES"):
        """Update multi-choice market state after trade.

        For YES position: Y' = Y - δ + C, N' = N + C
        For NO position:  Y' = Y + C, N' = N - δ + C
        Then applies auto-arbitrage to maintain Σp = 1.0

        Note: Internally converts position to signed shares for calculation,
        but the public API uses position parameter with positive shares.
        """
        # Find and update the specific answer
        for answer in self.data.get('answers', []):
            if answer.get('text') == outcome or answer.get('id') == outcome:
                # Support all pool field formats (poolYes/poolNo, poolYES/poolNO, pool.YES/pool.NO)
                Y = answer.get('poolYes') or answer.get('poolYES') or answer.get('pool', {}).get('YES')
                N = answer.get('poolNo') or answer.get('poolNO') or answer.get('pool', {}).get('NO')

                if Y is None or N is None:
                    raise ValueError(f"Answer '{outcome}' missing pool data")

                # Calculate pool state after trade, limit-aware.
                # Must use buy_with_limits to get correct pool states — the
                # AMM formula with limit-aware cost is wrong because limit
                # fills absorb shares without changing the AMM pool.
                answer_id = answer.get('id')
                p = self._p_of(answer)  # per-answer p (cpmm-multi-2); v1 -> 0.5
                order_book = self.get_order_book(answer_id) if answer_id else None

                if order_book is not None:
                    _cost, _shares_out, new_Y, new_N = buy_with_limits(
                        Y, N, position, order_book, target_shares=shares, p=p,
                        maker_balances=self._maker_balances if self._maker_balances else None,
                    )
                else:
                    if position == "YES":
                        cost = cost_for_shares(Y, N, shares, "YES", p=p)
                        new_Y = Y - shares + cost
                        new_N = N + cost
                    else:
                        cost = cost_for_shares(Y, N, shares, "NO", p=p)
                        new_Y = Y + cost
                        new_N = N - shares + cost

                answer['poolYes'] = new_Y
                answer['poolNo'] = new_N
                answer['poolYES'] = new_Y
                answer['poolNO'] = new_N

                # Update probability (per-answer p; v1 -> 0.5 == N / (Y + N))
                new_prob = probability_from_pool(
                    answer['poolYes'], answer['poolNo'], p
                )
                answer['prob'] = new_prob
                answer['probability'] = new_prob
                break

        # Apply auto-arbitrage to maintain Σp = 1.0 (only if shouldAnswersSumToOne is true)
        if self.data.get('shouldAnswersSumToOne', True):
            self._apply_auto_arbitrage()

    def _apply_arb_result_to_answers(
        self,
        new_pools: Dict[str, Dict[str, float]],
        new_probabilities: Dict[str, float],
    ) -> None:
        """Apply auto-arb result (new pools + probabilities) to answer dicts.

        Writes all pool field formats unconditionally (poolYes/poolYES/poolNo/poolNO
        + prob/probability). If the answer has a nested 'pool' dict, updates that too.

        Uses _get_answer() for O(1) lookup.

        Args:
            new_pools: {answer_id: {'YES': float, 'NO': float}}
            new_probabilities: {answer_id: float}
        """
        for aid, pool in new_pools.items():
            answer = self._get_answer(aid)
            assert answer is not None, (
                f"Answer {aid} in arb result not found in market data"
            )
            answer['poolYes'] = pool['YES']
            answer['poolNo'] = pool['NO']
            answer['poolYES'] = pool['YES']
            answer['poolNO'] = pool['NO']
            if 'pool' in answer:
                answer['pool']['YES'] = pool['YES']
                answer['pool']['NO'] = pool['NO']
            answer['prob'] = new_probabilities[aid]
            answer['probability'] = new_probabilities[aid]

    def _get_pool_values(self, answer: Dict) -> tuple[float, float]:
        """Get YES and NO pool values, handling both flat and nested formats.

        Args:
            answer: Answer dict with pool data

        Returns:
            Tuple of (YES_pool, NO_pool)
        """
        if 'poolYes' in answer or 'poolYES' in answer:
            return (
                answer.get('poolYes', answer.get('poolYES', 0)),
                answer.get('poolNo', answer.get('poolNO', 0))
            )
        else:
            pool = answer.get('pool', {})
            return (pool.get('YES', 0), pool.get('NO', 0))

    def _apply_auto_arbitrage(self):
        """Apply auto-arbitrage to maintain Σp = 1.0 in multi-choice markets.

        Following Manifold's iterative approach:
        1. Calculate current Σp
        2. If not 1.0, find NO shares needed to rebalance
        3. Apply NO shares to all answers
        4. Repeat until convergence (extraMana < 0.01)

        This ensures the fundamental invariant that probabilities sum to 1.0,
        representing that exactly one outcome will resolve YES.
        """
        if not self._is_multi_choice_type():
            return  # Only multi-choice markets need auto-arbitrage

        answers = self.data.get('answers', [])
        if not answers:
            return

        # Manifold uses different epsilons for different checks:
        # - EPSILON (1e-8) for general floating point equality
        # - floatingArbitrageEqual uses 0.001 for checking if amounts are negligible
        # We need tight precision for Σp = 1.0 check, looser for "should we arbitrage"
        SUM_EPSILON = 1e-8  # For checking if Σp ≈ 1.0
        ARBITRAGE_EPSILON = 0.001  # For checking if arbitrage amount is negligible
        MAX_ITERATIONS = 10  # Prevent infinite loops

        for _iteration in range(MAX_ITERATIONS):
            # Calculate current probability sum
            current_sum = sum(a.get('prob', 0) for a in answers)

            # Check if we're close enough to 1.0
            if abs(current_sum - 1.0) < SUM_EPSILON:
                break

            # Find NO shares needed to rebalance
            no_shares = self._find_no_shares_for_sum_one(answers, current_sum)

            # If no_shares is effectively zero, we're done
            # Use the arbitrage epsilon here since this is about whether to do a trade
            if abs(no_shares) < ARBITRAGE_EPSILON and abs(current_sum - 1.0) < SUM_EPSILON:
                # We're done - no arbitrage needed and sum is close to 1.0
                break
            # Otherwise continue trying with the calculated no_shares

            # Apply NO shares to all answers using limit-aware protocol methods
            for answer in answers:
                answer_id = answer.get('id')
                if not answer_id:
                    continue

                # Check for valid pool
                pool = self.get_pool(answer_id)
                Y, N = pool['YES'], pool['NO']
                if Y <= 0 or N <= 0:
                    continue  # Skip answers with no liquidity

                if no_shares > 0:
                    # Buying NO shares - use limit-aware cost calculation
                    cost = self.cost_for_shares(answer_id, no_shares, "NO")
                    self.apply_trade(answer_id, cost, "NO")
                else:
                    # Negative NO shares means buying YES
                    yes_shares = -no_shares
                    cost = self.cost_for_shares(answer_id, yes_shares, "YES")
                    self.apply_trade(answer_id, cost, "YES")

            # Check if we've made progress
            new_sum = sum(a.get('prob', 0) for a in answers)
            if abs(new_sum - current_sum) < 1e-10:
                # Not making progress, stop to avoid infinite loop
                break

    def _find_no_shares_for_sum_one(self, temp_answers: List[Dict], current_sum: float) -> float:
        """Binary search to find NO shares needed to make probabilities sum to 1.0.

        Uses clone + apply_trade pattern for limit-aware calculations.

        Args:
            temp_answers: List of answer dicts (unused, kept for signature compat)
            current_sum: Current probability sum

        Returns:
            Number of NO shares to buy in each answer (can be negative)
        """
        # If already at 1.0, no arbitrage needed
        if abs(current_sum - 1.0) < EPSILON:
            return 0.0

        def prob_sum_after_no_shares(no_shares: float) -> float:
            """Calculate Σp after buying no_shares NO in each answer (limit-aware)."""
            ctx = self.clone()
            for answer_id in ctx.get_answer_ids():
                pool = ctx.get_pool(answer_id)
                if pool['YES'] <= 0 or pool['NO'] <= 0:
                    continue

                try:
                    if no_shares > 0:
                        # Buying NO shares
                        cost = ctx.cost_for_shares(answer_id, no_shares, "NO")
                        ctx.apply_trade(answer_id, cost, "NO")
                    elif no_shares < 0:
                        # Negative NO shares means buying YES
                        yes_shares = -no_shares
                        cost = ctx.cost_for_shares(answer_id, yes_shares, "YES")
                        ctx.apply_trade(answer_id, cost, "YES")
                except (ValueError, ZeroDivisionError):
                    # Skip if math fails (e.g., too many shares for pool)
                    continue

            return sum(ctx.get_probability(aid) for aid in ctx.get_answer_ids())

        # Determine search bounds
        if current_sum > 1.0:
            # Need to buy NO shares (positive) to reduce probabilities
            low, high = 0.0, 100.0
            # Find upper bound where sum < 1.0
            while high < 10000:
                test_sum = prob_sum_after_no_shares(high)
                if test_sum < 1.0:
                    break
                high *= 10
        else:
            # Need negative NO shares (buy YES) to increase probabilities
            high, low = 0.0, -100.0
            # Find lower bound where sum > 1.0
            while low > -10000:
                test_sum = prob_sum_after_no_shares(low)
                if test_sum > 1.0:
                    break
                low *= 10

        # Binary search for exact NO shares
        for _ in range(50):
            mid = (low + high) / 2

            # Check if we've reached max precision
            if mid in (low, high):
                break

            test_sum = prob_sum_after_no_shares(mid)

            # Adjust bounds based on result
            if test_sum > 1.0:
                # Need more NO shares to reduce sum
                low = mid
            else:
                # Need fewer NO shares (or more negative)
                high = mid

        return (low + high) / 2

    def _resolve_outcome_to_answer(self, outcome: str) -> dict:
        """Resolve outcome text/ID to answer dict."""
        for answer in self.data['answers']:
            if answer.get('text') == outcome or answer.get('id') == outcome:
                return answer
        raise ValueError(f"Answer not found: {outcome}")

    def calculate_multi_answer_purchase_cost(self, outcomes: List[str], shares_per_outcome: float, position: str = "YES") -> float:
        """Calculate total cost to buy equal shares in multiple answers.

        Layer: 1, User-facing: YES. Use to query multi-bet cost.

        This properly handles the auto-arbitrage rebalancing for multi-choice markets.
        When buying YES shares in multiple answers, the market automatically:
        1. Sells YES shares in the selected answers
        2. Buys NO shares in ALL answers to maintain sum = 100%

        Args:
            outcomes: List of answer texts/IDs to buy
            shares_per_outcome: Shares to buy in each
            position: "YES" or "NO" position type (default: "YES")

        Returns:
            Total cost in mana including auto-arbitrage effects

        Raises:
            ValueError: If called on a set market (shouldAnswersSumToOne=false)
        """
        if not self._is_multi_choice_type():
            raise ValueError("Multi-answer purchase only for multi-choice markets")

        if not outcomes:
            return 0.0

        if shares_per_outcome <= 0:
            return 0.0

        # Check if this is a set market (shouldAnswersSumToOne=false)
        if not self.data.get('shouldAnswersSumToOne', True):
            raise ValueError(
                f"Multi-answer purchase not supported on set markets (shouldAnswersSumToOne=false).\n"
                f"Market: {self.slug}\n"
                f"Reason: Set market outcomes are independent - auto-arbitrage doesn't apply.\n"
                f"Manifold API behavior: Returns 400 'Not yet implemented'\n"
                f"Solution: Calculate cost for each outcome separately using calculate_buy_cost()"
            )

        # Use calculate_multi_shares_exact via LiquidityContext protocol (Layer 2).
        # clone() now properly tracks limit consumption (Phase 3 fix).
        from manifold.multi_choice import calculate_multi_shares_exact

        # Resolve outcome texts to answer IDs
        answer_ids = []
        for outcome in outcomes:
            answer = self._get_answer(outcome)
            if not answer:
                raise ValueError(f"Answer not found: {outcome}")
            answer_ids.append(answer.get('id', outcome))

        result = calculate_multi_shares_exact(self, answer_ids, shares_per_outcome, position)
        return result.total_cost

    def calculate_multi_shares_for_amount(
        self, outcomes: List[str], amount: float, position: str = "YES"
    ) -> MultiSharesResult:
        """Vendor-faithful AMOUNT → equal-shares for a multi-answer buy (does NOT mutate state).

        Layer: 1, User-facing: YES. The inverse of ``calculate_multi_answer_purchase_cost``:
        given the mana ``amount`` actually placed, return the equal shares Manifold's
        multi-bet API actually delivers per targeted answer.

        Manifold fills a multi-answer buy with a TRUNCATED iterative redemption loop
        (``while amountToBet > 0.01``), so it lands a hair short of the exact Option-2
        equilibrium ``calculate_multi_answer_purchase_cost`` inverts. Forecasting from this
        primitive — rather than committing the exact target — is what keeps forecast ==
        execution and avoids the per-leg shortfall halt. Pure-AMM (no limits/fees); the
        limit-aware extension is the deferred Strategy-1↔2 work. See
        ``tasks/multibet_sum_to_one_shortfall/onboarding.md``.

        Raises ValueError on non-MC or set markets (same contract as the cost query).
        """
        if not self._is_multi_choice_type():
            raise ValueError("Multi-answer buy only for multi-choice markets")
        if not outcomes:
            raise ValueError("outcomes must be non-empty")
        if amount <= 0:
            raise ValueError(f"amount must be positive, got {amount}")
        if not self.data.get('shouldAnswersSumToOne', True):
            raise ValueError(
                "Multi-answer buy not supported on set markets (shouldAnswersSumToOne=false)"
            )

        from manifold.multi_choice import (
            calculate_multi_shares_for_amount as _shares_for_amount,
        )

        answer_ids = []
        for outcome in outcomes:
            answer = self._get_answer(outcome)
            if not answer:
                raise ValueError(f"Answer not found: {outcome}")
            answer_ids.append(answer.get('id', outcome))

        return _shares_for_amount(self, answer_ids, amount, position)

    def simulate_multi_answer_buy(
        self,
        outcomes: List[str],
        shares_per_outcome: float,
        track_price_ranges: bool = False
    ) -> Dict[str, Any]:
        """Simulate buying equal shares in multiple outcomes atomically (multi-bet).

        Layer: 1, User-facing: YES. Use for multi-bet atomic simulation.

        This matches the multi-bet API behavior:
        1. All trades calculated against initial state
        2. Auto-arbitrage applied once at the end
        3. State updated atomically

        This MODIFIES the simulator's state!

        Args:
            outcomes: List of answer texts/IDs to buy
            shares_per_outcome: Shares to buy in each
            track_price_ranges: If True, track intermediate price ranges for all answers
                                (useful for limit order detection in multi-bet trades)

        Returns:
            Trade result with total cost and new probabilities.
            If track_price_ranges=True, includes 'price_ranges' dict mapping answer text
            to {initial, after_yes_purchase, final, min, max} prices.
        """
        if not self._is_multi_choice_type():
            raise ValueError("Multi-answer buy only for multi-choice markets")

        if not outcomes:
            return {"cost": 0.0, "shares": 0.0, "outcomes": []}

        if shares_per_outcome <= 0:
            return {"cost": 0.0, "shares": 0.0, "outcomes": []}

        # Step 1: Resolve outcome texts to answer IDs
        from manifold.multi_choice import calculate_multi_shares_exact

        answer_ids = []
        id_to_text: dict[str, str] = {}
        for outcome in outcomes:
            ans = self._get_answer(outcome)
            if not ans:
                raise ValueError(f"Answer not found: {outcome}")
            aid = ans.get('id', outcome)
            answer_ids.append(aid)
            id_to_text[aid] = outcome

        # Step 2: Delegate to L2 (always track price_ranges internally — cheap)
        l2_result = calculate_multi_shares_exact(
            self, answer_ids, shares_per_outcome, "YES",
            track_price_ranges=True,
        )

        # Diagnostic logging — if a probability prediction error recurs in production,
        # these logs (at DEBUG level) provide the L2 result needed to build a snapshot test.
        import logging
        _logger = logging.getLogger(__name__)
        if _logger.isEnabledFor(logging.DEBUG):
            for aid in answer_ids:
                pr = l2_result.price_ranges[aid]
                _logger.debug(
                    f"simulate_multi_answer_buy L2: {id_to_text.get(aid, aid)}: "
                    f"initial={pr['initial']:.6f} → after_primary={pr['after_yes_purchase']:.6f} "
                    f"→ final={pr['final']:.6f} (range [{pr['min']:.6f}, {pr['max']:.6f}])"
                )
            prob_sum = sum(l2_result.new_probabilities.values())
            _logger.debug(
                f"simulate_multi_answer_buy L2: cost={l2_result.total_cost:.4f}, "
                f"Σp={prob_sum:.10f}, slug={self.slug}"
            )

        self._apply_arb_result_to_answers(l2_result.new_pools, l2_result.new_probabilities)

        # Step 4: Build outcome_results from L2's price_ranges
        outcome_results = []
        for outcome in outcomes:
            ans = self._get_answer(outcome)
            aid = ans.get('id', outcome)
            pr = l2_result.price_ranges[aid]
            outcome_results.append({
                "outcome": outcome,
                "shares": shares_per_outcome,
                "initial_prob": pr['initial'],
                "prob_before_arb": pr['after_yes_purchase'],
                "final_prob": pr['final'],
            })

        result = {
            "cost": l2_result.total_cost,
            "shares_per_outcome": shares_per_outcome,
            "total_shares": shares_per_outcome * len(outcomes),
            "outcomes": outcome_results,
            "simulated": True
        }

        # Step 5: Convert answer-ID-keyed price_ranges to answer-text-keyed if requested
        if track_price_ranges:
            text_price_ranges = {}
            for answer in self.data.get('answers', []):
                aid = answer.get('id')
                answer_text = answer.get('text', aid)
                if aid in l2_result.price_ranges:
                    text_price_ranges[answer_text] = l2_result.price_ranges[aid]
            result["price_ranges"] = text_price_ranges

        return result

    def calculate_buy_cost(
        self, outcome: str, shares: float, position: str = "YES", ignore_limits: bool = False
    ) -> float:
        """Calculate exact cost to buy a specific number of shares.

        Layer: 1, User-facing: YES. Use this to query cost without mutating state.

        This uses the quadratic formula to solve for the amount needed.
        For binary markets with p=0.5, this simplifies nicely.

        Args:
            outcome: The outcome to buy ('YES', 'NO', or answer text for multi-choice)
            shares: Target number of shares to buy (positive)
            position: "YES" or "NO" for the position type (default: "YES")
            ignore_limits: If True, ignore limit orders even if set (default: False)

        Returns:
            Cost in mana to buy exactly that many shares (always positive)
        """
        if shares <= 0:
            return 0.0

        # For multi-choice markets, pass position parameter
        if self._is_multi_choice_type():
            return self._calculate_multi_choice_cost(
                outcome, shares, position, ignore_limits=ignore_limits
            )

        # For binary markets, use the internal method which handles limits
        if position not in ['YES', 'NO']:
            raise ValueError(f"Invalid position '{position}'. Must be 'YES' or 'NO' for binary markets")

        return max(0, self._calculate_binary_cost(position, shares, ignore_limits=ignore_limits))

    def __repr__(self) -> str:
        """String representation."""
        return f"MarketSimulator('{self.question[:50]}...', type={self.market_type})"
