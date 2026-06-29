"""Liquidity context protocol for multi-choice calculations.

LAYER ARCHITECTURE
==================

Layer 1 (User-facing): simulate_buy(), calculate_buy_cost(), simulate_multi_answer_buy()
    - Composite operations including auto-arb for multi-choice
    - What users call to execute trades

Layer 2 (Auto-arb): multi_choice.py
    - calculate_purchase_with_arbitrage(): Dollar-centric (given $C, how many shares?)
    - calculate_shares_exact(): Share-centric single-answer (given S shares, what cost?)
    - calculate_multi_shares_exact(): Share-centric multi-answer (given S shares in N answers, cost?)
    - Orchestrates multi-answer trades to maintain Σp = 1.0
    - Works on CLONED contexts, never modifies original
    - Uses Layer 3 for per-answer operations (treats liquidity as black box)

Note on strategy divergence: Strategy 1 (dollar-centric) and Strategy 2
(share-centric) can disagree when limit orders are present because limits
are consumed in different orders. This is fundamental, not a bug. The API
uses Strategy 1 (calculate_purchase_with_arbitrage via simulate_buy).

Layer 3 (This protocol): LiquidityContext
    - SINGLE-POOL operations with optional limit order support
    - NO auto-arb - that's Layer 2's job
    - Limit-aware: uses limit orders if available
    - Supports clone() for hypothetical calculations

Layer 4 (Primitives): amm_core functions
    - Pure math: cost_for_shares, pool_after_trade, etc.
    - No state, no limits, just formulas

CRITICAL: Layer 2 must NEVER call Layer 4 directly. All per-answer operations
go through Layer 3 (this protocol), which handles limit order awareness.

Limit consumption tracking: clone() MUST create independent limit consumption
state. apply_trade() MUST track which limits are consumed. get_order_book()
(or cost_for_shares/shares_for_cost) MUST subtract consumed amounts from
available limit sizes. Without this, binary search iterations in Layer 2
independently "fill" the same limits, causing Σp divergence.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LiquidityContext(Protocol):
    """Protocol for SINGLE-POOL liquidity calculations (Layer 3).

    IMPORTANT: These are primitive operations on individual pools.
    They do NOT include multi-choice auto-arb - that's Layer 2's job
    (calculate_purchase_with_arbitrage in multi_choice.py).

    Implementations:
    - MarketSimulator: Full implementation with limit order support
    - PoolsContext: Simple adapter for raw pools dict (no limits)

    Key features:
    - Single-pool: Operate on one answer at a time
    - Limit-aware: Use limit orders if available (implementation-dependent)
    - Clonable: clone() creates independent copy for hypothetical calculations
    - Mutable: apply_trade() updates internal state

    Query methods (no mutation):
    - get_answer_ids(), get_probability(), get_pool()
    - cost_for_shares(), shares_for_cost(), pool_after_trade()

    Mutation methods:
    - apply_trade(): Modify state in place
    - clone(): Create independent copy (for clone-then-mutate pattern)
    """

    def get_answer_ids(self) -> list[str]:
        """Get all answer IDs in the market."""
        ...

    def get_probability(self, answer_id: str) -> float:
        """Get current probability for an answer."""
        ...

    def probability_for(self, answer_id: str, pool: dict[str, float]) -> float:
        """Probability of a HYPOTHETICAL pool for `answer_id`, using its p weight.

        `get_probability` reads the answer's CURRENT pool. This variant prices a
        pool the caller computed (e.g. via `pool_after_trade`) without mutating
        state -- needed by Layer 2's auto-arb, which evaluates candidate pools.
        For cpmm-multi-1 (p=0.5) this is just NO/(YES+NO); the method keeps the
        per-answer p out of Layer 2.

        Args:
            answer_id: The answer whose p weight applies.
            pool: {'YES': float, 'NO': float} to price.
        """
        ...

    def get_pool(self, answer_id: str) -> dict[str, float]:
        """Get current pool state for an answer.

        Returns:
            {'YES': float, 'NO': float}
        """
        ...

    def cost_for_shares(
        self, answer_id: str, shares: float, position: str
    ) -> float:
        """Cost to buy `shares` in ONE pool (no auto-arb).

        Single-pool operation. Limit-aware if implementation supports it.

        Args:
            answer_id: The answer to buy in
            shares: Number of shares to buy (positive)
            position: 'YES' or 'NO'

        Returns:
            Cost in mana (always positive for buying)
        """
        ...

    def shares_for_cost(
        self, answer_id: str, amount: float, position: str
    ) -> float:
        """Shares received for `amount` in ONE pool (no auto-arb).

        Single-pool operation. Limit-aware if implementation supports it.

        Args:
            answer_id: The answer to buy in
            amount: Amount to spend (positive)
            position: 'YES' or 'NO'

        Returns:
            Number of shares received
        """
        ...

    def pool_after_trade(
        self, answer_id: str, amount: float, position: str
    ) -> dict[str, float]:
        """Pool state after trade in ONE pool (no auto-arb).

        Single-pool, PURE (no mutation). Limit-aware if supported.

        Args:
            answer_id: The answer to trade in
            amount: Amount to spend (positive)
            position: 'YES' or 'NO'

        Returns:
            New pool state {'YES': float, 'NO': float}
        """
        ...

    def clone(self) -> LiquidityContext:
        """Create an independent copy for hypothetical calculations.

        The clone shares no state with the original - mutations to the clone
        do not affect the original, and vice versa. This includes limit order
        consumption state: filling a limit on a clone must not affect the
        original or other clones.

        Used by Layer 2 (auto-arb) to try different rebalancing scenarios
        without affecting the original context.

        Returns:
            New LiquidityContext with identical state
        """
        ...

    def apply_trade(self, answer_id: str, cost: float, position: str) -> None:
        """Apply a trade to internal state. MUTATES SELF.

        Updates pool state after buying shares. This is the mutation
        counterpart to pool_after_trade() which returns new state without
        modifying self.

        Must track limit order consumption when limits are present. Subsequent
        calls to get_order_book() or cost_for_shares() should reflect any
        limits consumed by this trade.

        Used by Layer 2 (auto-arb) on CLONED contexts during binary search.

        Args:
            answer_id: The answer to trade in
            cost: Amount to spend (positive for buys)
            position: 'YES' or 'NO'
        """
        ...
