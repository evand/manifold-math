"""Runtime constants for numerical operations.

This module defines constants used at runtime for numerical comparisons.
For test-specific tolerances, see testing_constants.py.

## Philosophy

Limit prices from the API are exact (decimal values serialized as floats).
Pool probabilities are computed from pool state and have floating-point error.

When comparing these two types of values, we need tolerance. But when comparing
two limit prices (both exact), we can use exact equality.

See tasks/matching_engine_rewrite/onboarding.md for full design rationale.
"""

# =============================================================================
# Price comparison constants
# =============================================================================

PRICE_TOLERANCE = 1e-8
"""Tolerance for comparing computed pool probability to exact limit prices.

Matches Manifold's EPSILON constant (vendor/manifold/common/src/util/math.ts)
for consistent behavior with their floatingEqual/floatingGreaterEqual functions.

Use this ONLY for the specific question: "Is this limit at or behind our
current position?" This handles floating-point error in pool probability
calculation (prob = n / (y + n)).

Example:
    Pool computes probability as 0.18999999999999995
    Limit is at exact 0.19
    We need tolerance to detect we're "at" the limit.

DO NOT use for:
- Comparing two limit prices (both exact, use ==)
- Sorting limits by price (exact values, sort normally)
- Grouping limits at same price (exact equality OK)
"""

BUDGET_EPSILON = 1e-10
"""Threshold for "budget exhausted" checks.

When remaining_cost or remaining_shares falls below this value,
we consider it effectively zero and stop processing.

This is smaller than PRICE_TOLERANCE because budget values are
typically larger (dollars/shares vs probabilities 0-1).
"""

# =============================================================================
# Auto-arbitrage constants (from Manifold source)
# =============================================================================

MANIFOLD_ARB_EPSILON = 0.001
"""Threshold below which Manifold skips auto-arbitrage in multi-choice markets.

Source: vendor/manifold/common/src/calculate-cpmm-arbitrage.ts
(floatingArbitrageEqual function, epsilon=0.001)

When the binary search finds |η| < 0.001 arb shares needed, Manifold
clamps η to zero and skips the rebalancing step. This leaves a small
Σp residual bounded by: ΔΣp ≤ η * p(1-p) / T per answer, where T is
the pool total (Y + N). See docs/auto-arb-algorithms.tex for derivation.

Both auto-arb code paths must use this constant for consistency:
- multi_choice.py:calculate_purchase_with_arbitrage (Dollar-centric, Strategy 1)
- multi_choice.py:calculate_shares_exact / calculate_multi_shares_exact (Share-centric, Strategy 2)
"""
