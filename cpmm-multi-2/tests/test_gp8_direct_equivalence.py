"""GP8 — direct closed-form auto-arb == production nested search (p=0.5/no-limit).

cpmm-multi-2 Slice 2. `closed_form_arb.calculate_purchase_with_arbitrage_direct`
is an *independent* closed-form implementation of the dollar-centric auto-arb
(formulas written fresh from the GP theorems, not routed through `amm_core`).
This is the PR1 reference that gets ported to upstream TypeScript to replace
vendor's O(n log² N) nested binary search.

GP8 is the equivalence anchor: the direct path must reproduce the existing
production solver (`calculate_purchase_with_arbitrage`, which is vendor-faithful
at p=0.5 — see the calculation-vs-API regression tests) on pure-CPMM / no-limit
configs. Measured agreement across the sweep below is ~1e-13 on shares and
~1e-11 on pools (both machine precision), so the tolerances are set well above
the float noise floor yet far below any real algorithmic drift.

Only p=0.5 / no-limit is in scope (where v1 and the direct path provably agree);
general p and reversible limits are Slices 3+.
"""

import pytest
from manifold.closed_form_arb import (
    _buy_amount,
    _cost_for_shares,
    calculate_purchase_with_arbitrage_direct,
)
from manifold.multi_choice import PoolsContext, calculate_purchase_with_arbitrage
from tests.data.multi_choice_bet_1 import (
    ANSWER_ID as REAL_ANSWER_ID,
)
from tests.data.multi_choice_bet_1 import (
    BET_AMOUNT as REAL_BET_AMOUNT,
)
from tests.data.multi_choice_bet_1 import (
    BET_SHARES as REAL_BET_SHARES,
)
from tests.data.multi_choice_bet_1 import (
    get_pool_states,
)

# Tolerances: ~1e3x the measured worst-case agreement (shares 2.8e-13, pool
# 2.4e-11), still ~6 orders below any real algorithm bug.
SHARES_TOL = 1e-9
POOL_TOL = 1e-7
SIGMA_TOL = 1e-9


def _pools_from_probs(probs, L=100.0):
    """Balanced (p=0.5) pools at the given YES-probabilities: prob = N/(Y+N)."""
    pools = {}
    for i, q in enumerate(probs):
        N = L
        Y = N * (1 - q) / q
        pools[f"a{i}"] = {"YES": Y, "NO": N}
    return pools


# (prob vector, position). Mix of n, skew, uniform-but-non-trivial, and a 2-answer.
CONFIGS = [
    ([0.5, 0.5], "YES"),
    ([0.5, 0.5], "NO"),
    ([0.5, 0.3, 0.2], "YES"),
    ([0.5, 0.3, 0.2], "NO"),
    ([0.7, 0.2, 0.1], "YES"),
    ([0.7, 0.2, 0.1], "NO"),
    ([0.25, 0.25, 0.25, 0.25], "YES"),
    ([0.4, 0.25, 0.2, 0.15], "YES"),
    ([0.6, 0.15, 0.15, 0.1], "NO"),
    ([0.9, 0.05, 0.03, 0.02], "YES"),
]
AMOUNTS = [1.0, 10.0, 50.0, 200.0]


def _both(pools, target, amount, position):
    prod = calculate_purchase_with_arbitrage(
        PoolsContext({k: dict(v) for k, v in pools.items()}), target, amount, position
    )
    direct = calculate_purchase_with_arbitrage_direct(
        {k: dict(v) for k, v in pools.items()}, target, amount, position
    )
    return prod, direct


class TestDirectMatchesProduction:
    @pytest.mark.parametrize("probs,position", CONFIGS)
    @pytest.mark.parametrize("amount", AMOUNTS)
    def test_shares_and_pools_match(self, probs, position, amount):
        pools = _pools_from_probs(probs)
        prod, direct = _both(pools, "a0", amount, position)

        assert direct.shares_bought == pytest.approx(prod.shares_bought, abs=SHARES_TOL)
        assert direct.cost == pytest.approx(prod.cost, abs=1e-12)
        for aid in pools:
            assert direct.new_pools[aid]["YES"] == pytest.approx(
                prod.new_pools[aid]["YES"], abs=POOL_TOL
            )
            assert direct.new_pools[aid]["NO"] == pytest.approx(
                prod.new_pools[aid]["NO"], abs=POOL_TOL
            )

    @pytest.mark.parametrize("probs,position", CONFIGS)
    @pytest.mark.parametrize("amount", AMOUNTS)
    def test_probabilities_match(self, probs, position, amount):
        pools = _pools_from_probs(probs)
        prod, direct = _both(pools, "a0", amount, position)
        for aid in pools:
            assert direct.new_probabilities[aid] == pytest.approx(
                prod.new_probabilities[aid], abs=1e-9
            )


class TestDirectEquilibriumQuality:
    @pytest.mark.parametrize("probs,position", CONFIGS)
    @pytest.mark.parametrize("amount", AMOUNTS)
    def test_direct_restores_sigma_to_machine_precision(self, probs, position, amount):
        """GP5d in library code: the single bisection (no leftover-mana loop, no
        MANIFOLD_ARB_EPSILON clamp) drives Σp to ~machine precision."""
        pools = _pools_from_probs(probs)
        direct = calculate_purchase_with_arbitrage_direct(pools, "a0", amount, position)
        assert sum(direct.new_probabilities.values()) == pytest.approx(1.0, abs=SIGMA_TOL)

    def test_cost_equals_amount(self):
        """Dollar-centric: the user always spends exactly `amount`."""
        pools = _pools_from_probs([0.5, 0.3, 0.2])
        for amount in AMOUNTS:
            d = calculate_purchase_with_arbitrage_direct(pools, "a0", amount, "YES")
            assert d.cost == amount

    def test_fills_sign_convention(self):
        """Target fill signed by position; arb fills opposite. Matches production."""
        pools = _pools_from_probs([0.5, 0.3, 0.2])
        d = calculate_purchase_with_arbitrage_direct(pools, "a0", 50.0, "YES")
        assert d.fills["a0"] > 0  # bought YES in target
        assert d.fills["a1"] < 0 and d.fills["a2"] < 0  # NO arb in others


class TestDirectMatchesCapturedManifoldBet:
    """External anchor: the direct path reproduces a REAL captured Manifold bet
    (`tests/data/multi_choice_bet_1` — a $1 buy on a 10-answer market that
    actually executed for 6.185468080296236 shares). This is vendor ground
    truth, not our own solver. The bet crosses no resting maker, so it is in the
    p=0.5 / no-limit scope; production matches it to ~1e-9, direct to ~1e-12."""

    def test_direct_matches_real_bet_shares(self):
        before_pools, _ = get_pool_states()
        before_pools = {a: {"YES": p["YES"], "NO": p["NO"]} for a, p in before_pools.items()}
        direct = calculate_purchase_with_arbitrage_direct(
            before_pools, REAL_ANSWER_ID, REAL_BET_AMOUNT, "YES"
        )
        rel_err = abs(direct.shares_bought - REAL_BET_SHARES) / REAL_BET_SHARES
        assert rel_err < 1e-8, f"direct={direct.shares_bought}, manifold={REAL_BET_SHARES}, rel={rel_err:.2e}"
        assert sum(direct.new_probabilities.values()) == pytest.approx(1.0, abs=1e-12)

    def test_direct_matches_production_on_real_market(self):
        before_pools, _ = get_pool_states()
        before_pools = {a: {"YES": p["YES"], "NO": p["NO"]} for a, p in before_pools.items()}
        prod = calculate_purchase_with_arbitrage(
            PoolsContext({a: dict(p) for a, p in before_pools.items()}),
            REAL_ANSWER_ID, REAL_BET_AMOUNT, "YES",
        )
        direct = calculate_purchase_with_arbitrage_direct(
            before_pools, REAL_ANSWER_ID, REAL_BET_AMOUNT, "YES"
        )
        assert direct.shares_bought == pytest.approx(prod.shares_bought, abs=SHARES_TOL)


class TestDirectClosedForms:
    """The fresh single-pool closed forms invert each other and preserve the
    CPMM invariant (independent of the equilibrium loop)."""

    @pytest.mark.parametrize("position", ["YES", "NO"])
    @pytest.mark.parametrize("Y,N", [(100.0, 100.0), (40.0, 160.0), (250.0, 10.0)])
    def test_buy_amount_preserves_invariant(self, Y, N, position):
        _, pool = _buy_amount(Y, N, 25.0, position)
        assert pool["YES"] * pool["NO"] == pytest.approx(Y * N, rel=1e-12)

    @pytest.mark.parametrize("position", ["YES", "NO"])
    @pytest.mark.parametrize("Y,N", [(100.0, 100.0), (40.0, 160.0), (250.0, 10.0)])
    def test_cost_for_shares_round_trip(self, Y, N, position):
        """cost_for_shares(η) then buy_amount(cost) returns η shares, same pool."""
        eta = 12.5
        cost, pool_shares = _cost_for_shares(Y, N, eta, position)
        shares_back, pool_amount = _buy_amount(Y, N, cost, position)
        assert shares_back == pytest.approx(eta, abs=1e-9)
        assert pool_amount["YES"] == pytest.approx(pool_shares["YES"], abs=1e-7)
        assert pool_amount["NO"] == pytest.approx(pool_shares["NO"], abs=1e-7)


class TestDirectEdgeCases:
    def test_tiny_amount_no_arb(self):
        pools = _pools_from_probs([0.5, 0.3, 0.2])
        d = calculate_purchase_with_arbitrage_direct(pools, "a0", 1e-14, "YES")
        # No arb on a negligible amount: others untouched.
        assert d.fills["a1"] == 0.0 and d.fills["a2"] == 0.0

    def test_rejects_single_answer(self):
        with pytest.raises(ValueError, match="at least 2"):
            calculate_purchase_with_arbitrage_direct({"a0": {"YES": 1.0, "NO": 1.0}}, "a0", 10.0)

    def test_rejects_unknown_target(self):
        pools = _pools_from_probs([0.5, 0.5])
        with pytest.raises(ValueError, match="not in pools"):
            calculate_purchase_with_arbitrage_direct(pools, "missing", 10.0)

    def test_rejects_broken_market(self):
        # Both answers ~0.91 (YES=10, NO=100) → Σp ≈ 1.82, far from 1.
        with pytest.raises(ValueError, match="Broken market"):
            calculate_purchase_with_arbitrage_direct(
                {"a0": {"YES": 10.0, "NO": 100.0}, "a1": {"YES": 10.0, "NO": 100.0}},
                "a0", 10.0,
            )
