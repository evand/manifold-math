"""Test data from M$1 @ 18% limit order on answer '48' in multi-choice market.

Market: PlRCt0hEEl (Senate seats 2026)
Bet ID: RSCI8uLSpL2R
Timestamps: before=1751436584, after=1751436684
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

# File paths relative to project root
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "test_fixtures" / "multi_choice_bet_1"
BEFORE_FILE = DATA_DIR / "market_before.json"
AFTER_FILE = DATA_DIR / "market_after.json"
BETS_FILE = DATA_DIR / "bets.json"

# Bet details
BET_ID = "RSCI8uLSpL2R"
ANSWER_ID = "tNU0nEsCgd"  # '48'
ANSWER_TEXT = "48"
BET_AMOUNT = 1.0
BET_LIMIT_PROB = 0.18
BET_SHARES = 6.185468080296236

def load_market_state(filename: Path) -> Dict:
    """Load market JSON data."""
    with open(filename) as f:
        return json.load(f)

def load_bet_data() -> Dict:
    """Load the specific bet from bets file."""
    with open(BETS_FILE) as f:
        bets = json.load(f)
    return next(b for b in bets if b['id'] == BET_ID)

def get_answer_states() -> Tuple[List[Dict], List[Dict]]:
    """Get before/after states for all answers.

    Returns:
        (before_answers, after_answers) - Lists of answer dicts with id, text, probability, pool
    """
    before = load_market_state(BEFORE_FILE)
    after = load_market_state(AFTER_FILE)
    return before['answers'], after['answers']

def get_pool_states() -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Get before/after pool states indexed by answer ID.

    Returns:
        (before_pools, after_pools) - Dicts mapping answer_id -> {YES: float, NO: float}
    """
    before_answers, after_answers = get_answer_states()

    before_pools = {a['id']: a['pool'] for a in before_answers}
    after_pools = {a['id']: a['pool'] for a in after_answers}

    return before_pools, after_pools

def get_probability_states() -> Tuple[Dict[str, float], Dict[str, float]]:
    """Get before/after probabilities indexed by answer ID.

    Returns:
        (before_probs, after_probs) - Dicts mapping answer_id -> probability
    """
    before_answers, after_answers = get_answer_states()

    before_probs = {a['id']: a['probability'] for a in before_answers}
    after_probs = {a['id']: a['probability'] for a in after_answers}

    return before_probs, after_probs

def verify_probability_sum() -> Tuple[float, float]:
    """Verify probability sums equal 1.0.

    Returns:
        (before_sum, after_sum)
    """
    before_probs, after_probs = get_probability_states()
    return sum(before_probs.values()), sum(after_probs.values())

def get_answer_48_state() -> Dict:
    """Get complete state info for answer '48' before and after."""
    before_pools, after_pools = get_pool_states()
    before_probs, after_probs = get_probability_states()

    return {
        'before': {
            'probability': before_probs[ANSWER_ID],
            'pool': before_pools[ANSWER_ID],
            'k': (before_pools[ANSWER_ID]['YES'] * before_pools[ANSWER_ID]['NO']) ** 0.5
        },
        'after': {
            'probability': after_probs[ANSWER_ID],
            'pool': after_pools[ANSWER_ID],
            'k': (after_pools[ANSWER_ID]['YES'] * after_pools[ANSWER_ID]['NO']) ** 0.5
        },
        'changes': {
            'probability': after_probs[ANSWER_ID] - before_probs[ANSWER_ID],
            'pool_yes': after_pools[ANSWER_ID]['YES'] - before_pools[ANSWER_ID]['YES'],
            'pool_no': after_pools[ANSWER_ID]['NO'] - before_pools[ANSWER_ID]['NO']
        }
    }

def print_test_data_summary():
    """Print a summary of the test data for verification."""
    print("\n=== TEST DATA SUMMARY ===")
    print("Market: PlRCt0hEEl")
    print(f"Bet: M${BET_AMOUNT} @ {BET_LIMIT_PROB*100}% on answer '{ANSWER_TEXT}'")
    print(f"Actual shares received: {BET_SHARES}")

    # Answer 48 state
    state = get_answer_48_state()
    print("\nAnswer '48' before:")
    print(f"  Probability: {state['before']['probability']:.18f}")
    print(f"  Pool YES: {state['before']['pool']['YES']:.15f}")
    print(f"  Pool NO: {state['before']['pool']['NO']:.15f}")
    print(f"  k: {state['before']['k']:.15f}")

    print("\nAnswer '48' after:")
    print(f"  Probability: {state['after']['probability']:.18f}")
    print(f"  Pool YES: {state['after']['pool']['YES']:.15f}")
    print(f"  Pool NO: {state['after']['pool']['NO']:.15f}")
    print(f"  k: {state['after']['k']:.15f}")

    print("\nChanges:")
    print(f"  Probability: {state['changes']['probability']:+.18f}")
    print(f"  Pool YES: {state['changes']['pool_yes']:+.15f}")
    print(f"  Pool NO: {state['changes']['pool_no']:+.15f}")

    # Probability sums
    before_sum, after_sum = verify_probability_sum()
    print("\nProbability sums:")
    print(f"  Before: {before_sum:.18f}")
    print(f"  After: {after_sum:.18f}")
    print(f"  Change: {after_sum - before_sum:.18e}")

if __name__ == "__main__":
    print_test_data_summary()
