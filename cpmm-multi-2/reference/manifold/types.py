"""Type definitions for the Manifold Markets library."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Literal, Optional, TypedDict

# Type aliases for common types
Outcome = Literal["YES", "NO"]
MarketType = Literal["BINARY", "MULTIPLE_CHOICE", "NUMERIC", "PSEUDO_NUMERIC"]
BetStatus = Literal["PENDING", "FILLED", "CANCELLED", "PARTIALLY_FILLED"]


class PoolDict(TypedDict):
    """Type for pool state dictionaries."""
    YES: float
    NO: float


class AnswerDict(TypedDict, total=False):
    """Type for answer data in multi-choice markets."""
    id: str
    text: str
    probability: float
    pool: PoolDict


class MarketDict(TypedDict, total=False):
    """Type for market API responses."""
    id: str
    creatorId: str
    creatorUsername: str
    creatorName: str
    createdTime: int
    closeTime: Optional[int]
    question: str
    url: str
    outcomeType: MarketType
    mechanism: str
    probability: Optional[float]  # For binary markets
    pool: Optional[PoolDict]  # For binary markets
    answers: Optional[List[AnswerDict]]  # For multi-choice markets
    totalLiquidity: Optional[float]
    volume: float
    volume24Hours: float
    isResolved: bool
    resolution: Optional[str]
    resolutionTime: Optional[int]
    resolutionProbability: Optional[float]


class BetDict(TypedDict, total=False):
    """Type for bet API responses."""
    id: str
    userId: str
    contractId: str
    marketId: str
    amount: float
    shares: float
    outcome: str
    probBefore: float
    probAfter: float
    createdTime: int
    fills: Optional[List['FillDict']]
    isFilled: bool
    isCancelled: bool
    limitProb: Optional[float]
    orderAmount: Optional[float]


class FillDict(TypedDict):
    """Type for fill data within bets."""
    amount: float
    shares: float
    timestamp: int
    matchedBetId: Optional[str]


class PositionDict(TypedDict):
    """Type for user position data."""
    contractId: str
    marketId: str
    userId: str
    outcome: str
    shares: float
    totalShares: Dict[str, float]  # outcome -> shares


class UserDict(TypedDict, total=False):
    """Type for user API responses."""
    id: str
    username: str
    name: str
    bio: Optional[str]
    balance: float
    totalDeposits: float
    profitCached: Dict[str, float]
    createdTime: int


class FreeLoanPositionDict(TypedDict):
    """Type for a position in the free loan response."""
    contractId: str
    answerId: Optional[str]
    payout: float
    invested: float
    freeLoanContribution: float


class FreeLoanAvailableDict(TypedDict, total=False):
    """Type for get-free-loan-available response."""
    available: float
    canClaim: bool
    lastClaimTime: Optional[int]
    positions: List[FreeLoanPositionDict]
    currentFreeLoan: float
    currentMarginLoan: float
    totalLoan: float
    maxLoan: float
    dailyLimit: float
    todayLoans: float
    todaysFreeLoan: float


class FreeLoanDistributionDict(TypedDict):
    """Type for a distribution in the claim response."""
    contractId: str
    answerId: Optional[str]
    amount: float


class FreeLoanClaimDict(TypedDict):
    """Type for claim-free-loan response."""
    success: bool
    amount: float
    distributed: List[FreeLoanDistributionDict]


@dataclass
class Market:
    """Represents a Manifold market with type safety."""
    id: str
    question: str
    creator_id: str
    creator_username: str
    created_time: datetime
    close_time: Optional[datetime]
    market_type: MarketType
    volume: float
    is_resolved: bool
    resolution: Optional[str]

    # Binary market fields
    probability: Optional[float] = None
    pool: Optional[PoolDict] = None

    # Multi-choice market fields
    answers: Optional[List[AnswerDict]] = None
    total_liquidity: Optional[float] = None

    @classmethod
    def from_dict(cls, data: MarketDict) -> 'Market':
        """Create Market from API response dictionary."""
        close_time_value = data.get('closeTime')
        return cls(
            id=data['id'],
            question=data['question'],
            creator_id=data['creatorId'],
            creator_username=data['creatorUsername'],
            created_time=datetime.fromtimestamp(data['createdTime'] / 1000),
            close_time=(datetime.fromtimestamp(close_time_value / 1000)
                        if close_time_value is not None else None),
            market_type=data['outcomeType'],
            volume=data['volume'],
            is_resolved=data['isResolved'],
            resolution=data.get('resolution'),
            probability=data.get('probability'),
            pool=data.get('pool'),
            answers=data.get('answers'),
            total_liquidity=data.get('totalLiquidity')
        )


@dataclass
class Bet:
    """Represents a bet on Manifold."""
    id: str
    user_id: str
    market_id: str
    amount: float
    shares: float
    outcome: str
    prob_before: float
    prob_after: float
    created_time: datetime
    is_filled: bool
    is_cancelled: bool
    fills: List[FillDict]
    limit_prob: Optional[float] = None
    order_amount: Optional[float] = None

    @classmethod
    def from_dict(cls, data: BetDict) -> 'Bet':
        """Create Bet from API response dictionary."""
        return cls(
            id=data['id'],
            user_id=data['userId'],
            market_id=data.get('marketId', data.get('contractId', '')),
            amount=data['amount'],
            shares=data['shares'],
            outcome=data['outcome'],
            prob_before=data['probBefore'],
            prob_after=data['probAfter'],
            created_time=datetime.fromtimestamp(data['createdTime'] / 1000),
            is_filled=data['isFilled'],
            is_cancelled=data['isCancelled'],
            fills=data.get('fills') or [],
            limit_prob=data.get('limitProb'),
            order_amount=data.get('orderAmount')
        )


@dataclass
class UserPosition:
    """Represents a user's holdings/position in a market.

    This represents shares that a user already owns (from API responses),
    not a tradeable position. For trading operations, use Position from
    manifold.position instead.
    """
    market_id: str
    user_id: str
    outcome: str
    shares: float
    total_shares: Dict[str, float]

    @classmethod
    def from_dict(cls, data: PositionDict) -> 'UserPosition':
        """Create UserPosition from API response dictionary."""
        return cls(
            market_id=data.get('marketId', data.get('contractId', '')),
            user_id=data['userId'],
            outcome=data['outcome'],
            shares=data['shares'],
            total_shares=data['totalShares']
        )


@dataclass
class User:
    """Represents a Manifold user."""
    id: str
    username: str
    name: str
    balance: float
    total_deposits: float
    profit_cached: Dict[str, float]
    created_time: datetime
    bio: Optional[str] = None

    @classmethod
    def from_dict(cls, data: UserDict) -> 'User':
        """Create User from API response dictionary."""
        return cls(
            id=data['id'],
            username=data['username'],
            name=data['name'],
            balance=data['balance'],
            total_deposits=data['totalDeposits'],
            profit_cached=data['profitCached'],
            created_time=datetime.fromtimestamp(data['createdTime'] / 1000),
            bio=data.get('bio')
        )
