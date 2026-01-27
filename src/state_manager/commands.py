"""Command dataclasses for the command queue pattern.

Commands are immutable instructions emitted by state logic and consumed
by the CommandExecutor. This separates "what happened" (events) from
"what to do" (commands).

Key design principles:
- Commands are frozen dataclasses (immutable)
- Commands carry all data needed for execution
- Commands never mutate state directly
- Executor consumes commands and emits events back
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import time


class CommandType(Enum):
    """Types of commands the executor can process."""
    SUBMIT_ORDER = auto()
    CANCEL_ORDER = auto()
    FLATTEN = auto()


@dataclass(frozen=True)
class SubmitOrderCommand:
    """Command to submit a new order to a venue.

    The executor acquires the venue+market lock before submission
    and emits ORDER_SENT/ORDER_REJECTED events based on result.

    Attributes:
        attempt_id: Parent attempt ID
        venue: "kalshi" or "poly"
        market_id: Market/token ID
        side: "buy" or "sell" (trade direction: BUY/SELL)
        outcome: "YES" or "NO" (which outcome token is being traded)
        quantity: Order quantity
        price: Limit price
        client_order_id: Idempotent order ID (format: A-{attempt_suffix}-{VENUE}-{LEG})
        order_id: Local order ID (for event correlation)
        snapshot_id: Snapshot version used for this decision (for staleness check)
        created_at_ms: When command was created
    """
    attempt_id: str
    venue: str
    market_id: str
    side: str
    outcome: str  # "YES" or "NO" - which outcome token
    quantity: int
    price: float
    client_order_id: str
    order_id: str
    snapshot_id: tuple[int, int, int, int] = (0, 0, 0, 0)
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    @property
    def command_type(self) -> CommandType:
        return CommandType.SUBMIT_ORDER

    def is_stale(self, max_age_ms: int = 250) -> bool:
        """Check if command is older than max_age_ms."""
        age_ms = int(time.time() * 1000) - self.created_at_ms
        return age_ms > max_age_ms


@dataclass(frozen=True)
class CancelOrderCommand:
    """Command to cancel an existing order.

    The executor looks up the order, acquires the lock, and submits
    cancellation to the venue.

    Attributes:
        order_id: Local order ID to cancel
        venue_order_id: Venue-assigned order ID (if known)
        venue: "kalshi" or "poly"
        reason: Why the order is being cancelled
        created_at_ms: When command was created
    """
    order_id: str
    venue: str
    venue_order_id: Optional[str] = None
    reason: str = "user_requested"
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    @property
    def command_type(self) -> CommandType:
        return CommandType.CANCEL_ORDER


@dataclass(frozen=True)
class FlattenCommand:
    """Command to flatten (close) a position.

    Used when an attempt is imbalanced and needs hedging,
    or on shutdown/rollover to close all positions.

    Attributes:
        attempt_id: Parent attempt ID (if from an attempt)
        venue: "kalshi" or "poly" - which venue to flatten on
        quantity: Quantity to flatten (positive)
        side: "buy" or "sell" - direction to flatten
        market_id: Market to flatten
        reason: Why flattening (imbalance, shutdown, rollover)
        created_at_ms: When command was created
    """
    venue: str
    quantity: int
    side: str
    market_id: str
    attempt_id: Optional[str] = None
    reason: str = "imbalance"
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    @property
    def command_type(self) -> CommandType:
        return CommandType.FLATTEN


# Union type for all commands
Command = SubmitOrderCommand | CancelOrderCommand | FlattenCommand
